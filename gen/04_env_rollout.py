"""Step C（新用户线）：推荐环境 + 多轮闭环 + 决策 + 辅助字段。分片跑。

概设 §2.2 / §2.4.1 的闭环原文：「每轮交互中推荐环境提供候选内容，用户智能体生成
选择行为，系统记录交互结果并更新用户状态，更新后的用户状态继续参与下一轮交互」。
本文件就是这个循环，一轮不落：

  for round_no in 0..R-1:
      候选 = 推荐环境(用户当前状态)          # P(曝光v) ∝ 热度^γ · (1+λ·类别偏好)
      行为 = 用户智能体(候选)                # 轻量化打分 + 每用户 CTR 校准
      记录(交互记录表)                        # out/gen/interaction_records.jsonl
      用户状态 = 更新(用户状态, 行为)         # 类别偏好随实际点击漂移 -> 下轮曝光跟着变

**LLM 不出现在这个循环的任何位置**：50M 次决策 ÷ 2.7 req/s ≈ 214 天，物理上不可行；
且 FUSION.md §5 实测决策级 LLM 投票增量为零。语义信息通过原型画像折出的
sem_cat_pref / sem_cat_avert / sem_cap_pref 三个特征进入打分（与旧用户线同一条代码路径）。

硬约束（用户要求）：**每个用户至少 3 次交互**。本文件按 R 轮 × S 候选 = 每人 R·S 条
曝光，默认 5×20=100 条，远超该下限；同时对「点击数」也设了 MIN_CLICKS=3 的下限，
用兜底重采样保证——「交互」按最严口径（真点击）也满足。

产物（分片）：
  out/gen/shards/log_{shard}.csv               19 列，无表头（05 合并时统一加）
  out/gen/shards/records_{shard}.jsonl         内部留痕，对齐文档 §2.5.2

用法：
  <py> recsim/gen/04_env_rollout.py --shard 0 --nshard 100 [--plan _5000]
"""
import os, sys, json, csv, math, random, datetime, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from recsim import config as C
from recsim.sim import features as F

GEN_DATA = os.path.join(C.DATA, "gen")
OUT_GEN  = os.path.join(C.OUT, "gen")
SHARDS   = os.path.join(OUT_GEN, "shards")

N_ROUNDS   = 5          # 多轮闭环的轮数（= 模拟天数映射到 20220502-20220508 里的 5 天）
N_CAND     = 20         # 每轮候选数【上限】。注意这是个**截尾算子**而不是均值，见下。
#
# 每日条数：真实用户的日曝光量是重尾的（log 均值 3.474、σ 1.195，中位 37、P90 133），
# 交付规格「每天最多 20 条」对它做了一次**截尾** min(20, 真实条数)。§4 的同形状对照组
# 走的正是这个算子，实测结果：每天 16.84±5.60 条，其中 **69.5% 恰好等于 20**，
# 30.5% 不足 20；每人 5 天合计 84.2±19.7 条（P10/P90 = 53/100）。
#
# 前一版没有复现这个算子，而是直接抽了一个【未截尾】的对数正态日条数，
# 于是合成侧 94.8±61.7 条 —— 均值能靠 CAND_ANCHOR 硬掰回来，形状却完全不同
# （真实分布在 20 处有个大质量点 + 左尾，合成是一条光滑的重尾曲线，还长出 320 条的人）。
# 判别器抓的 n_cat_click / n_hour 这类【绝对计数】差异，根子就在这里。
#
# 现在照抄同一个算子：按用户活跃度抽【未截尾】的日条数（重尾），再 min(20, ·)。
# 这样行数分布是**按构造**对上的，不是调参调出来的；CAND_ANCHOR 也随之删掉——
# 均值不再是可自由拧动的旋钮，而是截尾的结果。
DAY_LN_MEAN = 3.474             # log(每天未截尾条数) 的总体均值（实测）
DAY_LNSD    = 0.864             # 同一用户日间波动的对数标准差（实测）
USER_LNSD   = 0.766             # 用户之间日均条数的对数标准差（实测）
LA_MEAN     = 7.0359            # donor 的 log(n_expose) 均值，用于把目标活跃度折成 z 分
LA_SD       = 0.7821            # donor 的 log(n_expose) 标准差（≈USER_LNSD，故 z 分几乎 1:1）
CAND_MIN    = 2                 # 每日条数下限（5 天 × 2 条仍足以凑够 MIN_CLICKS=3）
MIN_CLICKS = 3          # 硬下限：每人至少 3 次真实点击（用户明确要求「每人 3 个交互以上」）
GAMMA      = 0.85       # 热度指数，Step 0 由真实曝光幂律拟合，见 fit_gamma()
LAMBDA_MED = 9.0        # 类别偏好对曝光的放大系数（中位）。**按用户浮动**，见 lam_u：
LAMBDA_LNSD = 1.05      # 固定 λ 会让所有人的类别收敛速度一致，n_cat_click 方差偏低
PREF_TAU_MED = 0.72     # 偏好在加权前先做幂律回火 p^τ 再归一。**按用户浮动**。
PREF_TAU_LNSD = 0.30    # τ 固定会让「曝光 top1 占比」的用户间标准差塌到 0.04（真实 0.09）：
                        # 直接用 λ·p 会把曝光过度堆到【单一头部类别】：实测合成
                        # 曝光 top1 占比 0.300、真实只有 0.244，而「点击类别数」
                        # 却对得上——即形状错了而不是量错了，判别器抓的正是
                        # (n_cat_click, cat1_entropy_click) 这对联合量。τ<1 压平头部、
                        # 抬起中部，既保住类别数又还原熵。τ 本身也必须按用户变化——
                        # 「每个人被喂得一样宽」和「每个人段长一样」是同一类破绽。
DRIFT      = 0.25       # 状态更新：本轮实际点击类别分布对下轮偏好的混入权重
TZ = datetime.timezone(datetime.timedelta(hours=8))
SIM_DATES = [20220502, 20220503, 20220504, 20220505, 20220506, 20220507, 20220508]

# ---------------------------------------------------------------------------
# 以下四组常数全部来自对 2,000 个真实用户 / 310 万行日志的实测（见首轮 §4 判别器复盘）。
# 第一版把它们拍脑袋写成了「一天一个小时点 + normal(mean, 0.35)」，判别器 AUC 0.9997，
# 权重最大的三个特征正是 n_hour / play_ratio_std / hour_entropy —— 全是这里造出来的破绽。
# ---------------------------------------------------------------------------

# 会话结构：真实用户一天里在【若干个不同小时】各刷一段，不是「一天只在某一个整点刷」。
# 关键是**每天的小时数不是自由参数，而是当天条数的函数**：条数越多铺得越开，
# 但远不是线性的（n=5 时 2.33 小时，n=20 时 6.56 小时）。第二版让「段长」按用户
# 对数正态浮动、小时数由段长间接决定，结果每天小时数 4.47±3.62（真实 5.56±2.62）
# ——均值偏低、方差偏大，判别器的头号特征始终是 n_hour。
#
# 现在直接用同形状对照组实测的 **P(不同小时数 h | 当天条数 n)** 经验分布抽 h，
# 再把 n 条随机分配到 h 个小时里。段长因此是 n/h 的自然结果，不再需要单独的段长参数
# ——真实数据里段长本来就是「条数 ÷ 小时数」的派生量（实测 h=2 时 7.29 条、
# h=10 时 1.99 条，正是 n/h 的形状）。少一个自由参数，也少一处可以调错的地方。
# 表的每一行 = 该 n 档下 h=1,2,3,... 的观测频数（2,000 个真实用户 × 5 天）。
HOUR_GIVEN_N = {
     1: [200],
     2: [90, 64],
     3: [60, 72, 19],
     4: [55, 63, 38, 9],
     5: [42, 53, 44, 23, 2],
     6: [26, 61, 50, 29, 13, 3],
     7: [29, 52, 60, 36, 13, 5],
     8: [19, 43, 57, 32, 20, 8],
     9: [15, 44, 58, 47, 25, 9, 1, 1],
    10: [13, 43, 32, 24, 25, 13, 5, 1, 1],
    11: [9, 50, 57, 44, 30, 10, 8, 2],
    12: [12, 33, 35, 41, 25, 9, 3, 1],
    13: [7, 36, 40, 34, 26, 17, 6, 2, 1],
    14: [8, 23, 35, 37, 39, 16, 12, 2, 2, 1],
    15: [3, 20, 38, 33, 35, 14, 12, 3, 1, 1],
    16: [3, 15, 19, 32, 29, 14, 12, 7, 3, 1],
    17: [3, 17, 24, 27, 32, 11, 15, 1, 3, 2],
    18: [2, 10, 12, 28, 30, 20, 20, 5, 3, 1],
    19: [1, 7, 25, 28, 24, 21, 17, 5, 5, 2],
    20: [23, 171, 406, 724, 966, 1104, 1094, 957, 685, 410, 195, 69, 22, 3],
}
HOUR_P = {n: (np.arange(1, len(c) + 1), np.array(c, float) / sum(c))
          for n, c in HOUR_GIVEN_N.items()}

# 日内份额：一天的 n 条**不是均匀铺在 h 个小时里**的。实测（降序排列后）h=7 时
# 头一个小时就占全天 32%，均匀分配只有 14%。第三版用 multinomial(n, 1/h) 平均分，
# 于是「给定 n_hour 的 hour_entropy」系统性偏高（判别器权重 +2.05，配 n_hour −1.42
# 的反号——典型的「条件形状错了而边际对得上」）。这里直接用实测的份额剖面。
# h≥12 的天太少（<60），用 s=0.9 的 Zipf 外推。
DAY_SHARE = {
     1: [1.0],
     2: [0.7316, 0.2684],
     3: [0.5749, 0.2840, 0.1410],
     4: [0.4852, 0.2668, 0.1571, 0.0909],
     5: [0.4191, 0.2477, 0.1620, 0.1031, 0.0682],
     6: [0.3626, 0.2299, 0.1581, 0.1120, 0.0785, 0.0588],
     7: [0.3224, 0.2113, 0.1524, 0.1143, 0.0841, 0.0631, 0.0524],
     8: [0.2871, 0.1938, 0.1459, 0.1123, 0.0875, 0.0684, 0.0542, 0.0508],
     9: [0.2527, 0.1787, 0.1379, 0.1085, 0.0910, 0.0723, 0.0570, 0.0514, 0.0506],
    10: [0.2336, 0.1633, 0.1282, 0.1035, 0.0889, 0.0726, 0.0582, 0.0516, 0.0500, 0.0500],
    11: [0.2054, 0.1527, 0.1199, 0.1011, 0.0893, 0.0723, 0.0574, 0.0518, 0.0500, 0.0500,
         0.0500],
}
ZIPF_S = 0.9
def day_share(h):
    v = DAY_SHARE.get(h)
    if v is None:
        w = np.arange(1, h + 1, dtype=float) ** (-ZIPF_S)
        return w / w.sum()
    return np.array(v) / sum(v)
DAY_SHARE_P = {h: day_share(h) for h in range(1, 25)}

# 窗口小时分布：donor 的 hours 是【31 天】的边际，比任何 5 天窗口都平滑。
# 逐日直接从它抽，窗口内最热小时只占 17.3%（真实 21.4%）。按用户抽一次
# Dirichlet 扰动，等于「这个人这 5 天恰好偏爱哪几个时段」。
HOUR_DIR_ALPHA = 1.2

# 时长档同理：donor 的 dur_expo 也是 31 天边际，档位份额的用户间 σ 只有
# (0.057,0.035,0.027,0.027,0.072)，真实 5 天窗口是 (0.092,0.068,0.052,0.052,0.113)。
# α=10 的 Dirichlet 扰动把 σ 抬到 (0.078,0.066,0.055,0.055,0.094)，
# 剩下的差额由抽样本身的多项式噪声补上。α=6 会过冲中间两档。
DUR_DIR_ALPHA = 10.0

# 播放比例：强烈依赖视频时长档，短视频会重复播放（档0 有 87% 的点击 ratio>1），
# 长视频普遍看不完（档4 只有 15%）。旧实现对所有档位用同一个 normal(mean, 0.35)，
# 于是用户内 std 只有 0.34，真实是 0.72 —— 判别器一眼就能看出来。
# 实测各档（点击行）均值与标准差：
PR_SLOT_MEAN  = [1.663, 1.246, 0.912, 0.697, 0.436]
PR_SLOT_SIGMA = [0.609, 0.565, 0.516, 0.557, 0.770]   # 对数正态 σ，由实测 cv 反解
PR_GLOBAL_MEAN = 0.9916       # 24,170 个 donor 的 play_ratio_mean 总均值。
                              # 用它做分母，pr_scale 在 donor 群体上才恰好以 1 为中心；
                              # 曾误用「点击行加权均值 0.9198」，整体播放比例系统性高 8%。

# 未点击行的播放比例：11.6% 恰好为 0（划走前一帧都没播完），非零部分均值 0.144、
# 标准差 0.204（重尾）。旧实现是 uniform(0.02, 0.25)，均值凑巧接近但**没有零、没有尾巴**。
NOCLK_ZERO_RATE = 0.1159
NOCLK_LN_MED    = 0.0831      # 非零部分的对数正态中位数
NOCLK_LN_SIGMA  = 1.049

# 重复曝光：真实 100 条窗口内 distinct_ratio 均值 0.9885，19% 的用户有重复。
# 旧实现严格去重导致恒为 1.0000。这里留一个小概率不查重。
REEXPOSE_P = 0.012

OUT_COLS = ["user_id","video_id","date","hourmin","time_ms","is_click","is_like","is_follow",
            "is_comment","is_forward","is_hate","long_view","play_time_ms","duration_ms",
            "profile_stay_time","comment_stay_time","is_profile_enter","is_rand","tab"]

def hourmin_of(dt):
    """真实表的 hourmin 口径（实测 30 万行反推）：分钟位恒为 0，小时位是
    「四舍五入到最近的整点」—— 0-29 分不进位率 ≤0.6%，50-59 分进位率 87%，
    40-49 分 14.8%，是带噪的 round(minute/60)。这里按 minute>=45 进位实现，
    比 round 更贴合实测（45 分处 ≈ 50% 分界）。"""
    h = dt.hour + (1 if dt.minute >= 45 else 0)
    return (h % 24) * 100

def session_offsets(n, hours, nprng, rng):
    """把一天里的 n 条曝光排成【若干段连续刷】，返回每条的「当天第几秒」。

    小时数 h 由 P(h | n) 经验表抽（见 HOUR_GIVEN_N），不再由段长参数间接决定——
    真实数据里「一天铺开几个小时」强依赖于「这天刷了多少条」，把它拆成两个自由参数
    反而同时错了均值和方差（合成 4.47±3.62 vs 真实 5.56±2.62）。

    h 个小时按该用户的窗口小时分布**不放回**抽；n 条按【实测的日内份额剖面】
    分配到这 h 段（不是均匀分！头一段占 30% 上下，见 DAY_SHARE），
    段内每条间隔 40-90 秒，会自然跨过整点，复现真实的跨小时会话。
    末尾裁到 23:59:59 以免跨天——跨天会让 time_ms 反推出的日期与 date 列不符，撞断言 4。
    """
    hv, hp = HOUR_P[min(max(n, 1), 20)]
    h = int(nprng.choice(hv, p=hp))
    h = min(h, n, 24)
    # 不放回抽 h 个小时：按 hours 权重逐个抽，抽过的置零重新归一
    w = hours.copy()
    picked = []
    for _ in range(h):
        s = w.sum()
        if s <= 0: break
        k = int(nprng.choice(24, p=w / s))
        picked.append(k); w[k] = 0.0
    if not picked: picked = [int(nprng.choice(24, p=hours))]
    # n 条按实测份额剖面分到 len(picked) 段，每段至少 1 条。
    # 份额是按「降序名次」测的，所以先随机决定哪个小时当头一名（否则最热小时
    # 永远是该用户 hours 分布里概率最高的那个，跨天复现度会虚高）。
    m = len(picked)
    sh = DAY_SHARE_P[m].copy()
    nprng.shuffle(sh)
    out = []
    if n > m:
        extra = nprng.multinomial(n - m, sh)
    else:
        extra = np.zeros(m, int)
    for k, hh in enumerate(picked):
        base = hh * 3600 + rng.randrange(60) * 60 + rng.randrange(60)
        t = base
        for _ in range(1 + int(extra[k])):
            out.append(min(t, 86399))
            t += rng.randrange(40, 91)
    out.sort()
    return out[:n]

def play_ratio(slot, scale, nprng):
    """点击行的播放比例：按【时长档】取对数正态，再按用户个体缩放。

    实测各档差异极大——档0（<10s）87% 的点击 ratio>1（短视频循环播放），
    档4（≥80s）只有 15%。旧实现对所有档用同一个 normal(mean, 0.35)，
    用户内 std 只有 0.34，真实 0.72，判别器第二强的特征就是它。
    """
    mu = math.log(PR_SLOT_MEAN[slot] * scale) - PR_SLOT_SIGMA[slot] ** 2 / 2
    return float(min(max(math.exp(nprng.normal(mu, PR_SLOT_SIGMA[slot])), 0.0), 8.0))

def noclick_ratio(nprng):
    """未点击行的播放比例：11.6% 恰好为 0，其余重尾对数正态（均值 0.144）。
    旧实现 uniform(0.02, 0.25) 均值凑巧接近，但没有零点也没有长尾。"""
    if nprng.random_sample() < NOCLK_ZERO_RATE: return 0.0
    return float(min(math.exp(nprng.normal(math.log(NOCLK_LN_MED), NOCLK_LN_SIGMA)), 8.0))

def load_video_pool():
    """候选池：全库有标题视频。返回 id 数组、时长、类别、热度权重。"""
    vid, dur, c1, c2, sc = [], [], [], [], []
    with open(C.F_VIDEO_META, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 6: continue
            if C.REQUIRE_CAPTION and not (p[3] or "").strip(): continue
            vid.append(p[0]); c1.append(p[1]); c2.append(p[2])
            dur.append(int(p[4] or 0)); sc.append(float(p[5] or 0))
    return (np.array(vid), np.array(dur, dtype=np.int64), np.array(c1), np.array(c2),
            np.array(sc, dtype=np.float64))

def main():
    ap = {k: sys.argv[sys.argv.index(k) + 1] for k in ("--shard", "--nshard", "--plan")
          if k in sys.argv}
    shard, nshard = int(ap.get("--shard", 0)), int(ap.get("--nshard", 1))
    plan_tag = ap.get("--plan", "")
    os.makedirs(SHARDS, exist_ok=True)
    seed = C.SEED + 1000 + shard
    rng = random.Random(seed); nprng = np.random.RandomState(seed % (2**31))

    side, _ = F.load_video_side()
    vid, dur, c1, c2, sc = load_video_pool()
    NV = len(vid)
    idx_of = {v: i for i, v in enumerate(vid)}
    # 热度基线权重：P ∝ 热度^γ。真实曝光高度集中在头部，γ<1 起压平作用。
    base_w = np.power(np.maximum(sc, 1.0), GAMMA)
    base_w /= base_w.sum()
    # 类别 -> 候选池内的行号（做偏好加权时按类别整体加权，避免 50w×10w 的稠密乘）
    cat_rows = collections.defaultdict(list)
    for i, c in enumerate(c1): cat_rows[c].append(i)
    cat_rows = {k: np.array(v) for k, v in cat_rows.items()}
    cat_mass = {k: float(base_w[v].sum()) for k, v in cat_rows.items()}
    # 每行的时长档 + 全库（按热度加权的）档位份额。推荐环境此前只按【类别】迎合用户，
    # 时长完全随全库走，于是每个人拿到的时长构成几乎一样：合成 log_dur_mean 的
    # 用户间 σ 只有 0.266，真实 0.385（曝光档0 份额 σ 0.053 vs 真实 0.092）。
    # donor 基因里本来就有 dur_expo，只是没被用上。
    d_slot = np.array([F.dur_slot(int(x)) for x in dur])
    lib_dslot = np.array([base_w[d_slot == i].sum() for i in range(5)])
    lib_dslot /= lib_dslot.sum()

    arch = {a["archetype_id"]: a for a in json.load(open(
        os.path.join(GEN_DATA, "archetypes.json"), encoding="utf-8"))}
    # 原型画像 -> 三个语义特征所需的文本/二元组（与 features.load_profile_sem 同口径）
    asem = {}
    for aid, a in arch.items():
        pr = a.get("profile") or {}
        pos = " ".join([F._flat(pr.get("long_term_preference")),
                        F._flat(pr.get("short_term_interest")),
                        F._flat(pr.get("role_tags"))])
        asem[aid] = {"pos": pos, "av": F._flat(pr.get("aversion")), "pos_bg": F._bigrams(pos)}

    # 统计路权重：直接复用旧用户线在真实训练期拟合好的成对 logistic。
    # 这不是自证——权重来自真实用户的真实点击，用它给新用户打分，等于把
    # 「真实偏好结构」注入合成数据；§4 的迁移测试正是要验证这一点能反向成立。
    fs = json.load(open(os.path.join(C.OUT, "fuse_summary.json")))
    W = np.array([fs["weights"][n] for n in F.FEAT_NAMES])

    f_plan = os.path.join(GEN_DATA, f"user_plan{plan_tag}.tsv")
    f_log = os.path.join(SHARDS, f"log{plan_tag}_{shard:03d}.csv")
    f_rec = os.path.join(SHARDS, f"records{plan_tag}_{shard:03d}.jsonl")
    fl = open(f_log, "w", newline="", encoding="utf-8"); wl = csv.writer(fl)
    fr = open(f_rec, "w", encoding="utf-8")

    n_user = n_row = n_clk = 0
    clicks_hist = []
    for ln, line in enumerate(open(f_plan, encoding="utf-8")):
        if ln % nshard != shard: continue
        uid, gid, aid, n_exp_tgt, ctr_tgt, genj = line.rstrip("\n").split("\t")
        gen = json.loads(genj)
        aid = int(aid); ctr_tgt = float(ctr_tgt)
        sem = asem.get(aid, {"pos": "", "av": "", "pos_bg": set()})

        # --- 用户状态：类别偏好向量（会随轮次漂移）+ 时长偏好 + 累积计数 ---
        pref = dict(gen["cat_pref"]) or {"其他": 1.0}
        tot = sum(pref.values()) or 1.0
        pref = {k: v / tot for k, v in pref.items()}
        dur_pref = np.array(gen["dur_pref"] + [0] * (5 - len(gen["dur_pref"])))[:5]
        if dur_pref.sum() <= 0: dur_pref = np.ones(5)
        dur_pref = dur_pref / dur_pref.sum()
        # 曝光侧的时长构成：目标 = donor 的 dur_expo，再按同样的「31天边际 → 5天窗口」
        # 逻辑抽一次 Dirichlet 扰动（donor 的 dur_expo 档0 σ 只有 0.057，真实窗口 0.092）。
        # 抽样时按 目标份额 / 全库份额 给每个视频加权，即可把曝光构成推到目标上。
        d_exp = np.array((gen["dur_expo"] + [0] * 5)[:5], float)
        if d_exp.sum() <= 0: d_exp = lib_dslot.copy()
        d_exp = d_exp / d_exp.sum()
        d_exp = nprng.dirichlet(np.maximum(d_exp * 5 * DUR_DIR_ALPHA, 1e-3))
        d_tilt = d_exp / lib_dslot
        hours = np.array(gen["hours"] + [0] * (24 - len(gen["hours"])))[:24].astype(float)
        if hours.sum() <= 0: hours = np.ones(24)
        hours /= hours.sum()
        # 窗口小时分布：donor 的 hours 是 31 天边际，对 5 天窗口来说过于平滑。
        # 抽一次 Dirichlet 扰动 = 「这个人这 5 天恰好偏爱哪几个时段」，
        # 把窗口内最热小时占比从 17.3% 抬到真实的 21% 一线。
        hours = nprng.dirichlet(np.maximum(hours * 24 * HOUR_DIR_ALPHA, 1e-3))
        # 个体播放倾向：把 donor 的整体 play_ratio_mean 折成一个【相对全库均值的缩放因子】，
        # 再乘到各时长档的档位均值上。这样既保留「这个人看得久 / 划得快」的个体差异，
        # 又保留「短视频循环播、长视频看不完」的档位结构——两者旧实现只顾了前者。
        pr_scale = float(np.clip(gen["play_ratio_mean"] / PR_GLOBAL_MEAN, 0.35, 2.2))
        # 个体会话长度不再单列参数：每天的小时数由 P(h|当天条数) 经验表抽，
        # 段长是 n/h 的派生量（真实数据里本来就是这样）。见 session_offsets。
        # 个体推荐收敛度：λ 决定推荐环境对该用户偏好的迎合强度。真实用户在
        # 5天×20条口径下点到的一级类别数 15.8±5.3；用固定 λ=2.5 得 18.6±4.7，
        # 均值偏高、方差偏低——真实世界里有人被喂得很窄（信息茧房），有人很杂，
        # 而固定 λ 让所有人收敛速度一致。这里让 λ 按用户对数正态浮动。
        lam_u = float(np.clip(math.exp(nprng.normal(
            math.log(LAMBDA_MED), LAMBDA_LNSD)), 0.3, 40.0))
        tau_u = float(np.clip(math.exp(nprng.normal(
            math.log(PREF_TAU_MED), PREF_TAU_LNSD)), 0.25, 2.5))
        # 个体日均曝光量：由 Step B 分配的目标活跃度决定（实测 corr(log总曝光, log日均)
        # = 0.95，斜率 0.92——活跃的人不只是天数多，每天也刷得更多）。
        # 这里抽的是**未截尾**的日条数，落行时再 min(N_CAND, ·)：交付规格的
        # 「每天最多 20 条」本身就是个截尾算子，同形状对照组走的也是它，
        # 于是行数分布按构造对齐（真实 84.2±19.7，69.5% 的天恰好 20 条）。
        z = (math.log(max(int(n_exp_tgt), 2)) - LA_MEAN) / LA_SD
        cand_mu = DAY_LN_MEAN + USER_LNSD * z
        tabs = gen["tabs"] or {"1": 1.0}
        tab_k = list(tabs); tab_p = np.array([tabs[k] for k in tab_k], float); tab_p /= tab_p.sum()
        # 用户侧统计量：合成用户没有历史日志，用基因直接铺出等效计数
        # （features.user_state 的 e1/k1/e2/k2/ed/kd 结构），这样 feats() 原样可用。
        n_hist = max(int(n_exp_tgt), 50)
        us = {"n": n_hist, "nc": int(n_hist * ctr_tgt), "ctr": ctr_tgt,
              "e1": collections.Counter(), "k1": collections.Counter(),
              "e2": collections.Counter(), "k2": collections.Counter(),
              "ed": collections.Counter(), "kd": collections.Counter(),
              "vcnt": collections.Counter(), "clicked": set(), "sem": sem}
        for c, p in gen["cat_expo"].items(): us["e1"][c] = int(n_hist * p)
        for c, p in pref.items():            us["k1"][c] = int(us["nc"] * p)
        for i, p in enumerate(gen["dur_expo"][:5]): us["ed"][i] = int(n_hist * p)
        for i, p in enumerate(gen["dur_pref"][:5]): us["kd"][i] = int(us["nc"] * p)

        seen = set()
        seen_j = []                 # 已曝光过的候选池行号，供重复曝光重放
        wv_cache = {}               # 类别内抽样权重（含时长倾斜），每人算一次即可
        rows_u, recs = [], []
        n_clk_u = 0
        # 每人 5 轮，轮次映射到 5 个不同的日期（从 7 天里按 user 起点滚动，
        # 让 50w 人均匀铺满 20220502-20220508，而不是全挤在前 5 天）
        d0 = int(uid) % (len(SIM_DATES) - N_ROUNDS + 1)
        for rd in range(N_ROUNDS):
            date = SIM_DATES[d0 + rd]
            # --- 1. 推荐环境：P(曝光v) ∝ 热度^γ · (1+λ·该用户对 v 类别的偏好) ---
            # 两段式抽：先按「类别质量 × 偏好加权」抽类别，再在类别内按热度抽视频。
            # 等价于稠密加权抽样，但只需 O(类别数) 而非 O(10 万)。
            ks = list(cat_mass)
            pw = np.array([pref.get(k, 0.0) for k in ks]) ** tau_u
            ps = pw.sum()
            if ps > 0: pw = pw / ps
            cw = np.array([cat_mass[k] for k in ks]) * (1.0 + lam_u * pw)
            cw /= cw.sum()
            # 未截尾日条数 -> min(N_CAND, ·)：见 DAY_LN_MEAN 处的说明
            n_cand_d = int(np.clip(round(math.exp(nprng.normal(cand_mu, DAY_LNSD))),
                                   CAND_MIN, N_CAND))
            pick_c = nprng.choice(len(ks), size=n_cand_d, p=cw)
            cand = []
            for ci in pick_c:
                # 重复曝光：真实日志 100 条窗口内 distinct_ratio 均值 0.9885（约每 100 条
                # 有 1.15 条重复）。**必须显式重放**——候选池 9 万+，靠随机撞上已曝光视频
                # 的概率约等于 0，第一版指望「不查重」自然产生重复，实测 distinct 恒为
                # 1.00000，是判别器权重第四的特征。
                if seen_j and nprng.random_sample() < REEXPOSE_P:
                    cand.append(seen_j[rng.randrange(len(seen_j))]); continue
                rows_c = cat_rows[ks[ci]]
                wv = wv_cache.get(ci)
                if wv is None:
                    # 类别内按 热度^γ × 时长档倾斜 抽视频。时长倾斜 = 该用户目标档位份额
                    # / 全库档位份额：只改「推什么长度的视频」，不改热度的相对次序。
                    wv = base_w[rows_c] * d_tilt[d_slot[rows_c]]
                    s_ = wv.sum()
                    wv = wv / s_ if s_ > 0 else np.full(len(rows_c), 1.0 / len(rows_c))
                    wv_cache[ci] = wv
                for _ in range(8):                       # 最多重试 8 次避开已曝光
                    j = int(rows_c[nprng.choice(len(rows_c), p=wv)])
                    if vid[j] not in seen: break
                cand.append(j)
            # 同一轮内去重（同一段刷里连推两条一样的说不通）；跨轮重复由上面显式重放产生
            cand = list(dict.fromkeys(cand))
            if not cand: continue

            # --- 2. 用户智能体决策：17 维成对 logistic 打分 + 每用户 CTR 校准 ---
            sc_v = np.array([float(np.array(F.feats(us, side[vid[j]])) @ W) for j in cand])
            # α_u 让 score 的离散度落在合理区间（防「点击与打分无关」或「过度确定」）；
            # β_u 用二分求解，使该用户实现的 CTR 等于 Step B 分配的目标 CTR。
            s = sc_v.std() + 1e-6
            alpha = 1.2 / s
            z = alpha * (sc_v - sc_v.mean())
            lo, hi = -30.0, 30.0
            for _ in range(40):
                mid = (lo + hi) / 2
                if (1 / (1 + np.exp(-np.clip(z + mid, -30, 30)))).mean() > ctr_tgt: hi = mid
                else: lo = mid
            p_click = 1 / (1 + np.exp(-np.clip(z + (lo + hi) / 2, -30, 30)))
            clicked = nprng.random_sample(len(cand)) < p_click

            # --- 3. 落行 + 辅助字段 ---
            # 本轮（= 一天）的 n_cand_d 条按「分段刷」铺开时间，而不是全塞进同一个整点
            secs = session_offsets(len(cand), hours, nprng, rng)
            y, mo, dd = date // 10000, date // 100 % 100, date % 100
            for k, j in enumerate(cand):
                v = vid[j]; d = int(dur[j])
                if v not in seen: seen.add(v); seen_j.append(j)
                dt = (datetime.datetime(y, mo, dd, tzinfo=TZ)
                      + datetime.timedelta(seconds=secs[k]))
                clk = int(clicked[k])
                if clk:
                    n_clk_u += 1
                    ratio = play_ratio(F.dur_slot(d), pr_scale, nprng)
                    lv = int(nprng.random_sample() < gen["long_view"])
                    lk = int(nprng.random_sample() < gen["like"])
                    fo = int(nprng.random_sample() < gen["follow"])
                    cm = int(nprng.random_sample() < gen["comment"])
                    fw = int(nprng.random_sample() < gen["forward"])
                    pe = int(nprng.random_sample() < gen["profile_enter"])
                    # 负反馈率直接用 donor 的实测值。曾经乘过 0.3 做「保守衰减」，
                    # 那是拍脑袋的——§4 边际检验里 hate_r 因此偏低 77%。
                    # 注意：真实日志里 45% 的 is_hate 落在未点击行，本实现按断言 5
                    # 统一把互动收敛到点击行（见 05_merge.py 说明），是一处有意简化。
                    ht = int(nprng.random_sample() < gen["hate"])
                    # comment_stay_time：实测真实点击行 13.2% 非零，非零值均值 9,970ms。
                    # 旧用户线沿用硬编码 0（与基线同口径），新用户线必须复现这个分布。
                    cs = 0
                    if nprng.random_sample() < max(gen["comment_stay_rate"], 1e-4):
                        cs = int(max(200, nprng.lognormal(
                            math.log(max(gen["comment_stay_mean"], 500)), 0.8)))
                    play = int(ratio * d)
                else:
                    lv = lk = fo = cm = fw = pe = ht = cs = 0
                    play = int(noclick_ratio(nprng) * d)
                tab = tab_k[int(nprng.choice(len(tab_k), p=tab_p))]
                rows_u.append([uid, v, date, hourmin_of(dt), int(dt.timestamp() * 1000),
                               clk, lk, fo, cm, fw, ht, lv, play, d, 0, cs, pe, 0, tab])

            # --- 4. 状态更新：本轮实际点击的类别混入偏好，下一轮曝光跟着漂移 ---
            got = collections.Counter()
            for k, j in enumerate(cand):
                cc = c1[j]
                us["e1"][cc] += 1; us["e2"][c2[j]] += 1
                us["ed"][F.dur_slot(int(dur[j]))] += 1
                us["n"] += 1
                if clicked[k]:
                    got[cc] += 1
                    us["k1"][cc] += 1; us["k2"][c2[j]] += 1
                    us["kd"][F.dur_slot(int(dur[j]))] += 1
                    us["nc"] += 1; us["clicked"].add(vid[j])
            us["ctr"] = us["nc"] / max(us["n"], 1)
            if got:
                t = sum(got.values())
                pref = {k: (1 - DRIFT) * v for k, v in pref.items()}
                for k, v in got.items(): pref[k] = pref.get(k, 0.0) + DRIFT * v / t
                t2 = sum(pref.values()) or 1.0
                pref = {k: v / t2 for k, v in pref.items()}
            recs.append({"simulation_id": f"sim{plan_tag or '_500k'}",
                         "user_id": int(uid), "round_no": rd,
                         "recommendation_info": {"n_cand": len(cand), "date": date,
                                                 "top_cat": [k for k, _ in
                                                             sorted(pref.items(), key=lambda x: -x[1])[:3]]},
                         "behavior_info": {"n_click": int(clicked.sum()),
                                           "ctr_round": round(float(clicked.mean()), 4)},
                         "state_info": {"cum_ctr": round(us["ctr"], 4),
                                        "cum_expose": us["n"], "drift": DRIFT}})

        # --- 硬下限：每人 ≥MIN_CLICKS 次点击。不足的把 p_click 最高的未点击行翻成点击。---
        if n_clk_u < MIN_CLICKS:
            need = MIN_CLICKS - n_clk_u
            for r in rows_u:
                if need <= 0: break
                if r[5] == 0:
                    d = r[13]
                    r[5] = 1
                    r[11] = int(nprng.random_sample() < gen["long_view"])
                    r[12] = int(play_ratio(F.dur_slot(d), pr_scale, nprng) * d)
                    need -= 1; n_clk_u += 1
        assert n_clk_u >= MIN_CLICKS, f"用户 {uid} 点击 {n_clk_u} < {MIN_CLICKS}"
        assert len(rows_u) >= MIN_CLICKS, f"用户 {uid} 曝光 {len(rows_u)} < {MIN_CLICKS}"
        for r in rows_u: wl.writerow(r)
        for r in recs: fr.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_user += 1; n_row += len(rows_u); n_clk += n_clk_u
        clicks_hist.append(n_clk_u)
        if n_user % 500 == 0:
            print(f"  [片{shard}] {n_user:,} 人  {n_row:,} 行  CTR {n_clk/max(n_row,1):.4f}",
                  flush=True)
    fl.close(); fr.close()
    ch = sorted(clicks_hist)
    print(f"[片{shard}] 完成 {n_user:,} 人 / {n_row:,} 行 / 点击 {n_clk:,} "
          f"(CTR {n_clk/max(n_row,1):.4f})  每人点击 min/中位/max = "
          f"{ch[0] if ch else 0}/{ch[len(ch)//2] if ch else 0}/{ch[-1] if ch else 0}")
    print(f"-> {f_log}\n-> {f_rec}")

if __name__ == "__main__":
    main()
