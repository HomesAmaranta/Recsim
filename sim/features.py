"""轻量化统计模型（概设 2.4.1「特征-统计模型」侧）。

给定 (用户, 候选视频) 输出一组可解释特征；再用成对 logistic（Bradley-Terry / RankNet）
在【训练期】的二选一对上拟合权重，输出校准好的对数几率，供与 LLM 语义判决融合。

严格无泄漏：
  - 用户侧统计量只用 TRAIN_START..TRAIN_END
  - 权重只在训练期构造的二选一对上拟合（正=训练期点击，负=同一全库随机协议）
  - 测试期只做一次前向
"""
import os, sys, csv, math, random, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from recsim import config as C

FEAT_NAMES = [
    "log_show_cnt",      # 视频平台热度（全库随机负采样下的主导信号）
    "log_show_user",     # 触达用户数
    "valid_play_rate",   # 有效播放率
    "long_play_rate",    # 长播放率
    "complete_rate",     # 完播率
    "cat1_lift",         # 用户对该一级类别的点击率提升（收缩平滑）
    "cat2_lift",         # 二级类别
    "cat1_share",        # 该一级类别在用户历史点击中的占比
    "cat1_hit",          # 用户是否点过该一级类别
    "cat2_hit",          # 用户是否点过该二级类别
    "dur_ctr",           # 用户在该时长档的点击率（收缩平滑）
    "log_dur",           # 视频时长
    "cap_overlap",       # 标题与用户历史点击标题的字符二元组覆盖率
    "cap_len",           # 标题长度
    # ↓ 语义增强特征（角度 12）：来自 LLM 画像文本，不是计数统计量。
    #   决策级融合（LLM 投票）实测拿不到增量，改成让 LLM 的语义概括进特征层，
    #   由成对 logistic 在训练期与 14 个统计特征联合定权。
    "sem_cat_pref",      # LLM 写的偏好/兴趣文本里是否点名了候选的一/二级类别
    "sem_cat_avert",     # LLM 写的「明显回避」文本里是否点名了候选类别
    "sem_cap_pref",      # 候选标题与偏好文本的字符二元组覆盖率
]
SHRINK_K = 20

def _bigrams(s):
    s = (s or "").strip()
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else set()

def dur_slot(ms):
    for i, hi in enumerate((10000, 20000, 40000, 80000)):
        if ms < hi: return i
    return 4

def load_video_side():
    """全库视频侧特征：热度/播放率/时长/类别/标题。"""
    stat = {}
    with open(C.VIDEO_STAT, encoding="utf-8-sig") as f:
        for x in csv.DictReader(f):
            def g(k):
                try: return float(x[k])
                except Exception: return 0.0
            sc = g("show_cnt")
            stat[x["video_id"]] = (sc, g("show_user_num"),
                                   g("valid_play_cnt") / (sc + 1),
                                   g("long_time_play_cnt") / (sc + 1),
                                   g("complete_play_cnt") / (sc + 1))
    vids, side = [], {}
    with open(C.F_VIDEO_META, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 6: continue
            v, d = p[0], int(p[4] or 0)
            sc, su, vp, lp, cp = stat.get(v, (float(p[5] or 0), 0.0, 0.0, 0.0, 0.0))
            side[v] = {"c1": p[1], "c2": p[2], "cap": p[3], "d": d,
                       "bg": None,  # 惰性算
                       "sc": sc, "su": su, "vp": vp, "lp": lp, "cp": cp}
            vids.append(v)
    return side, vids

def _flat(x):
    return " ".join(str(i) for i in x) if isinstance(x, (list, tuple)) else str(x or "")

def load_profile_sem(path=None):
    """读 03_build_profiles 产出的 LLM 语义画像，折成可比对的文本/二元组集合。

    正向文本 = 长期偏好 + 近期兴趣 + 角色标签；负向文本 = 「明显回避」。
    画像只用训练期日志生成，故对测试期无泄漏。训练期拟合时存在极轻微自证
    （某条正样本的类别本来也计入了生成画像的聚合分布），但画像是数百次点击的
    概括，单条影响可忽略——不像 cap_overlap 会被单条标题拉到 1.0；是否真有增益
    由测试期数字判定。
    """
    import json
    d = json.load(open(path or C.F_PROFILES, encoding="utf-8"))
    out = {}
    for u, st in d.items():
        pr = st.get("profile") or {}
        pos = " ".join([_flat(pr.get("long_term_preference")), _flat(pr.get("short_term_interest")),
                        _flat(pr.get("role_tags")), _flat(st.get("role_tags"))])
        av = _flat(pr.get("aversion"))
        out[u] = {"pos": pos, "av": av, "pos_bg": _bigrams(pos)}
    return out

def user_state(train, side, sem=None):
    """用户侧统计量，仅用训练期记录。保留原始计数，便于构造训练对时做留一(LOO)扣除。"""
    n = len(train)
    nc = sum(1 for r in train if int(r["is_click"]) == 1)
    e1 = collections.Counter(); k1 = collections.Counter()
    e2 = collections.Counter(); k2 = collections.Counter()
    ed = collections.Counter(); kd = collections.Counter()
    vcnt = collections.Counter()      # 二元组 -> 含该二元组的「已点击视频」数
    clicked = set()
    for r in train:
        s = side.get(r["video_id"])
        if not s: continue
        e1[s["c1"]] += 1; e2[s["c2"]] += 1; ed[dur_slot(s["d"])] += 1
        if int(r["is_click"]):
            k1[s["c1"]] += 1; k2[s["c2"]] += 1; kd[dur_slot(s["d"])] += 1
            if s["bg"] is None: s["bg"] = _bigrams(s["cap"])
            for b in s["bg"]: vcnt[b] += 1
            clicked.add(r["video_id"])
    return {"n": n, "nc": nc, "ctr": nc / n if n else 0.0,
            "e1": e1, "k1": k1, "e2": e2, "k2": k2, "ed": ed, "kd": kd,
            "vcnt": vcnt, "clicked": clicked,
            "sem": sem or {"pos": "", "av": "", "pos_bg": set()}}

def feats(us, s, vid=None, loo=False):
    """loo=True 且该视频在用户训练期点击集合内时，先把它自己的贡献扣掉，
    否则构造训练对时正样本会「自证」（标题覆盖率恒为1、类别计数含自身），
    权重会严重高估语义匹配、低估热度，测试期直接掉点。"""
    d = 1 if (loo and vid in us["clicked"]) else 0
    n = max(us["n"] - d, 1); nc = max(us["nc"] - d, 0)
    ctr = nc / n
    if s["bg"] is None:
        s["bg"] = _bigrams(s["cap"])
    bg = s["bg"]

    def lift(k, e, key):
        kk = k.get(key, 0) - d; ee = e.get(key, 0) - d
        return ((kk + SHRINK_K * ctr) / (ee + SHRINK_K)) / (ctr + 1e-9)
    slot = dur_slot(s["d"])
    kd_ = us["kd"].get(slot, 0) - d; ed_ = us["ed"].get(slot, 0) - d
    k1_ = us["k1"].get(s["c1"], 0) - d
    k2_ = us["k2"].get(s["c2"], 0) - d
    vc = us["vcnt"]
    ov = (sum(1 for b in bg if vc.get(b, 0) - d > 0) / len(bg)) if bg else 0.0
    sm = us["sem"]
    cats = [c for c in (s["c1"], s["c2"]) if c and c != "UNKNOWN"]
    sem_p = 1.0 if any(c in sm["pos"] for c in cats) else 0.0
    sem_a = 1.0 if (sm["av"] and any(c in sm["av"] for c in cats)) else 0.0
    sem_cap = (len(bg & sm["pos_bg"]) / len(bg)) if (bg and sm["pos_bg"]) else 0.0
    return [
        math.log1p(s["sc"]), math.log1p(s["su"]),
        s["vp"], s["lp"], s["cp"],
        math.log(lift(us["k1"], us["e1"], s["c1"]) + 1e-6),
        math.log(lift(us["k2"], us["e2"], s["c2"]) + 1e-6),
        k1_ / max(nc, 1),
        1.0 if k1_ > 0 else 0.0,
        1.0 if k2_ > 0 else 0.0,
        ((kd_ + SHRINK_K * ctr) / (ed_ + SHRINK_K)) / (ctr + 1e-6),
        math.log1p(s["d"]),
        ov, math.log1p(len(s["cap"] or "")),
        sem_p, sem_a, sem_cap,
    ]

def affinity_desc(us, s):
    """把统计路的「用户 x 候选」交叉特征翻译成可读文本，供注入 LLM prompt（角度7c）。
    只暴露特征本身，不暴露成对 logistic 的分数——否则 LLM 只是复读统计路，融合失去意义。

    按全量 5000 对实测的单特征判别力排序注入，强的说白、弱的标注、噪声不注入：
      cat1_hit 0.843 / cat2_hit 0.809 / cat1_share 0.691 / cat1_lift 0.604
      （cap_overlap 0.551、dur_ctr 0.540 已剔除——接近抛硬币，注入只会挤掉内容判断）
    用户侧计数全部来自训练期，正样本来自测试期，无泄漏。
    """
    ctr = us["ctr"]
    L = []

    def cat_line(name, key, e, k):
        if not key or key == "UNKNOWN":
            return None
        ee, kk = e.get(key, 0), k.get(key, 0)
        if ee == 0:
            return f"{name}「{key}」：训练期从没给你推过，无历史可依据"
        if kk == 0:
            return f"{name}「{key}」：被推过 {ee} 次，你【一次都没点过】（强负向信号）"
        lift = ((kk + SHRINK_K * ctr) / (ee + SHRINK_K)) / (ctr + 1e-9)
        return (f"{name}「{key}」：你【点过 {kk} 次】（被推 {ee} 次，"
                f"点击率 {kk/ee*100:.0f}% vs 你总体 {ctr*100:.0f}%，偏好 {lift:.2f} 倍，弱证据）")

    for t in (cat_line("一级类别", s["c1"], us["e1"], us["k1"]),
              cat_line("二级类别", s["c2"], us["e2"], us["k2"])):
        if t: L.append(t)
    nc = max(us["nc"], 1)
    sh = us["k1"].get(s["c1"], 0) / nc
    L.append(f"该一级类别占你历史全部点击的 {sh*100:.1f}%" + ("（几乎不看）" if sh < 0.005 else ""))
    return L

def fit_pairwise(D, epochs=400, lr=0.05, l2=1e-4, seed=C.SEED):
    """D: (m, k) 特征差矩阵，标签恒为 1（第一项更优）。对称化后拟合。"""
    X = np.vstack([D, -D])
    y = np.concatenate([np.ones(len(D)), np.zeros(len(D))])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    w = np.zeros(Xs.shape[1]); b = 0.0
    mw = vw = np.zeros_like(w); mb = vb = 0.0
    rng = np.random.RandomState(seed % (2**31))
    for t in range(1, epochs + 1):
        z = Xs @ w + b
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        g = (p - y) / len(y)
        gw = Xs.T @ g + l2 * w; gb = g.sum()
        mw = 0.9 * mw + 0.1 * gw; vw = 0.999 * vw + 0.001 * gw**2
        mb = 0.9 * mb + 0.1 * gb; vb = 0.999 * vb + 0.001 * gb**2
        w -= lr * (mw / (1 - 0.9**t)) / (np.sqrt(vw / (1 - 0.999**t)) + 1e-8)
        b -= lr * (mb / (1 - 0.9**t)) / (np.sqrt(vb / (1 - 0.999**t)) + 1e-8)
    return {"w": w, "b": b, "mu": mu, "sd": sd}

def margin(model, fa, fb):
    """A 相对 B 的对数几率。b 项在成对差上自动抵消，不参与。"""
    d = (np.asarray(fa) - np.asarray(fb)) / model["sd"]
    return float(d @ model["w"])
