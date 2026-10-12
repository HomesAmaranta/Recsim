"""Step B（新用户线）：50 万模拟用户特征表（A-README §4.2，31 字段）+ manifest。

不用 LLM，不用 GPU。纯 numpy + 标准库。

构造顺序（**先定群，再定 donor，再定字段**，顺序不能换）：
  1. group_id ~ 真实 8 群的规模分布
  2. 行为 donor：群内抽 2-3 个真实用户，Dirichlet 权重混合其行为基因
  3. 原型绑定：同群 200 个原型里按类别偏好余弦相似度 softmax 抽一个，继承画像文本
  4. 活跃度 × CTR **联合抽**（实测 corr(log活跃度, CTR) = -0.1168）
     独立抽会造出「超高活跃 + 超高 CTR」这种真实数据里不存在的组合，判别器一抓一个准
  5. 特征表 31 字段：分块 bootstrap（块内保持联合分布，块间换 donor 以免整行照抄）
     - 社交三元组整体从同 user_active_degree 层的 donor 复制 + 抖动
     - 四个 *_range 由数值【现算】（range_maps.json），内部一致性免费成立
     - onehot 18 列整块从另一个 donor 原样复制（加密特征强相关，逐列独立重采样毁联合分布；
       空值与 "1.0" 浮点串格式也靠整块复制自然带上）

user_id 从 1,000,000 起（真实最大 27,284），与真实 id 空间完全不重叠。

产物：
  out/gen/sim_user_features_500k.csv   500,001 行 × 31 列 ← **交付表之二**
  out/gen/manifest.jsonl               每人的 donor / 原型 / 目标活跃度与 CTR，可追溯性依据
  data/gen/user_plan.tsv               04_env_rollout.py 用的紧凑计划表（分片读，不吃内存）

用法：
  <py> recsim/gen/03_users.py --n 5000        # 先小规模跑 §4 四项检验
  <py> recsim/gen/03_users.py --n 500000      # 正式
"""
import os, sys, json, csv, math, random, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from recsim import config as C

GEN_DATA = os.path.join(C.DATA, "gen")
OUT_GEN  = os.path.join(C.OUT, "gen")
F_GENOME = os.path.join(GEN_DATA, "donor_genomes.jsonl")
F_ARCH   = os.path.join(GEN_DATA, "archetypes.json")
F_RANGE  = os.path.join(GEN_DATA, "range_maps.json")
F_MAR    = os.path.join(GEN_DATA, "real_marginals.json")

UID_BASE = 1000000
N_DONOR_MIX = (2, 3)      # 每人混几个行为 donor
MAX_REUSE_TARGET = 60     # §4 多样性护栏：单 donor 复用上限

USER_COLS = ["user_id","user_active_degree","is_lowactive_period","is_live_streamer",
             "is_video_author","follow_user_num","follow_user_num_range",
             "fans_user_num","fans_user_num_range","friend_user_num","friend_user_num_range",
             "register_days","register_days_range"] + [f"onehot_feat{i}" for i in range(18)]
assert len(USER_COLS) == 31

def lookup(buckets, v):
    """数值 -> 区间标签。与 00_range_maps.lookup 同一实现。"""
    lab = buckets[0][2]
    for lo, _, l in buckets:
        if v >= lo: lab = l
        else: break
    return lab

def cosine(a, b):
    s = sum(a.get(k, 0) * b.get(k, 0) for k in a)
    na = math.sqrt(sum(v * v for v in a.values())); nb = math.sqrt(sum(v * v for v in b.values()))
    return s / (na * nb) if na and nb else 0.0

def mix_genomes(gs, w):
    """Dirichlet 权重混合若干真实用户的行为基因。

    比例型量（类别分布/时长分布/互动率/小时分布/tab）按权重线性混合；
    活跃度与 CTR **不在这里定** —— 它们由第 4 步的联合抽样单独给，
    否则混合会把两人的活跃度平均掉，破坏 (log活跃度, CTR) 的真实联合分布。
    """
    def mix_dict(key, norm=True):
        out = collections.defaultdict(float)
        for g, wi in zip(gs, w):
            d = g[key]; t = sum(d.values()) or 1
            for k, v in d.items(): out[k] += wi * (v / t if norm else v)
        return dict(out)
    def mix_list(key, norm=True):
        n = max(len(g[key]) for g in gs)
        out = [0.0] * n
        for g, wi in zip(gs, w):
            a = g[key]; t = (sum(a) or 1) if norm else 1
            for i, v in enumerate(a): out[i] += wi * v / t
        return out
    return {
        "cat_pref":  mix_dict("cat_click"),        # 点击类别分布（偏好）
        "cat_expo":  mix_dict("cat_expose"),       # 曝光类别分布（推荐环境见过什么）
        "dur_pref":  mix_list("dur_click"),
        "dur_expo":  mix_list("dur_expose"),
        "hours":     mix_list("hours"),
        "tabs":      mix_dict("tabs"),
        "play_ratio_mean":   sum(wi * g["play_ratio_mean"] for g, wi in zip(gs, w)),
        "comment_stay_rate": sum(wi * g["comment_stay_rate"] for g, wi in zip(gs, w)),
        "comment_stay_mean": sum(wi * g["comment_stay_mean"] for g, wi in zip(gs, w)),
        **{k: sum(wi * g[k] for g, wi in zip(gs, w))
           for k in ("like","follow","comment","forward","hate","long_view","profile_enter")},
    }

def main():
    n_users = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 500000
    os.makedirs(OUT_GEN, exist_ok=True)
    rng = random.Random(C.SEED + 7)
    nprng = np.random.RandomState((C.SEED + 7) % (2**31))
    tag = "" if n_users >= 500000 else f"_{n_users}"

    print(f"[1/6] 载入 donor 基因 / 原型 / 区间映射 / 真实边际 ...")
    genomes = [json.loads(l) for l in open(F_GENOME, encoding="utf-8")]
    arch = json.load(open(F_ARCH, encoding="utf-8"))
    rmaps = json.load(open(F_RANGE))
    mar = json.load(open(F_MAR))
    real_feat = {r["user_id"]: r for r in
                 csv.DictReader(open(C.USER_FEAT, encoding="utf-8-sig"))}
    byg = collections.defaultdict(list)
    for g in genomes: byg[g["group_id"]].append(g)
    arch_byg = collections.defaultdict(list)
    for a in arch: arch_byg[a["group_id"]].append(a)
    gids = sorted(byg)
    gsize = np.array([len(byg[g]) for g in gids], dtype=float); gsize /= gsize.sum()
    print(f"  donor {len(genomes):,}  原型 {len(arch):,}  群规模 "
          f"{[f'{x:.1%}' for x in gsize]}")
    for g in gids:
        assert arch_byg[g], f"群{g} 没有原型，先跑 02_archetypes.py"

    print("[2/6] 建 (log活跃度, CTR) 二维经验分布 ...")
    # 联合抽样：直接从真实 24,292 个 (log活跃度, CTR) 点里整体抽一个 + 局部抖动。
    # 抖动幅度取各维标准差的 1/12——大到能造出新点，小到不破坏二维形状。
    la = np.array([math.log(g["n_expose"]) for g in genomes])
    ct = np.array([g["ctr"] for g in genomes])
    gen_by_uid = {g["user_id"]: g for g in genomes}
    jitter = (la.std() / 12.0, ct.std() / 12.0)
    real_corr = float(((la - la.mean()) * (ct - ct.mean())).mean() / (la.std() * ct.std()))
    print(f"  真实 corr(log活跃度, CTR) = {real_corr:+.4f}  抖动 σ = "
          f"({jitter[0]:.4f}, {jitter[1]:.4f})")

    # 社交三元组 + onehot 的 donor 池，按 user_active_degree 分层
    soc_by_act = collections.defaultdict(list)
    for r in real_feat.values():
        soc_by_act[r["user_active_degree"]].append(r)
    act_names = list(mar["user_active_degree"])
    act_p = np.array([mar["user_active_degree"][a] for a in act_names], float); act_p /= act_p.sum()
    # 活跃度档位：用真实的【条件分布】P(user_active_degree | log曝光分箱)。
    # 曾用「取 log 曝光中位数最接近的档位」，实测灾难性失真：full_active 合成 2.1% vs
    # 真实 67.8% —— 因为 low_active 这类稀有档的中位数恰好落在人群密集区，
    # 最近邻规则把大量用户判给了它。条件分布法则自动同时复现边际与相关性
    # （lx 本来就是从真实 (log活跃度,CTR) 分布抽的）。
    N_ABIN = 20
    abin_edges = np.quantile(la, np.linspace(0, 1, N_ABIN + 1)[1:-1])
    act_cond = collections.defaultdict(collections.Counter)
    for r in real_feat.values():
        g = gen_by_uid.get(r["user_id"])
        if not g: continue
        b = int(np.searchsorted(abin_edges, math.log(g["n_expose"])))
        act_cond[b][r["user_active_degree"]] += 1
    act_cond_v = {}
    for b, cnt in act_cond.items():
        ks = [k for k in cnt if k in soc_by_act]
        pv = np.array([cnt[k] for k in ks], float); pv /= pv.sum()
        act_cond_v[b] = (ks, pv)
    onehot_pool = list(real_feat.values())
    sxa = list(mar["streamer_x_author"]); sxa_p = np.array(
        [mar["streamer_x_author"][k] for k in sxa], float); sxa_p /= sxa_p.sum()

    print(f"[3/6] 生成 {n_users:,} 个新用户 ...")
    f_user = os.path.join(OUT_GEN, f"sim_user_features_500k{tag}.csv")
    f_mani = os.path.join(OUT_GEN, f"manifest{tag}.jsonl")
    f_plan = os.path.join(GEN_DATA, f"user_plan{tag}.tsv")
    fu = open(f_user, "w", newline="", encoding="utf-8")   # 不写 BOM：源表的 BOM 是瑕疵
    wu = csv.writer(fu); wu.writerow(USER_COLS)
    fm = open(f_mani, "w", encoding="utf-8")
    fp = open(f_plan, "w", encoding="utf-8")
    reuse = collections.Counter()
    tgt_la, tgt_ct = [], []
    # 预抽向量化随机量（50w 次 Python 级 randrange 太慢）
    G = nprng.choice(len(gids), size=n_users, p=gsize)
    JP = nprng.choice(len(genomes), size=n_users)                 # (活跃度,CTR) 联合抽的行号
    JX = nprng.randn(n_users) * jitter[0]
    JY = nprng.randn(n_users) * jitter[1]
    SA = nprng.choice(len(sxa), size=n_users, p=sxa_p)

    for i in range(n_users):
        uid = UID_BASE + i
        gid = gids[G[i]]
        pool = byg[gid]
        # --- 2. 行为 donor 混合 ---
        k = rng.randint(*N_DONOR_MIX)
        ds = [pool[rng.randrange(len(pool))] for _ in range(k)]
        w = nprng.dirichlet([1.6] * k)          # α=1.6：偏均匀但仍有主 donor
        gen = mix_genomes(ds, w)
        for d in ds: reuse[d["user_id"]] += 1
        # --- 3. 原型绑定：同群原型里按类别偏好余弦相似度 softmax 抽 ---
        cands = arch_byg[gid]
        if len(cands) > 24:                     # 只在随机 24 个里比，省 50w × 200 次余弦
            cands = [cands[rng.randrange(len(cands))] for _ in range(24)]
        sims = np.array([cosine(gen["cat_pref"], a["cat_pref"]) for a in cands])
        ex = np.exp((sims - sims.max()) * 6.0); ex /= ex.sum()
        a = cands[int(nprng.choice(len(cands), p=ex))]
        # --- 4. 活跃度 × CTR 联合抽（整体抽点 + 局部抖动，保持二维形状）---
        j = JP[i]
        lx = float(np.clip(la[j] + JX[i], math.log(50), la.max()))
        cy = float(np.clip(ct[j] + JY[i], 0.0, 0.995))
        n_expose_total = int(round(math.exp(lx)))
        tgt_la.append(lx); tgt_ct.append(cy)
        # --- 5. 特征表字段 ---
        # user_active_degree：按抽到的 log 活跃度落分箱，再从该箱的真实条件分布里抽
        b = int(np.searchsorted(abin_edges, lx))
        ks, pv = act_cond_v.get(b) or (act_names, act_p)
        act = ks[int(nprng.choice(len(ks), p=pv))]
        if act not in soc_by_act: act = act_names[int(nprng.choice(len(act_names), p=act_p))]
        sd = soc_by_act[act][rng.randrange(len(soc_by_act[act]))]        # 社交 donor
        od = onehot_pool[rng.randrange(len(onehot_pool))]                # onehot donor（另抽）
        st, au = sxa[SA[i]].split("|")
        def jig(v):
            try: x = int(float(v))
            except (ValueError, TypeError): return 0
            return max(0, int(round(x * rng.uniform(0.85, 1.15))))
        fo_n, fa_n, fr_n = (jig(sd["follow_user_num"]), jig(sd["fans_user_num"]),
                            jig(sd["friend_user_num"]))
        fr_n = min(fr_n, fo_n)                       # 互关数不可能超过关注数
        rd = max(14, jig(sd["register_days"]))
        row = [uid, act, 0, st, au,
               fo_n, lookup(rmaps["follow_user_num_range"]["buckets"], fo_n),
               fa_n, lookup(rmaps["fans_user_num_range"]["buckets"], fa_n),
               fr_n, lookup(rmaps["friend_user_num_range"]["buckets"], fr_n),
               rd,   lookup(rmaps["register_days_range"]["buckets"], rd)]
        row += [od[f"onehot_feat{t}"] for t in range(18)]     # 整块复制，含空值与 "1.0"
        wu.writerow(row)
        fm.write(json.dumps({
            "user_id": uid, "group_id": gid, "archetype_id": a["archetype_id"],
            "behavior_donors": [d["user_id"] for d in ds],
            "donor_weights": [round(float(x), 4) for x in w],
            "social_donor": sd["user_id"], "onehot_donor": od["user_id"],
            "target_n_expose": n_expose_total, "target_ctr": round(cy, 5),
            "user_active_degree": act}, ensure_ascii=False) + "\n")
        # 紧凑计划表：04 分片读这个，不必载入 manifest 与全部基因
        fp.write("\t".join([str(uid), str(gid), str(a["archetype_id"]),
                            str(n_expose_total), f"{cy:.5f}",
                            json.dumps(gen, ensure_ascii=False, separators=(",", ":"))]) + "\n")
        if (i + 1) % 100000 == 0: print(f"    {i+1:,}/{n_users:,}")
    fu.close(); fm.close(); fp.close()

    print("[4/6] 自检 ...")
    # 联合分布是否复现
    LA, CT = np.array(tgt_la), np.array(tgt_ct)
    corr = float(((LA - LA.mean()) * (CT - CT.mean())).mean() / (LA.std() * CT.std()))
    print(f"  合成 corr(log活跃度, CTR) = {corr:+.4f}  vs 真实 {real_corr:+.4f}  "
          f"偏差 {abs(corr-real_corr):.4f} {'OK' if abs(corr-real_corr)<0.05 else '!! 超 0.05'}")
    mx = reuse.most_common(1)[0] if reuse else ("-", 0)
    print(f"  donor 复用：涉及 {len(reuse):,} 人，最大 {mx[1]} 次（{mx[0]}），"
          f"中位 {sorted(reuse.values())[len(reuse)//2]}，护栏线 {MAX_REUSE_TARGET}")
    # 行列数 + range 一致性
    with open(f_user, encoding="utf-8") as f:
        rd_ = csv.DictReader(f); rows = list(rd_)
        assert rd_.fieldnames == USER_COLS, "列名/顺序与 §4.2 不一致"
    assert len(rows) == n_users, f"{len(rows)} != {n_users}"
    bad = 0
    for r in rows[:20000]:
        for rngf, mp in rmaps.items():
            if lookup(mp["buckets"], int(r[mp["num_field"]])) != r[rngf]: bad += 1
    assert bad == 0, f"{bad} 处 *_range 与数值不一致"
    ids = {int(r["user_id"]) for r in rows}
    assert len(ids) == n_users and min(ids) >= UID_BASE, "user_id 不唯一或与真实空间重叠"
    assert all(r["is_lowactive_period"] == "0" for r in rows)
    assert all(r["is_live_streamer"] in ("-124", "1") for r in rows)
    print(f"  行数 {len(rows):,} × {len(USER_COLS)} 列，*_range 一致性 20,000 行抽检 0 处不符，"
          f"user_id 唯一且 ≥{UID_BASE:,}")

    print("[5/6] 脏特征复现核对 ...")
    nz = {c: sum(1 for r in rows if r[c] == "") for c in
          ("onehot_feat4",) + tuple(f"onehot_feat{i}" for i in range(12, 18))}
    exp = {c: mar["quirks"]["empty_counts"][c] / mar["n_users"] for c in nz}
    for c in nz:
        print(f"    {c}: 空值 {nz[c]}（{nz[c]/len(rows):.4%}）vs 真实 {exp[c]:.4%}")
    ndot = sum(1 for r in rows if "." in r["onehot_feat12"])
    print(f"    onehot_feat12 带 '.0' 的行 {ndot:,}/{len(rows):,}（真实该列全是浮点串）")

    print("[6/6] 边际分布对比（前 6 个活跃度档）...")
    got = collections.Counter(r["user_active_degree"] for r in rows)
    for a in act_names[:6]:
        print(f"    {a:18s} 合成 {got[a]/len(rows):7.4f}  真实 "
              f"{mar['user_active_degree'][a]/mar['n_users']:7.4f}")
    json.dump({"n_users": n_users, "corr_logact_ctr": round(corr, 4),
               "real_corr": round(real_corr, 4),
               "donor_max_reuse": mx[1], "donor_used": len(reuse)},
              open(os.path.join(OUT_GEN, f"users_summary{tag}.json"), "w"), indent=1)
    print(f"\n-> {f_user}\n-> {f_mani}\n-> {f_plan}")

if __name__ == "__main__":
    main()
