"""Step 5: 语义 × 统计 融合判决 + 产出最终模拟互动日志。

概设 2.4.1 的两路模块在此汇合：
  语义路  LLM 多角度画像/记忆判决        -> sim_trace.jsonl 里的 choice / llm_score
  统计路  轻量化成对 logistic（features.py）-> 对数几率 margin

融合器（v2，门控堆叠）。v1 是 logit = a·margin + β·L_llm，实测 b(LLM)≈0——因为
prompt 里注入了平台热度块，LLM 与统计路吃同一个捷径，共线，投票级融合拿不到增量
（统计路错时 LLM 命中率仅 0.417，比抛硬币还差）。v2 改三点：

  1) 正交化：融合用的语义通道关掉热度块（`--angles item_popularity=0`）。LLM 单独
     从 0.673 降到 0.625，但两路一致率 0.685→0.603，统计路错时救回率 0.429→0.629，
     门控上限 0.950→0.968。**弱而正交的通道才是好的融合伙伴。**
  2) 去位置偏置：A/B 互换各跑一次，取 sA=(s1−s2)/2（s=logP(A)−logP(B)）。
     位置项在相减中抵消，实测选 A 率 0.380→0.460（真值 0.420）。
  3) 门控：LLM 的权重随统计路自信度衰减 g=1/(1+(|m|/τ)²)。统计路在 |m|<0.5 时
     只有 0.545（近抛硬币），在 |m|≥4 时 0.998——LLM 只该在前者说话。

  logit = a₁·m + a₂·m|m| + b₁·L·g(|m|) + b₂·tanh(s/4)·g(|m|)
  语义项一律带门控 g，先验直接写进特征——否则 100 对小样本上会学出
  「统计路自信时把 LLM 反号用」的退化解（实测 L=−1.35 / L·g=+1.59，融合掉点）。
  所有特征都是 A↔B 互换下的奇函数，保证融合器本身不引入位置偏置。
  (a, b) 按用户分 2 折交叉拟合，每个用户用的权重都来自另一折用户。

  4) 特征级融合（真正拿到增量的一路）：不让 LLM 投票，而是把它写的画像文本折成
     sem_cat_pref / sem_cat_avert / sem_cap_pref 三个特征（features.py），
     在训练期与 14 个统计特征联合定权。统计路 0.8968 → 0.8986。

统计路权重只在【训练期】构造的二选一对上拟合（正=训练期点击，负=同一全库随机协议），
测试期只做一次前向，无泄漏。
"""
import os, sys, json, math, random, collections, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from recsim import config as C
from recsim.sim import features as F

LLM_DEV_ACC = 0.68                      # 开发集(200对)实测的 LLM 单独准确率
BETA_FIXED = math.log(LLM_DEV_ACC / (1 - LLM_DEV_ACC))   # 朴素贝叶斯式固定权重（对照）
TRAIN_PAIRS_PER_USER = 60
N_FOLD = 2                              # 融合权重按用户分折交叉拟合
GATE_TAU = 1.5                          # 门控宽度：|margin| 小于它时 LLM 才有发言权

STACK_NAMES = ["m", "m|m|", "L·gate", "tanh(s/4)·gate"]

def stack_feats(m, L, s):
    """构造门控堆叠特征。全部是 (m,L,s)->(-m,-L,-s) 下的奇函数。

    语义项一律带门控因子 g，即「统计路越自信，LLM 越没发言权」的先验直接写进特征，
    模型学不出「统计路自信时把 LLM 反号用」这种小样本退化解。
    """
    g = 1.0 / (1.0 + (np.abs(m) / GATE_TAU) ** 2)
    return np.stack([m, m * np.abs(m), L * g, np.tanh(s / 4.0) * g], 1)

def fit_logit(X, y, epochs=1500, lr=0.03, l2=1e-3):
    """对称化拟合：同时喂 (X, y) 与 (-X, 1-y)，保证无常数项/无位置项。"""
    X2 = np.concatenate([X, -X]); t = np.concatenate([y, 1 - y]).astype(float)
    sd = X2.std(0) + 1e-9
    Z = X2 / sd
    w = np.zeros(Z.shape[1]); mw = vw = np.zeros(Z.shape[1])
    for k in range(1, epochs + 1):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ ((p - t) / len(t)) + l2 * w
        mw = 0.9 * mw + 0.1 * g; vw = 0.999 * vw + 0.001 * g ** 2
        w -= lr * (mw / (1 - 0.9 ** k)) / (np.sqrt(vw / (1 - 0.999 ** k)) + 1e-8)
    return w / sd                       # 折回原尺度，可直接点乘原特征

def fit_fusion(m, L, y, epochs=800, lr=0.05):
    """v1 对照：logit = a*m + b*L。"""
    X = np.stack([np.concatenate([m, -m]), np.concatenate([L, -L])], 1)
    t = np.concatenate([y, 1 - y]).astype(float)
    w = np.array([1.0, 0.3]); mw = vw = np.zeros(2)
    for k in range(1, epochs + 1):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ ((p - t) / len(t))
        mw = 0.9 * mw + 0.1 * g; vw = 0.999 * vw + 0.001 * g ** 2
        w -= lr * (mw / (1 - 0.9 ** k)) / (np.sqrt(vw / (1 - 0.999 ** k)) + 1e-8)
    return w

COLS = ["user_id","video_id","date","hourmin","time_ms","is_click","is_like","is_follow",
        "is_comment","is_forward","is_hate","long_view","play_time_ms","duration_ms",
        "is_profile_enter","tab"]
OUT_COLS = ["user_id","video_id","date","hourmin","time_ms","is_click","is_like","is_follow",
            "is_comment","is_forward","is_hate","long_view","play_time_ms","duration_ms",
            "profile_stay_time","comment_stay_time","is_profile_enter","is_rand","tab"]

def load_logs():
    per = collections.defaultdict(list)
    for line in open(C.F_USER_LOGS, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) < 16: continue
        per[p[0]].append(dict(zip(COLS, p)))
    for u in per:
        per[u].sort(key=lambda r: int(r["time_ms"]))
    return per

def behavior_pool(train, side):
    pool = collections.defaultdict(list)
    for r in train:
        if int(r["is_click"]) != 1: continue
        d = int(r["duration_ms"] or 0) or side.get(r["video_id"], {}).get("d", 0)
        if d <= 0: continue
        pool[F.dur_slot(d)].append((min(int(r["play_time_ms"]) / d, 8.0), int(r["long_view"]),
                                    int(r["is_like"]), int(r["is_follow"]), int(r["is_comment"]),
                                    int(r["is_forward"]), int(r["is_profile_enter"])))
    allp = [x for v in pool.values() for x in v]
    return pool, allp

def main():
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
    sfx = f"_{tag}" if tag else ""
    f_trace = C.F_SIM_TRACE.replace(".jsonl", f"{sfx}.jsonl")
    if "--nosem" in sys.argv: sfx += "_nosem"      # 消融产物另存，便于配对比较
    # --noheat：把统计路的热度特征清零。全库随机负采样下热度是压倒性捷径
    # （log_show_user/log_show_cnt 权重 1.39/1.06，其余全部 ≤0.53），
    # 统计路单独就能到 0.897，融合无处可加。拿掉捷径后两路都只靠内容判断，
    # 才能看出「语义 × 统计」这个融合机制本身有没有增量。
    noheat = "--noheat" in sys.argv
    if noheat: sfx += "_noheat"
    f_out = C.F_SIM_LOG.replace(".csv", f"{sfx}.csv")
    f_dec = os.path.join(C.OUT, f"decisions{sfx}.jsonl")
    rng = random.Random(C.SEED + 1)

    print("[1/4] 载入视频侧特征 / 日志 / LLM trace ...")
    side, vids = F.load_video_side()
    # --nosem：把语义增强特征清零（消融对照），验证 LLM 画像在特征层的净贡献
    nosem = "--nosem" in sys.argv
    psem = {} if nosem else F.load_profile_sem()
    if nosem: print("  [消融] 语义增强特征已清零（sem_cat_pref/sem_cat_avert/sem_cap_pref）")
    msk = np.ones(len(F.FEAT_NAMES))
    if noheat:
        for nm in ("log_show_cnt", "log_show_user"):
            msk[F.FEAT_NAMES.index(nm)] = 0.0
        print("  [消融] 统计路热度特征已清零（log_show_cnt/log_show_user）")
    fv = lambda *a, **kw: np.array(F.feats(*a, **kw)) * msk
    per = load_logs()
    rows = [json.loads(l) for l in open(f_trace, encoding="utf-8")]
    pool = [v for v in vids if (side[v]["cap"] or "").strip()] if C.REQUIRE_CAPTION else vids
    print(f"  全库视频 {len(vids):,}  候选池(有标题) {len(pool):,}  测试期二选一 {len(rows):,} 对")
    # 曝光时间：取该用户测试期对该正样本视频的真实记录（旧版 trace 未存这三列时兜底）
    tstamp = {}
    for u, rs_ in per.items():
        for x in rs_:
            if C.TEST_START <= int(x["date"]) <= C.TEST_END:
                tstamp.setdefault((u, x["video_id"]), (x["date"], x["hourmin"], x["time_ms"]))

    print(f"[2/4] 训练期构造二选一对并拟合成对 logistic (每用户 {TRAIN_PAIRS_PER_USER} 对) ...")
    users = sorted({r["user_id"] for r in rows})
    ustate, D = {}, []
    for u in users:
        allrows = per[u]
        train = [r for r in allrows if C.TRAIN_START <= int(r["date"]) <= C.TRAIN_END]
        ustate[u] = F.user_state(train, side, sem=psem.get(u))
        seen = {r["video_id"] for r in allrows}
        pos = [r["video_id"] for r in train if int(r["is_click"]) == 1 and r["video_id"] in side
               and (not C.REQUIRE_CAPTION or (side[r["video_id"]]["cap"] or "").strip())]
        pos = list(dict.fromkeys(pos))
        rng.shuffle(pos)
        for pv in pos[:TRAIN_PAIRS_PER_USER]:
            while True:
                nv = pool[rng.randrange(len(pool))]
                if nv not in seen: break
            D.append(fv(ustate[u], side[pv], pv, loo=True)
                     - fv(ustate[u], side[nv], nv, loo=True))
    D = np.array(D)
    print(f"  训练对 {len(D):,}")
    model = F.fit_pairwise(D)
    tr_acc = float(((D / model["sd"]) @ model["w"] > 0).mean())
    print(f"  训练期成对准确率 {tr_acc:.4f}")
    print("  权重（标准化尺度，正=越大越像用户会看的）:")
    for nm, wv in sorted(zip(F.FEAT_NAMES, model["w"]), key=lambda kv: -abs(kv[1])):
        print(f"    {nm:16s} {wv:+.3f}")

    print("[3/4] 测试期打分 + 按用户 2 折交叉拟合融合权重 ...")
    M, LV, SV, Y, FOLD = [], [], [], [], []
    ufold = {u: i % N_FOLD for i, u in enumerate(users)}
    n_score = 0
    for r in rows:
        u, pv, nv, truth, ch = r["user_id"], r["pos"], r["neg"], r["truth"], r["choice"]
        us = ustate[u]
        fa, fb = ((fv(us, side[pv]), fv(us, side[nv])) if truth == "A"
                  else (fv(us, side[nv]), fv(us, side[pv])))
        M.append(F.margin(model, fa, fb))
        # 语义通道：有 A/B 双向 logprobs 就用去偏后的连续分，否则退回 1 bit 投票
        s = r.get("llm_score")
        if s is None:
            s = 0.0 if ch is None else (4.0 if ch == "A" else -4.0)
        else:
            n_score += 1
        SV.append(float(s))
        LV.append(0.0 if s == 0 else (1.0 if s > 0 else -1.0))
        Y.append(1.0 if truth == "A" else 0.0)
        FOLD.append(ufold[u])
    M = np.array(M); LV = np.array(LV); SV = np.array(SV); Y = np.array(Y); FOLD = np.array(FOLD)
    print(f"  语义通道：{n_score}/{len(rows)} 条有 A/B 双向 logprobs 连续分"
          f"{'（其余退回投票）' if n_score < len(rows) else ''}")
    X = stack_feats(M, LV, SV)
    fused = np.zeros(len(M)); fused_v1 = np.zeros(len(M)); ws, ws1 = [], []
    for k in range(N_FOLD):
        tr, te = FOLD != k, FOLD == k
        w = fit_logit(X[tr], Y[tr]); ws.append(w)
        fused[te] = X[te] @ w
        w1 = fit_fusion(M[tr], LV[tr], Y[tr]); ws1.append(w1)
        fused_v1[te] = w1[0] * M[te] + w1[1] * LV[te]
        print(f"    折{k}: 在另 {int(tr.sum())} 条上拟合 -> "
              + "  ".join(f"{nm}={wv:+.3f}" for nm, wv in zip(STACK_NAMES, w))
              + f"  | v1 对照 a={w1[0]:.3f} b={w1[1]:.3f}")
    # 门控在不同区间的等效 LLM 权重（给报告用）
    # L 与 tanh(s/4) 相关系数 0.952（62% 的样本 |tanh(s/4)|>0.9 已饱和），两项近乎共线，
    # 权重怎么在它们之间分摊在折间不稳定（实测 b1 = +0.10 / −0.39），只有【和】有意义。
    # 故这里报 b1+b2（饱和时的语义项合计），不要只报 b1。
    bsem = float(np.mean([w[2] + w[3] for w in ws]))
    print(f"    语义项合计 b1+b2 = {bsem:+.3f}（对比统计项 a1 = "
          f"{float(np.mean([w[0] for w in ws])):+.3f}）")
    for am in (0.2, 1.0, 3.0, 6.0):
        g = 1.0 / (1.0 + (am / GATE_TAU) ** 2)
        print(f"    等效 LLM 票权重 @|margin|={am:.1f}: {bsem*g:+.4f}  (门控 g={g:.3f})")
    print(f"    |margin| 中位 {float(np.median(np.abs(M))):.2f}，"
          f"|margin|<0.5 的样本占 {float((np.abs(M)<0.5).mean()):.4f}"
          f" —— 语义路只在这部分够大到能翻转结果")

    print("[4/4] 写最终模拟日志与决策明细 ...")
    n = collections.Counter()
    fo = open(f_out, "w", newline="", encoding="utf-8"); w = csv.writer(fo); w.writerow(OUT_COLS)
    fd = open(f_dec, "w", encoding="utf-8")
    pools = {}
    for i, r in enumerate(rows):
        u, pv, nv, truth, ch = r["user_id"], r["pos"], r["neg"], r["truth"], r["choice"]
        if u not in pools:
            train = [x for x in per[u] if C.TRAIN_START <= int(x["date"]) <= C.TRAIN_END]
            pools[u] = behavior_pool(train, side)
        m = float(M[i]); L = float(LV[i]); s = float(SV[i])
        pick_stat = "A" if m > 0 else "B"
        pick_llm  = "A" if L > 0 else ("B" if L < 0 else None)   # 去位置偏置后的语义票
        pick_fix  = "A" if m + BETA_FIXED * L > 0 else "B"
        pick_v1   = "A" if fused_v1[i] > 0 else "B"
        pick_fuse = "A" if fused[i] > 0 else "B"
        n["all"] += 1
        n["llm"] += (ch == truth)
        n["llm_db"] += (pick_llm == truth)
        n["stat"] += (pick_stat == truth)
        n["fix"] += (pick_fix == truth)
        n["v1"] += (pick_v1 == truth)
        n["fuse"] += (pick_fuse == truth)
        n["agree"] += (pick_stat == pick_llm)

        chosen = pv if pick_fuse == truth else nv
        rejected = nv if pick_fuse == truth else pv
        fd.write(json.dumps({**{k: r[k] for k in ("key","user_id","pos","neg","truth","group_id","train_ctr")},
                             "llm_choice": ch, "llm_choice_db": pick_llm,
                             "llm_score": round(s, 3), "stat_margin": round(m, 4),
                             "fused_logit": round(float(fused[i]), 4),
                             "stat_choice": pick_stat, "fused_choice": pick_fuse,
                             "llm_correct": (None if ch is None else ch == truth),
                             "llm_db_correct": (None if pick_llm is None else pick_llm == truth),
                             "stat_correct": pick_stat == truth,
                             "fused_correct": pick_fuse == truth,
                             "chosen_video": chosen}, ensure_ascii=False) + "\n")
        pool, allp = pools[u]
        for vid, clk in ((chosen, 1), (rejected, 0)):
            s = side[vid]; d = s["d"]
            if clk:
                cand = pool.get(F.dur_slot(d)) or allp or [(0.6, 0, 0, 0, 0, 0, 0)]
                ratio, lv, lk, fol, cm, fw, pe = rng.choice(cand)
                play = int(ratio * d)
            else:
                lv = lk = fol = cm = fw = pe = 0
                play = int(rng.uniform(0.02, 0.25) * d)
            dt, hm, tm = tstamp.get((u, pv), (r.get("date",""), r.get("hourmin",""), r.get("time_ms","")))
            w.writerow([u, vid, dt, hm, tm,
                        clk, lk, fol, cm, fw, 0, lv, play, d, 0, 0, pe, 0, 1])
    fo.close(); fd.close()

    N = n["all"]
    # 配对显著性（McNemar）：融合 vs 统计路
    b_ = sum(1 for i, r in enumerate(rows)
             if (("A" if fused[i] > 0 else "B") == r["truth"]) and not (("A" if M[i] > 0 else "B") == r["truth"]))
    c_ = sum(1 for i, r in enumerate(rows)
             if not (("A" if fused[i] > 0 else "B") == r["truth"]) and (("A" if M[i] > 0 else "B") == r["truth"]))
    chi = (abs(b_ - c_) - 1) ** 2 / max(b_ + c_, 1)
    pval = math.erfc(math.sqrt(chi / 2)) if b_ + c_ else 1.0
    print(f"\n{'='*64}")
    print(f"  语义路 LLM 单向     {n['llm']}/{N} = {n['llm']/N:.4f}")
    print(f"  语义路 A/B双向去偏  {n['llm_db']}/{N} = {n['llm_db']/N:.4f}")
    print(f"  统计路 单独         {n['stat']}/{N} = {n['stat']/N:.4f}")
    print(f"  固定β融合(对照)     {n['fix']}/{N} = {n['fix']/N:.4f}")
    print(f"  v1 线性融合(对照)   {n['v1']}/{N} = {n['v1']/N:.4f}")
    print(f"  v2 门控堆叠融合     {n['fuse']}/{N} = {n['fuse']/N:.4f}   <- 最终")
    print(f"     vs 统计路: 救回 {b_} / 拖累 {c_}  McNemar p={pval:.4f}")
    print(f"  两路一致率          {n['agree']/N:.4f}      随机基线 0.5000")
    print(f"{'='*64}")
    json.dump({"n": N, "llm": n["llm"] / N, "llm_debiased": n["llm_db"] / N,
               "stat": n["stat"] / N, "fuse": n["fuse"] / N,
               "fixed_beta_fuse": n["fix"] / N, "v1_linear_fuse": n["v1"] / N,
               "agree": n["agree"] / N,
               "mcnemar_vs_stat": {"rescued": b_, "hurt": c_, "p": round(pval, 5)},
               "gate_tau": GATE_TAU,
               "sem_weight_total": round(bsem, 4),      # b1+b2，语义路的实际话语权
               "stat_weight_a1": round(float(np.mean([w[0] for w in ws])), 4),
               "stack_weights": {nm: [round(float(w[j]), 4) for w in ws]
                                 for j, nm in enumerate(STACK_NAMES)},
               "v1_weights": [[round(float(x), 4) for x in w1] for w1 in ws1],
               "train_pair_acc": tr_acc,
               "weights": dict(zip(F.FEAT_NAMES, [round(float(x), 4) for x in model["w"]]))},
              open(os.path.join(C.OUT, f"fuse_summary{sfx}.json"), "w"), ensure_ascii=False, indent=1)
    print(f"-> {f_out}\n-> {f_dec}")

if __name__ == "__main__":
    main()
