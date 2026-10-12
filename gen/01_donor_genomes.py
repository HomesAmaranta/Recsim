"""Step 1（新用户线）：全量真实用户的行为基因 + 群体聚类。不用 LLM，不用 GPU。

为什么需要这一步：Step B 造 50w 用户时要从【全部 24,170 个真实用户】里抽 donor，
而 recsim/data/ 里只有当初抽中的 200 人的日志。donor 混合需要每人的行为基因向量，
所以这里对 36.9M 行日志做【一趟流式聚合】，产出每人一份基因。

用 awk 扫（纯 Python 扫 3690 万行要几分钟，awk 约 1-2 分钟），视频侧的
类别/时长从 video_meta.tsv 在 BEGIN 里建表。

产出 data/gen/donor_genomes.jsonl，每行一个真实用户：
  活跃度/CTR、一级类别的曝光与点击分布、5 个时长档的曝光与点击、
  点击后的互动率（赞/关注/评论/转发/主页/长看/负反馈）、
  24 小时时段分布、tab 分布、播放比例均值、comment_stay 非零率
外加 KMeans k=8 的 group_id（与 prep/03 同一套聚类口径：类别向量 + 总CTR）。

注意：这里【不做】训练/测试期切分——新用户是凭空造的，不存在「测试期真值」，
整个 20220408-20220508 都是可用的行为素材。这与旧用户线的时间切分不冲突：
旧用户线要防泄漏，新用户线没有可泄漏的对象。
"""
import os, sys, json, subprocess, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from recsim import config as C

GEN_DATA = os.path.join(C.DATA, "gen")
F_RAW    = os.path.join(GEN_DATA, "donor_raw.tsv")
F_GENOME = os.path.join(GEN_DATA, "donor_genomes.jsonl")

DUR_EDGES = [10000, 20000, 40000, 80000]          # 与 features.dur_slot 同口径，5 档
N_DUR = 5

# 日志列（0-based -> awk 的 1-based）：
#   1 user_id 2 video_id 3 date 4 hourmin 5 time_ms 6 is_click 7 is_like 8 is_follow
#   9 is_comment 10 is_forward 11 is_hate 12 long_view 13 play_time_ms 14 duration_ms
#   15 profile_stay 16 comment_stay 17 is_profile_enter 18 is_rand 19 tab
AWK = r'''
BEGIN {
  FS = ","
  while ((getline line < VM) > 0) {          # video_meta.tsv: video_id \t cat1 \t ...
    nf = split(line, m, "\t")                # 不能叫 n：下面 n[u] 是数组，awk 不允许同名
    if (nf >= 5) { vcat[m[1]] = m[2]; vdur[m[1]] = m[5] + 0 }
  }
}
NR == 1 { next }
{
  u = $1; clk = $6 + 0
  n[u]++; c[u] += clk
  cat = vcat[$2]; if (cat == "") cat = "UNKNOWN"
  ce[u SUBSEP cat]++; if (clk) cc[u SUBSEP cat]++
  seen_cat[u SUBSEP cat] = 1

  d = $14 + 0; if (d <= 0) d = vdur[$2]
  s = 0
  if (d >= 10000) s = 1
  if (d >= 20000) s = 2
  if (d >= 40000) s = 3
  if (d >= 80000) s = 4
  de[u SUBSEP s]++; if (clk) dc[u SUBSEP s]++

  h = int(($4 + 0) / 100); if (h < 0 || h > 23) h = 0
  hh[u SUBSEP h]++
  tb[u SUBSEP ($19 + 0)]++

  if (clk) {
    lk[u] += $7; fo[u] += $8; cm[u] += $9; fw[u] += $10;
    ht[u] += $11; lv[u] += $12; pe[u] += $17
    if ($16 + 0 > 0) { csn[u]++; css[u] += $16 + 0 }
    if (d > 0) { pr[u] += ($13 + 0) / d; prn[u]++ }
  }
}
END {
  # 一趟扫 key 拼字符串。不能对每个用户再遍历一次全表：24,170 用户 × 36 万 key 会卡死。
  for (k in seen_cat) {
    split(k, a, SUBSEP)
    catstr[a[1]] = catstr[a[1]] a[2] ":" ce[k] ":" (cc[k] + 0) ";"
  }
  for (k in tb) {
    split(k, a, SUBSEP)
    tabstr[a[1]] = tabstr[a[1]] a[2] ":" tb[k] ";"
  }
  for (u in n) {
    ds = ""; for (i = 0; i < 5; i++) ds = ds (de[u SUBSEP i] + 0) ":" (dc[u SUBSEP i] + 0) ";"
    hs = ""; for (i = 0; i < 24; i++) hs = hs (hh[u SUBSEP i] + 0) ";"
    printf "%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.6f\t%d\t%.1f\t%s\t%s\t%s\t%s\n",
      u, n[u], c[u], lk[u]+0, fo[u]+0, cm[u]+0, fw[u]+0, ht[u]+0, lv[u]+0, pe[u]+0,
      (prn[u] ? pr[u]/prn[u] : 0), csn[u]+0, (csn[u] ? css[u]/csn[u] : 0),
      catstr[u], ds, hs, tabstr[u]
  }
}
'''

RAW_COLS = ["user_id","n_expose","n_click","like","follow","comment","forward","hate",
            "long_view","profile_enter","play_ratio_mean","comment_stay_n","comment_stay_mean",
            "cats","durs","hours","tabs"]

def kmeans(X, k, seed, iters=60):
    """与 prep/03_build_profiles 同一套实现，保证 group_id 口径一致。"""
    rng = np.random.RandomState(seed)
    cen = X[rng.choice(len(X), k, replace=False)].copy()
    lab = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - cen[None, :, :]) ** 2).sum(2)
        nl = d.argmin(1)
        if (nl == lab).all(): break
        lab = nl
        for j in range(k):
            m = lab == j
            if m.any(): cen[j] = X[m].mean(0)
    return lab

def main():
    os.makedirs(GEN_DATA, exist_ok=True)
    if not os.path.exists(F_RAW):
        print("[1/3] awk 流式聚合 36.9M 行日志 -> 每用户基因原料（约 1-3 分钟）...")
        with open(F_RAW, "w") as fo:
            subprocess.run(["awk", "-v", f"VM={C.F_VIDEO_META}", AWK, C.LOG],
                           stdout=fo, check=True)
    else:
        print(f"[1/3] 复用已有 {F_RAW}")

    print("[2/3] 解析为基因向量 ...")
    genomes = []
    for line in open(F_RAW, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) < len(RAW_COLS): continue
        r = dict(zip(RAW_COLS, p))
        ne, nc = int(r["n_expose"]), int(r["n_click"])
        if ne < 50: continue                      # 素材太少的用户不适合当 donor
        cats = {}
        for it in r["cats"].split(";"):
            if not it: continue
            name, e, c = it.rsplit(":", 2)
            cats[name] = [int(e), int(c)]
        durs = [[int(x) for x in it.split(":")] for it in r["durs"].split(";") if it]
        hours = [int(x) for x in r["hours"].split(";") if x != ""]
        tabs = {}
        for it in r["tabs"].split(";"):
            if not it: continue
            t, n_ = it.split(":"); tabs[t] = int(n_)
        genomes.append({
            "user_id": r["user_id"], "n_expose": ne, "n_click": nc,
            "ctr": round(nc / ne, 5),
            "cat_expose": {k: v[0] for k, v in cats.items()},
            "cat_click":  {k: v[1] for k, v in cats.items() if v[1]},
            "dur_expose": [d[0] for d in durs], "dur_click": [d[1] for d in durs],
            "hours": hours, "tabs": tabs,
            "play_ratio_mean": float(r["play_ratio_mean"]),
            "comment_stay_rate": round(int(r["comment_stay_n"]) / nc, 5) if nc else 0.0,
            "comment_stay_mean": float(r["comment_stay_mean"]),
            # 互动率一律以【点击数】为分母（未点击不可能有互动）
            **{k: (round(int(r[k]) / nc, 5) if nc else 0.0)
               for k in ("like","follow","comment","forward","hate","long_view","profile_enter")},
        })
    print(f"  合格 donor {len(genomes):,} 人（曝光>=50）")

    print(f"[3/3] KMeans k={C.N_CLUSTERS} 群体聚类 ...")
    allcats = sorted({c for g in genomes for c in g["cat_click"]})
    X = np.array([[g["cat_click"].get(c, 0) / max(g["n_click"], 1) for c in allcats]
                  + [g["ctr"]] for g in genomes], dtype=np.float64)
    lab = kmeans(X, C.N_CLUSTERS, C.SEED)
    for g, l in zip(genomes, lab): g["group_id"] = int(l)
    with open(F_GENOME, "w", encoding="utf-8") as f:
        for g in genomes: f.write(json.dumps(g, ensure_ascii=False) + "\n")

    print(f"  一级类别 {len(allcats)} 个")
    for gid in range(C.N_CLUSTERS):
        mem = [g for g in genomes if g["group_id"] == gid]
        agg = collections.Counter()
        for g in mem:
            for c, v in g["cat_click"].items(): agg[c] += v / max(g["n_click"], 1)
        ctr = sum(g["ctr"] for g in mem) / max(len(mem), 1)
        act = sorted(g["n_expose"] for g in mem)
        print(f"    群{gid}: {len(mem):5d} 人 ({len(mem)/len(genomes):.1%})  平均CTR {ctr:.3f}  "
              f"曝光中位 {act[len(act)//2]:5d}  关注 {[c for c,_ in agg.most_common(3)]}")

    # 活跃度 × CTR 的相关（新用户抽样必须联合抽，独立抽会造出不存在的组合）
    la = np.log([g["n_expose"] for g in genomes]); ct = np.array([g["ctr"] for g in genomes])
    corr = float(((la - la.mean()) * (ct - ct.mean())).mean() / (la.std() * ct.std()))
    print(f"  corr(log 活跃度, CTR) = {corr:+.4f}   <- 02_users.py 的联合抽样必须复现这个数")
    json.dump({"corr_logact_ctr": round(corr, 4), "n_donors": len(genomes),
               "cats": allcats,
               "group_sizes": [int((lab == g).sum()) for g in range(C.N_CLUSTERS)]},
              open(os.path.join(GEN_DATA, "donor_summary.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"-> {F_GENOME}\n-> {GEN_DATA}/donor_summary.json")

if __name__ == "__main__":
    main()
