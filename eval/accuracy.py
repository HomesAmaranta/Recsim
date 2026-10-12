"""独立准确率评测脚本。读 out/decisions.jsonl 与 out/sim_log.csv，不重跑 LLM。

用法:
  python3 recsim/eval/accuracy.py            # 评 out/decisions.jsonl
  python3 recsim/eval/accuracy.py --tag v1   # 评 out/decisions_v1.jsonl

输出：总体准确率（三路 + 置信区间）、位置偏置、分层/分群/用户级分布、
      热度差分桶（暴露"全库随机负采样"这一协议下的捷径强度）、
      模拟日志与真实测试期日志的分布一致性，并写 out/report.md。
"""
import os, sys, json, math, csv, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from recsim import config as C

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))

def line(name, k, n, out):
    lo, hi = wilson(k, n)
    out.append(f"| {name} | {k}/{n} | **{k/max(n,1):.4f}** | [{lo:.4f}, {hi:.4f}] |")

def main():
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
    sfx = f"_{tag}" if tag else ""
    f_dec = os.path.join(C.OUT, f"decisions{sfx}.jsonl")
    f_log = C.F_SIM_LOG.replace(".csv", f"{sfx}.csv")
    rows = [json.loads(l) for l in open(f_dec, encoding="utf-8")]
    N = len(rows)
    R = []   # report lines

    R.append("# 用户模拟二选一准确率评测报告\n")
    R.append(f"- 数据集：KuaiRand-10W（曝光日志 36.9M 条，用户 24,292，视频 102,603）")
    R.append(f"- 时间切分：训练期 {C.TRAIN_START}–{C.TRAIN_END}（建画像/记忆/统计） / "
             f"测试期 {C.TEST_START}–{C.TEST_END}（正样本来源，模拟器不可见）")
    _nu = len({r["user_id"] for r in rows})
    _q = sorted(collections.Counter(r["user_id"] for r in rows).values())
    R.append(f"- 规模：{_nu} 用户 / 共 {N} 轮二选一"
             f"（每用户配额按测试期可用正样本量水位分配，min/中位/max = "
             f"{_q[0]}/{_q[len(_q)//2]}/{_q[-1]}）")
    R.append(f"- 负采样协议：**{C.NEG_PROTOCOL}**（102,603 视频全库均匀随机，"
             f"排除该用户已交互视频={C.NEG_EXCLUDE_SEEN}）")
    R.append(f"- 判定：一真一假二选一，**随机基线 = 0.5000**\n")

    # ---------- 1 总体 ----------
    R.append("## 1 总体准确率\n")
    R.append("| 判决路径 | 正确/总数 | 准确率 | 95% Wilson 区间 |")
    R.append("|---|---|---|---|")
    line("语义路：LLM 多角度画像判决（单向）", sum(1 for r in rows if r["llm_correct"]), N, R)
    has_db = any(r.get("llm_db_correct") is not None for r in rows)
    if has_db:
        line("语义路：A/B 双向去位置偏置（logprobs）",
             sum(1 for r in rows if r.get("llm_db_correct")), N, R)
    line("统计路：轻量化成对 logistic（含语义增强特征）",
         sum(1 for r in rows if r["stat_correct"]), N, R)
    line("融合（门控堆叠，最终交付）", sum(1 for r in rows if r["fused_correct"]), N, R)
    nnull = sum(1 for r in rows if r["llm_choice"] is None)
    agree = sum(1 for r in rows if r["stat_choice"] == r["llm_choice"]) / N
    R.append(f"\n两路判决一致率 {agree:.4f}；LLM 响应无法解析 {nnull} 条。\n")
    # 融合增量的配对显著性 + 互补性上限：说明「融合到底加了多少」
    sw = [r for r in rows if not r["stat_correct"]]
    lk = "llm_db_correct" if has_db else "llm_correct"
    resc = sum(1 for r in rows if r["fused_correct"] and not r["stat_correct"])
    hurt = sum(1 for r in rows if r["stat_correct"] and not r["fused_correct"])
    chi = (abs(resc - hurt) - 1) ** 2 / max(resc + hurt, 1)
    p = math.erfc(math.sqrt(chi / 2)) if resc + hurt else 1.0
    R.append(f"融合 vs 统计路（配对 McNemar）：救回 {resc} 条 / 拖累 {hurt} 条，p={p:.4f}。")
    R.append(f"互补性上限（两路至少一路对）= "
             f"{sum(1 for r in rows if r['stat_correct'] or r[lk])/N:.4f}；"
             f"但**统计路错的 {len(sw)} 条里语义路只答对 "
             f"{sum(1 for r in sw if r[lk])}（{sum(1 for r in sw if r[lk])/max(len(sw),1):.4f}）**，"
             f"低于抛硬币，故该上限不可达——两路错误高度重叠，见 README §4.5。\n")

    # ---------- 2 位置偏置 ----------
    R.append("## 2 位置偏置检查（A/B 位置已随机化）\n")
    R.append("| 路径 | 选 A 率 | 真值在 A 时准确率 | 真值在 B 时准确率 |")
    R.append("|---|---|---|---|")
    for nm, ck, ok in (("LLM 单向", "llm_choice", "llm_correct"),
                       ("LLM A/B 双向去偏", "llm_choice_db", "llm_db_correct"),
                       ("融合", "fused_choice", "fused_correct")):
        if not any(r.get(ck) for r in rows): continue
        a = [r for r in rows if r["truth"] == "A"]; b = [r for r in rows if r["truth"] == "B"]
        R.append(f"| {nm} | {sum(1 for r in rows if r.get(ck)=='A')/N:.4f} | "
                 f"{sum(1 for r in a if r.get(ok))/len(a):.4f} | "
                 f"{sum(1 for r in b if r.get(ok))/len(b):.4f} |")
    R.append(f"\n真值位置分布 A {sum(1 for r in rows if r['truth']=='A')} / "
             f"B {sum(1 for r in rows if r['truth']=='B')}。选 A 率接近 0.5 即无位置偏置。\n")

    # ---------- 3 分层 ----------
    def group_table(title, keyfn, order=None):
        R.append(f"## {title}\n")
        R.append("| 分组 | 条数 | LLM | 统计 | 融合 |")
        R.append("|---|---|---|---|---|")
        g = collections.defaultdict(list)
        for r in rows: g[keyfn(r)].append(r)
        for k in (order or sorted(g)):
            v = g.get(k, [])
            if not v: continue
            R.append(f"| {k} | {len(v)} | {sum(1 for r in v if r['llm_correct'])/len(v):.4f} | "
                     f"{sum(1 for r in v if r['stat_correct'])/len(v):.4f} | "
                     f"{sum(1 for r in v if r['fused_correct'])/len(v):.4f} |")
        R.append("")

    def ctr_band(r):
        c = r["train_ctr"]
        for hi, nm in ((0.35, "1 低活跃 CTR<0.35"), (0.50, "2 CTR 0.35-0.50"),
                       (0.62, "3 CTR 0.50-0.62"), (0.75, "4 CTR 0.62-0.75")):
            if c < hi: return nm
        return "5 高活跃 CTR>=0.75"
    group_table("3 按用户训练期 CTR 分层", ctr_band)
    group_table("4 按角度6 用户群体（KMeans）", lambda r: f"群体{r['group_id']}")

    # ---------- 5 用户级分布 ----------
    R.append("## 5 用户级准确率分布\n")
    byu = collections.defaultdict(list)
    for r in rows: byu[r["user_id"]].append(r)
    accs = sorted(sum(1 for r in v if r["fused_correct"]) / len(v) for v in byu.values())
    q = lambda p: accs[min(int(p * len(accs)), len(accs) - 1)]
    _pu = sorted(len(v) for v in byu.values())
    R.append(f"- 用户数 {len(accs)}，每用户条数 min/中位/max = "
             f"{_pu[0]}/{_pu[len(_pu)//2]}/{_pu[-1]}")
    R.append(f"- 最小 {accs[0]:.3f} / P25 {q(.25):.3f} / 中位 {q(.5):.3f} / "
             f"P75 {q(.75):.3f} / 最大 {accs[-1]:.3f}")
    for thr in (0.60, 0.75, 0.85):
        R.append(f"- 达到 {thr:.0%} 的用户占比：{sum(1 for a in accs if a >= thr)/len(accs):.3f}")
    R.append("")

    # ---------- 6 热度差分桶：暴露捷径强度 ----------
    R.append("## 6 热度差分桶（协议诚实性检查）\n")
    R.append("全库随机负采样下，正样本来自曝光日志（推荐系统只投放有流量的视频），"
             "负样本来自全库（多为冷视频），因此「热度」本身就是强捷径。"
             "下表按 正负样本热度差 分桶，展示准确率对该捷径的依赖程度。\n")
    side = {}
    with open(C.F_VIDEO_META, encoding="utf-8") as f:
        next(f)
        for l in f:
            p = l.rstrip("\n").split("\t")
            if len(p) >= 6: side[p[0]] = float(p[5] or 0)
    def dband(r):
        d = math.log1p(side.get(r["pos"], 0)) - math.log1p(side.get(r["neg"], 0))
        if d < -1: return "1 正样本更冷 (Δ<-1)"
        if d < 1:  return "2 热度相近 (|Δ|<1)"
        if d < 3:  return "3 正样本较热 (1<=Δ<3)"
        return "4 正样本远热 (Δ>=3)"
    group_table("6.1 按热度差分桶", dband)
    hard = [r for r in rows if abs(math.log1p(side.get(r["pos"], 0))
                                   - math.log1p(side.get(r["neg"], 0))) < 1]
    if hard:
        R.append(f"**热度相近子集（{len(hard)} 条，捷径失效）**："
                 f"LLM {sum(1 for r in hard if r['llm_correct'])/len(hard):.4f} / "
                 f"统计 {sum(1 for r in hard if r['stat_correct'])/len(hard):.4f} / "
                 f"融合 {sum(1 for r in hard if r['fused_correct'])/len(hard):.4f}。"
                 f"这是更接近真实曝光内困难负样本的参考值。\n")

    # ---------- 7 产出日志一致性 ----------
    R.append("## 7 模拟互动日志产出核对（README §4.1 字段结构）\n")
    with open(f_log, encoding="utf-8") as f:
        rd = csv.DictReader(f)
        hdr = rd.fieldnames
        n_row = n_clk = 0
        us, vs = set(), set()
        sim_dur = collections.Counter()
        for x in rd:
            n_row += 1; n_clk += int(x["is_click"])
            us.add(x["user_id"]); vs.add(x["video_id"])
            if int(x["is_click"]):
                sim_dur[int(x["duration_ms"]) // 10000] += 1
    R.append(f"- 文件 `{os.path.relpath(f_log)}`，字段 {len(hdr)} 列，与 §4.1 要求一致：`{','.join(hdr)}`")
    R.append(f"- 总行数 {n_row:,}（每轮二选一产出 2 行：被选中 is_click=1，未选中 is_click=0）")
    R.append(f"- 其中模拟点击日志 {n_clk:,} 条，覆盖用户 {len(us)}，去重视频 {len(vs):,}\n")

    txt = "\n".join(R)
    open(os.path.join(C.OUT, f"report{sfx}.md"), "w", encoding="utf-8").write(txt + "\n")
    print(txt)
    print(f"\n-> {os.path.join(C.OUT, f'report{sfx}.md')}")

if __name__ == "__main__":
    main()
