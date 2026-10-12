"""Step A（新用户线）：8 群 × 200 = 1,600 个代表性智能体画像。**唯一用 LLM 的一步。**

概设文档 §1.2.1 / §2.2 的原话是「采用用户聚类、代表性智能体和轻量化模型等方式提高
大规模模拟效率」「选取代表性用户使用大模型进行模拟，再根据实际需要采用轻量化方式
扩展用户规模」。这一步就是「选取代表性用户 + 大模型」，后面 03/04 是「轻量化扩展」。

50 万新用户 ÷ 1,600 个原型 ≈ 每个原型被 313 个新用户继承画像文本。
LLM 只在这里调 1,600 次（约 10-15 分钟），绝不进 04 的逐条曝光循环
（50M 次决策 ÷ 20 req/s ≈ 29 天，物理上不可行；且 FUSION.md §5 已实测决策级
LLM 投票对准确率增量为零）。

流程：
  1. 从 01 产出的 donor_genomes.jsonl 里，按 group_id 分组，**组内按活跃度分 10 层**
     各取 20 人 = 200 个种子（不分层会只抽到头部重度用户，画像同质化）。
  2. awk 一趟扫 36.9M 行，只抽这 1,600 人的日志。
  3. 复用 prep/03_build_profiles.py 的 denoise / build_state / PROFILE_PROMPT，
     **一个字都不改** —— 保证原型画像与旧用户线的画像同口径，
     features.load_profile_sem() 能直接吃。
  4. 落 data/gen/archetypes.json。

与旧用户线的唯一差别：**不做训练/测试期切分**，整个 20220408-20220508 都当观察期。
新用户是凭空造的，不存在「测试期真值」，没有可泄漏的对象。

用法：
  <py> recsim/gen/02_archetypes.py --smoke 8 2     # 8 群 × 2 人 = 16 份，人工看质量
  <py> recsim/gen/02_archetypes.py                 # 正式：8 群 × 200 = 1,600 份
"""
import os, sys, json, collections, subprocess, importlib.util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from recsim import config as C
from recsim.sim import llm

GEN_DATA  = os.path.join(C.DATA, "gen")
F_GENOME  = os.path.join(GEN_DATA, "donor_genomes.jsonl")
F_SEEDS   = os.path.join(GEN_DATA, "archetype_seeds.txt")
F_ALOGS   = os.path.join(GEN_DATA, "archetype_logs.tsv")
F_ARCH    = os.path.join(GEN_DATA, "archetypes.json")

N_STRATA  = 10        # 组内活跃度分层数
SEEDS_PER_GROUP = 200

# 复用 prep/03 的实现。模块名以数字开头不能 import，用 importlib 按路径加载。
def load_prep03():
    p = os.path.join(C.ROOT, "prep", "03_build_profiles.py")
    spec = importlib.util.spec_from_file_location("prep03", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

# 与 prep/02_extract_logs.py 的 AWK_LOG 同口径，额外多带一列 comment_stay_time（f[16]）——
# 新用户线要复现这个字段的非零分布（实测点击行 13.2% 非零），旧用户线沿用硬编码 0。
# 源 CSV 是 CRLF。prep/02 里 tab 是最后一列，"1\r\n" 被 Python 的 universal newline
# 当成行尾一起吃掉，看不出问题；这里 comment_stay 排在 tab 之后，\r 落到行中间，
# Python 会把它当换行符**把每行劈成两半**（实测 6,131 行读出 12,262 条全部字段不足）。
# 所以先 gsub 掉。
AWK_LOG = r'''
BEGIN { while ((getline u < UF) > 0) { split(u, a, "\t"); keep[a[1]] = 1 } }
NR==1 { next }
{
  gsub(/\r/, "")
  split($0, f, ",")
  if (!(f[1] in keep)) next
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
    f[1],f[2],f[3],f[4],f[5],f[6],f[7],f[8],f[9],f[10],f[11],f[12],f[13],f[14],f[17],f[19],f[16]
}
'''
ACOLS = ["user_id","video_id","date","hourmin","time_ms","is_click","is_like","is_follow",
         "is_comment","is_forward","is_hate","long_view","play_time_ms","duration_ms",
         "is_profile_enter","tab","comment_stay_time"]

def pick_seeds(genomes, per_group, n_strata=N_STRATA):
    """组内按活跃度（n_expose）分层等额抽种子。分层是为了让原型覆盖
    「轻度-中度-重度」全谱；直接按 CTR 或随机抽都会被头部用户主导。
    确定性：层内按 user_id 升序等间隔取，不用随机数。"""
    byg = collections.defaultdict(list)
    for g in genomes:
        byg[g["group_id"]].append(g)
    seeds = []
    for gid in sorted(byg):
        mem = sorted(byg[gid], key=lambda x: (x["n_expose"], int(x["user_id"])))
        k = min(per_group, len(mem))
        per_s = max(1, k // n_strata)
        got = []
        for si in range(n_strata):
            lo = len(mem) * si // n_strata
            hi = len(mem) * (si + 1) // n_strata
            band = sorted(mem[lo:hi], key=lambda x: int(x["user_id"]))
            if not band: continue
            need = min(per_s, len(band))
            step = max(1, len(band) // need)
            got += [band[i * step] for i in range(need)]
        # 分层取整可能少几个，用该组剩余用户按 user_id 升序补齐
        have = {g["user_id"] for g in got}
        for g in sorted(mem, key=lambda x: int(x["user_id"])):
            if len(got) >= k: break
            if g["user_id"] not in have:
                got.append(g); have.add(g["user_id"])
        seeds += got[:k]
    return seeds

def main():
    smoke = None
    if "--smoke" in sys.argv:
        i = sys.argv.index("--smoke")
        smoke = (int(sys.argv[i + 1]), int(sys.argv[i + 2]))
    per_group = smoke[1] if smoke else SEEDS_PER_GROUP
    sfx = "_smoke" if smoke else ""
    f_seeds = F_SEEDS.replace(".txt", f"{sfx}.txt")
    f_alogs = F_ALOGS.replace(".tsv", f"{sfx}.tsv")
    f_arch  = F_ARCH.replace(".json", f"{sfx}.json")

    print("[1/5] 载入 donor 基因 ...")
    genomes = [json.loads(l) for l in open(F_GENOME, encoding="utf-8")]
    print(f"  donor {len(genomes):,} 人 / {C.N_CLUSTERS} 群")

    print(f"[2/5] 每群按活跃度分 {N_STRATA} 层，各取 {per_group} 个种子 ...")
    seeds = pick_seeds(genomes, per_group)
    gsz = collections.Counter(g["group_id"] for g in seeds)
    for gid in sorted(gsz):
        act = sorted(g["n_expose"] for g in seeds if g["group_id"] == gid)
        print(f"    群{gid}: {gsz[gid]} 个种子  曝光 min/中位/max = "
              f"{act[0]}/{act[len(act)//2]}/{act[-1]}")
    with open(f_seeds, "w") as f:
        for g in seeds: f.write(f'{g["user_id"]}\t{g["group_id"]}\n')

    if not os.path.exists(f_alogs):
        print(f"[3/5] awk 抽这 {len(seeds)} 人的日志（扫 36.9M 行，约 1-3 分钟）...")
        with open(f_alogs, "w") as fo:
            subprocess.run(["awk", "-v", f"UF={f_seeds}", AWK_LOG, C.LOG], stdout=fo, check=True)
    else:
        print(f"[3/5] 复用已有 {f_alogs}")

    P = load_prep03()
    meta = P.load_video_meta()
    per = collections.defaultdict(list)
    for line in open(f_alogs, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) < len(ACOLS): continue
        per[p[0]].append(dict(zip(ACOLS, p)))
    for u in per: per[u].sort(key=lambda r: int(r["time_ms"]))

    print(f"[4/5] 构建角度 1/2/4/5 结构化状态（全期 {C.TRAIN_START}-{C.TEST_END}，无需切分）...")
    states, seed_of = {}, {g["user_id"]: g for g in seeds}
    miss = 0
    for g in seeds:
        u = g["user_id"]
        rows = P.denoise(per.get(u, []))
        if not rows: miss += 1; continue
        states[u] = P.build_state(u, rows, meta)
    print(f"  成功 {len(states)}/{len(seeds)}（无日志 {miss}）")

    print(f"[5/5] LLM 生成原型画像（{len(states)} 次，并发 {C.LLM_CONCURRENCY}）...")
    jobs = []
    for u, s in states.items():
        p = P.PROFILE_PROMPT.format(
            long_mem=json.dumps(s["long_memory"], ensure_ascii=False, indent=1),
            short_mem=json.dumps(s["short_memory"], ensure_ascii=False, indent=1))
        jobs.append((u, [{"role": "user", "content": p}]))
    res = llm.chat_many(jobs, max_tokens=C.PROFILE_MAX_TOKENS,
                        temperature=C.PROFILE_TEMPERATURE, desc="原型画像")
    llm.print_stats(" 原型画像")

    arch, ok = [], 0
    for aid, (u, s) in enumerate(sorted(states.items(), key=lambda kv: int(kv[0]))):
        j = llm.extract_json(res.get(u))
        ok += bool(j)
        g = seed_of[u]
        nc = max(g["n_click"], 1)
        arch.append({
            "archetype_id": aid, "group_id": g["group_id"], "seed_user_id": u,
            "profile": j, "role_tags": (j or {}).get("role_tags", []),
            # 类别偏好向量：03_users.py 用它与新用户基因算余弦相似度来绑原型
            "cat_pref": {c: round(v / nc, 5) for c, v in g["cat_click"].items()},
            "ctr": g["ctr"], "n_expose": g["n_expose"],
            "long_memory": s["long_memory"],      # 04 的状态更新用得上，也是过程留痕
        })
    print(f"  画像 JSON 解析成功 {ok}/{len(states)}")
    json.dump(arch, open(f_arch, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"-> {f_arch}")
    if arch:
        e = arch[0]
        print(f"\n示例 原型{e['archetype_id']} (群{e['group_id']} 种子{e['seed_user_id']} "
              f"CTR {e['ctr']:.3f}):\n{json.dumps(e['profile'], ensure_ascii=False, indent=1)}")

if __name__ == "__main__":
    main()
