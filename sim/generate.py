"""Step 4: 二选一（2AFC）样本生成 + 模拟互动日志产出。

协议（生成与评测同一套，写死在 config，不可中途换）：
  正样本  : 该用户【测试期】真实点击过的视频（模拟器建画像时看不到测试期）
  负样本  : NEG_PROTOCOL="global_random" —— 从 102,603 视频全库均匀随机，
            排除该用户在训练期+测试期出现过的任何 video_id
  位置    : A/B 随机化，消除位置偏置；评测时会单独统计位置偏置
  判定    : LLM 只输出一个字母，答对即"该条模拟正确"，随机基线 = 50%

产物：
  out/sim_log.csv    按 README §4.1 的 19 字段结构（模拟用户互动日志）
                     每轮二选一产出 2 行：被选中的 is_click=1，未被选中的 is_click=0
  out/sim_trace.jsonl 逐条决策明细，供 eval/accuracy.py 计算准确率

注：日志中 play_time_ms / long_view / is_like 等辅助字段无法从二选一里得到，
    采用「该用户训练期内同时长档的经验分布」重采样填充，属轻量化统计模型，
    不参与准确率计算，仅为了让产出日志字段结构完整可用。
"""
import os, sys, json, random, collections, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from recsim import config as C
from recsim.sim import llm, prompts
from recsim.sim import features as F

COLS = ["user_id","video_id","date","hourmin","time_ms","is_click","is_like","is_follow",
        "is_comment","is_forward","is_hate","long_view","play_time_ms","duration_ms",
        "is_profile_enter","tab"]
OUT_COLS = ["user_id","video_id","date","hourmin","time_ms","is_click","is_like","is_follow",
            "is_comment","is_forward","is_hate","long_view","play_time_ms","duration_ms",
            "profile_stay_time","comment_stay_time","is_profile_enter","is_rand","tab"]

DUR_BUCKETS = [(0, 10000, "0-10s"), (10000, 20000, "10-20s"), (20000, 40000, "20-40s"),
               (40000, 80000, "40-80s"), (80000, 10**12, "80s+")]

def dur_label(ms):
    ms = int(ms or 0)
    if ms <= 0: return "未知"
    for lo, hi, name in DUR_BUCKETS:
        if lo <= ms < hi: return name
    return "80s+"

def load_meta():
    """读 26 列全量元数据。返回 (meta, 全库id, 有标题id)。"""
    meta, ids, cap_ids = {}, [], []
    with open(C.F_VIDEO_META, encoding="utf-8") as f:
        cols = f.readline().rstrip("\n").split("\t")
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(cols): continue
            m = dict(zip(cols, p))
            m["duration_ms"] = int(m["duration_ms"] or 0)
            m["dur_label"] = dur_label(m["duration_ms"])
            meta[p[0]] = m
            ids.append(p[0])
            if (m.get("caption") or "").strip():
                cap_ids.append(p[0])
    return meta, ids, cap_ids

def load_logs():
    per = collections.defaultdict(list)
    for line in open(C.F_USER_LOGS, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) < 16: continue
        per[p[0]].append(dict(zip(COLS, p)))
    for u in per:
        per[u].sort(key=lambda r: int(r["time_ms"]))
    return per

def behavior_model(train, meta):
    """从训练期学该用户的辅助字段经验分布：按时长档存 (播放占比, long_view, like...) 样本池。"""
    pool = collections.defaultdict(list)
    for r in train:
        if int(r["is_click"]) != 1: continue
        d = int(r["duration_ms"] or meta.get(r["video_id"], {}).get("duration_ms", 0) or 0)
        if d <= 0: continue
        ratio = min(int(r["play_time_ms"]) / d, 8.0)
        pool[dur_label(d)].append((ratio, int(r["long_view"]), int(r["is_like"]),
                                   int(r["is_follow"]), int(r["is_comment"]),
                                   int(r["is_forward"]), int(r["is_profile_enter"])))
    allp = [x for v in pool.values() for x in v]
    return pool, allp

def fill_click_row(rng, pool, allp, dl):
    cand = pool.get(dl) or allp
    if not cand:
        return (0.6, 0, 0, 0, 0, 0, 0)
    return rng.choice(cand)

def ab_logit(lp, floor=-14.0, cap=12.0):
    """把首 token 的 top_logprobs 折成「选 A 的对数几率」= logP(A) - logP(B)。
    A 或 B 掉出 top-10 时按 floor 兜底（说明模型极其确定另一边）。"""
    if not lp: return None
    a = lp.get("A", lp.get("Ａ", lp.get("a")))
    b = lp.get("B", lp.get("Ｂ", lp.get("b")))
    if a is None and b is None: return None
    if a is None: a = floor
    if b is None: b = floor
    return max(-cap, min(cap, a - b))

def pos_candidates(test, meta):
    """该用户测试期真实点击过的视频，按 video_id 去重、要求有标题。正样本池。"""
    out, dedup = [], set()
    for r in test:
        if (int(r["is_click"]) == 1 and r["video_id"] not in dedup and r["video_id"] in meta
                and (not C.REQUIRE_CAPTION or (meta[r["video_id"]].get("caption") or "").strip())):
            dedup.add(r["video_id"]); out.append(r)
    return out

def waterfill(avail, total):
    """水位法配额：取最小的 q 使 sum_u min(avail_u, q) >= total，再削到恰好 total。

    为什么不用「每人固定条数」：各用户测试期可用正样本量差异极大（实测 P25=84 / P90=393），
    固定条数要么被最少的人卡死（25 条/人只能出 5,000 条），要么大量用户填不满。
    水位法让活跃用户多出、低活用户出满即止，既凑够总量又不改变用户集本身。

    削减规则固定为「按 user_id 升序对 avail>=q 的用户各减 1」，与随机数无关，可复现。
    """
    if not avail: return {}, 0
    lo, hi = 1, max(avail.values())
    while lo < hi:                                    # 二分最小水位
        mid = (lo + hi) // 2
        if sum(min(v, mid) for v in avail.values()) >= total: hi = mid
        else: lo = mid + 1
    q = lo
    quota = {u: min(v, q) for u, v in avail.items()}
    over = sum(quota.values()) - total
    if over > 0:                                      # 只削「顶到水位」的用户，低活用户不动
        for u in sorted(quota, key=lambda x: int(x)):
            if over <= 0: break
            if avail[u] >= q and quota[u] > 1:
                quota[u] -= 1; over -= 1
    return quota, q

def main():
    os.makedirs(C.OUT, exist_ok=True)
    rng = random.Random(C.SEED)
    # 冒烟测试：python3 generate.py --smoke 3 5  -> 3 用户 × 5 条，产物写 *_smoke.*
    smoke = None
    if "--smoke" in sys.argv:
        i = sys.argv.index("--smoke")
        smoke = (int(sys.argv[i + 1]), int(sys.argv[i + 2]))
    n_users, per_user = smoke if smoke else (C.N_USERS, None)
    # --n 20000：覆盖 config.N_SAMPLES（总轮数）。冒烟模式下忽略，由 --smoke 的两个参数决定。
    n_samples = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else C.N_SAMPLES
    # 角度开关覆盖：--angles cot=0,item_popularity=1   产物加 --tag 后缀
    angles = dict(C.ANGLES)
    if "--angles" in sys.argv:
        for kv in sys.argv[sys.argv.index("--angles") + 1].split(","):
            k, v = kv.split("="); angles[k.strip()] = bool(int(v))
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ("smoke" if smoke else "")
    # --dual：每对再跑一次 A/B 互换的 prompt，用 logprobs 取连续置信度并抵掉位置偏置
    dual = "--dual" in sys.argv
    sfx = f"_{tag}" if tag else ""
    f_log = C.F_SIM_LOG.replace("sim_log.csv", f"sim_log_llm{sfx}.csv")
    f_trace = C.F_SIM_TRACE.replace(".jsonl", f"{sfx}.jsonl")

    print("[1/4] 载入元数据 / 日志 / 画像 ...")
    meta, all_ids, cap_ids = load_meta()
    pool_ids = cap_ids if C.REQUIRE_CAPTION else all_ids   # 候选池（负采样从这里抽）
    side, _ = F.load_video_side()          # 统计路的视频侧特征（算用户x候选交叉量用）
    per = load_logs()
    states = json.load(open(C.F_PROFILES, encoding="utf-8"))
    groups = json.load(open(C.F_GROUPS, encoding="utf-8"))
    users = [l.split("\t")[0] for l in open(C.F_USERS)][:n_users]
    tr_ctr = {l.split("\t")[0]: float(l.split("\t")[5]) for l in open(C.F_USERS)}
    print(f"  全库视频 {len(all_ids):,}  候选池(REQUIRE_CAPTION={C.REQUIRE_CAPTION}) {len(pool_ids):,}")
    print(f"  用户 {len(users)}  已启用角度 "
          f"{[k for k,v in angles.items() if v]}")

    # 配额：冒烟模式每人固定 per_user；正式跑用水位法按可用量分配到恰好 n_samples
    test_of = {u: [r for r in per[u] if C.TEST_START <= int(r["date"]) <= C.TEST_END] for u in users}
    poscand = {u: pos_candidates(test_of[u], meta) for u in users}
    avail = {u: len(v) for u, v in poscand.items()}
    if smoke:
        quota = {u: min(per_user, avail[u]) for u in users}; q = per_user
    else:
        quota, q = waterfill(avail, n_samples)
    qs = sorted(quota.values())
    print(f"[2/4] 构造二选一样本 (总 {sum(quota.values()):,} 轮, 负采样={C.NEG_PROTOCOL}) ...")
    print(f"  可用正样本(测试期/去重/有标题) 合计 {sum(avail.values()):,}  "
          f"水位 q={q}  配额 min/中位/max = {qs[0]}/{qs[len(qs)//2]}/{qs[-1]}")
    if not smoke:
        assert sum(quota.values()) == n_samples, \
            f"配额 {sum(quota.values())} != 目标 {n_samples}（可用总量 {sum(avail.values())} 是否不足？）"
    rounds, jobs = [], []
    for u in users:
        rows = per[u]
        train = [r for r in rows if C.TRAIN_START <= int(r["date"]) <= C.TRAIN_END]
        seen = {r["video_id"] for r in rows} if C.NEG_EXCLUDE_SEEN else set()
        pool, allp = behavior_model(train, meta)
        # 用户 x 作者 交叉：训练期该作者被推过几次、点了几次
        ah = {}
        for x in train:
            m = meta.get(x["video_id"])
            if not m: continue
            a = m.get("author_id", "UNK")
            e, c = ah.get(a, (0, 0))
            ah[a] = (e + 1, c + int(x["is_click"]))
        us = F.user_state(train, side)      # 统计路用户侧计数（仅训练期）

        take = rng.sample(poscand[u], quota[u])

        for i, pr in enumerate(take):
            # 负样本：全库均匀随机，排除已见
            while True:
                nv = pool_ids[rng.randrange(len(pool_ids))]
                if nv not in seen and nv in meta:
                    break
            pv = pr["video_id"]
            truth = "A" if rng.random() < 0.5 else "B"
            cands = ([("A", meta[pv]), ("B", meta[nv])] if truth == "A"
                     else [("A", meta[nv]), ("B", meta[pv])])
            aff = {tg: F.affinity_desc(us, side[m["video_id"]]) for tg, m in cands
                   if m["video_id"] in side}
            key = f"{u}#{i}"
            rounds.append({"key": key, "user_id": u, "round": i, "pos": pv, "neg": nv,
                           "truth": truth, "group_id": states[u].get("group_id"),
                           "train_ctr": tr_ctr[u],
                           "date": pr["date"], "hourmin": pr["hourmin"], "time_ms": pr["time_ms"],
                           "_pool": pool, "_allp": allp})
            jobs.append((key, prompts.build_decide_prompt(states[u], groups, cands, angles, ah, aff)))
            if dual:
                sw = [("A", cands[1][1]), ("B", cands[0][1])]        # A/B 互换
                jobs.append((key + "~s",
                             prompts.build_decide_prompt(states[u], groups, sw, angles, ah, aff)))
    print(f"  二选一轮次 {len(rounds):,}  (真值位置 A {sum(1 for r in rounds if r['truth']=='A')} / "
          f"B {sum(1 for r in rounds if r['truth']=='B')})")
    print(f"  单条 prompt 字符数 中位 ~{sorted(len(j[1][1]['content']) for j in jobs)[len(jobs)//2]}")

    print(f"[3/4] 调用 LLM 判决 ({len(jobs):,} 次, 并发 {C.DECIDE_CONCURRENCY}"
          f"{', A/B 双向+logprobs' if dual else ''}) ...")
    res = llm.chat_many(jobs, max_tokens=C.DECIDE_MAX_TOKENS,
                        temperature=C.DECIDE_TEMPERATURE,
                        concurrency=C.DECIDE_CONCURRENCY, desc="判决", logprobs=dual)
    llm.print_stats(" 判决阶段")

    print("[4/4] 写模拟日志与 trace ...")
    n_ok = n_bad = n_null = 0
    n_dual_ok = n_dual_n = 0
    fl = open(f_log, "w", newline="", encoding="utf-8")
    w = csv.writer(fl); w.writerow(OUT_COLS)
    ft = open(f_trace, "w", encoding="utf-8")
    for r in rounds:
        got = res.get(r["key"])
        raw, lp1 = got if dual else (got, None)
        ch = prompts.parse_choice(raw)
        # 连续置信度：s1 = 正序下「选A」的对数几率，s2 = 互换后「选A」（其实是原B）的对数几率
        # 去位置偏置的分数 sA = (s1 - s2)/2 —— 任何对「排前面」的固定偏好在相减中抵消
        s1 = s2 = sA = None
        if dual:
            s1 = ab_logit(lp1 or {})
            _, lp2 = res.get(r["key"] + "~s", (None, {}))
            s2 = ab_logit(lp2 or {})
            if s1 is not None and s2 is not None:
                sA = (s1 - s2) / 2.0
                n_dual_n += 1
                n_dual_ok += (("A" if sA > 0 else "B") == r["truth"])
        if ch is None:
            n_null += 1
            correct = None
        else:
            correct = (ch == r["truth"])
            n_ok += correct; n_bad += (not correct)
        chosen   = r["pos"] if ch == r["truth"] else r["neg"]
        rejected = r["neg"] if ch == r["truth"] else r["pos"]

        ft.write(json.dumps({k: r[k] for k in
                             ("key","user_id","round","pos","neg","truth","group_id","train_ctr","date","hourmin","time_ms")}
                            | {"choice": ch, "correct": correct,
                               "llm_s1": s1, "llm_s2": s2, "llm_score": sA,
                               "chosen_video": (chosen if ch else None),
                               "raw": (raw or "")[:400]}, ensure_ascii=False) + "\n")
        if ch is None:
            continue
        # 模拟互动日志：被选中的记 is_click=1，另一个记 is_click=0
        for vid, clicked in ((chosen, 1), (rejected, 0)):
            m = meta[vid]
            d = m["duration_ms"]
            if clicked:
                ratio, lv, lk, fo, cm, fw, pe = fill_click_row(rng, r["_pool"], r["_allp"],
                                                               m["dur_label"])
                play = int(ratio * d)
            else:
                ratio, lv, lk, fo, cm, fw, pe = 0, 0, 0, 0, 0, 0, 0
                play = int(rng.uniform(0.02, 0.25) * d)
            w.writerow([r["user_id"], vid, r["date"], r["hourmin"], r["time_ms"], clicked,
                        lk, fo, cm, fw, 0, lv, play, d, 0, 0, pe, 0, 1])
    fl.close(); ft.close()

    n = n_ok + n_bad
    print(f"\n{'='*58}")
    print(f"二选一准确率 = {n_ok}/{n} = {n_ok/max(n,1):.4f}   (随机基线 0.5000)")
    if dual and n_dual_n:
        print(f"A/B 双向去偏 (sign of (s1-s2)/2) = {n_dual_ok}/{n_dual_n} = {n_dual_ok/n_dual_n:.4f}")
    print(f"无法解析 A/B 的响应 {n_null} 条")
    print(f"负采样协议: {C.NEG_PROTOCOL} (排除已见={C.NEG_EXCLUDE_SEEN})")
    print(f"{'='*58}")
    print(f"-> {f_log}\n-> {f_trace}")
    print("详细分组准确率请跑: python3 recsim/eval/accuracy.py")

if __name__ == "__main__":
    main()
