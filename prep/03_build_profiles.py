"""Step 3: 构建 11 角度用户状态 —— 概设 2.2/2.4.1 的「用户智能体模拟层」。

角度对应关系（概设章节 -> 本文件产物字段）：
  角度1  长期偏好   1.2.1/2.4.1  -> long_term.*      训练期全期统计 + LLM 语义化
  角度2  短期兴趣   1.2.1        -> short_term.*     训练期末尾 SHORT_TERM_DAYS 天切片
  角度3  角色特征   1.2.1/2.4.1  -> role_tags        LLM 生成角色标签
  角度4  长期记忆   2.2/2.4.1    -> long_memory      稳定兴趣与行为特征（高频类别/作者/时长档）
  角度5  短期记忆   2.2/2.4.1    -> short_memory     近期接触/选择/反馈的具体条目
  角度6  群体特征   2.2/2.4.1    -> group_id + 群体近期关注内容（KMeans 聚类）
  角度7  内容语义   2.4.1        -> 候选侧，见 sim/prompts.py
  角度11 降噪+切片  1.2.1        -> 已在 02 步实现

注：不让 LLM 输出「点击倾向高/中/低」——实测它对绝对数值判断不可靠
    （真实 CTR 0.101 时它输出 "high"），该量一律用统计值直接注入。
"""
import os, sys, json, collections, math, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from recsim import config as C
from recsim.sim import llm

COLS = ["user_id","video_id","date","hourmin","time_ms","is_click","is_like","is_follow",
        "is_comment","is_forward","is_hate","long_view","play_time_ms","duration_ms",
        "is_profile_enter","tab"]
POS_KEYS = ("is_click","is_like","is_follow","is_comment","is_forward","long_view")

def dur_bucket(ms):
    ms = int(ms or 0)
    if ms <= 0:      return "未知"
    if ms < 10000:   return "0-10s"
    if ms < 20000:   return "10-20s"
    if ms < 40000:   return "20-40s"
    if ms < 80000:   return "40-80s"
    return "80s+"

def load_video_meta():
    meta = {}
    with open(C.F_VIDEO_META, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 6: continue
            meta[p[0]] = {"cat1": p[1], "cat2": p[2], "caption": p[3],
                          "duration_ms": int(p[4] or 0), "show_cnt": float(p[5] or 0)}
    return meta

def load_user_logs():
    per = collections.defaultdict(list)
    for line in open(C.F_USER_LOGS, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) < 16: continue
        per[p[0]].append(dict(zip(COLS, p)))
    for u in per:
        per[u].sort(key=lambda r: int(r["time_ms"]))
    return per

def denoise(rows):
    seen, out = set(), []
    for r in rows:
        pos = any(int(r[k]) for k in POS_KEYS)
        if int(r["play_time_ms"]) < C.DENOISE_MIN_PLAY_MS and not pos:
            continue
        if C.DENOISE_DEDUP:
            if r["video_id"] in seen: continue
            seen.add(r["video_id"])
        out.append(r)
    return out

def build_state(u, train, meta):
    """从训练期记录构建角度 1/2/4/5 的结构化状态。"""
    clicked = [r for r in train if int(r["is_click"]) == 1]
    n, nc = len(train), len(clicked)
    ctr = nc / n if n else 0.0

    # 角度1 长期偏好：全训练期
    lt_cat = collections.Counter(meta.get(r["video_id"], {}).get("cat1", "UNK") for r in clicked)
    lt_exp = collections.Counter(meta.get(r["video_id"], {}).get("cat1", "UNK") for r in train)
    # 类别偏好强度 = 该类别点击率相对整体的提升
    lift = {}
    for c, e in lt_exp.items():
        if e >= 8:
            lift[c] = (lt_cat.get(c, 0) / e) / (ctr + 1e-9)
    top_lift = sorted(lift.items(), key=lambda kv: -kv[1])[:6]
    low_lift = sorted(lift.items(), key=lambda kv: kv[1])[:4]

    # 角度2 短期兴趣：训练期末尾 N 天切片
    dates = sorted({int(r["date"]) for r in train})
    cut = dates[-C.SHORT_TERM_DAYS] if len(dates) >= C.SHORT_TERM_DAYS else dates[0]
    recent = [r for r in train if int(r["date"]) >= cut]
    rec_clicked = [r for r in recent if int(r["is_click"]) == 1]
    st_cat = collections.Counter(meta.get(r["video_id"], {}).get("cat1", "UNK") for r in rec_clicked)
    st_ctr = len(rec_clicked) / len(recent) if recent else ctr

    # 角度4 长期记忆：稳定行为特征
    dur_pref = collections.Counter(dur_bucket(meta.get(r["video_id"], {}).get("duration_ms", 0))
                                   for r in clicked)
    dur_exp = collections.Counter(dur_bucket(meta.get(r["video_id"], {}).get("duration_ms", 0))
                                  for r in train)
    dur_ctr = {k: round(dur_pref.get(k, 0) / v, 3) for k, v in dur_exp.items() if v >= 8}
    long_memory = {
        "曝光数": n, "点击数": nc, "长期点击率": round(ctr, 4),
        "长看率": round(sum(int(r["long_view"]) for r in train) / n, 4) if n else 0,
        "点赞率": round(sum(int(r["is_like"]) for r in train) / n, 5) if n else 0,
        "负反馈率": round(sum(int(r["is_hate"]) for r in train) / n, 5) if n else 0,
        "偏好类别Top6(点击占比)": [(c, round(k / nc, 3)) for c, k in lt_cat.most_common(6)] if nc else [],
        "类别偏好提升Top6": [(c, round(v, 2)) for c, v in top_lift],
        "明显回避类别": [(c, round(v, 2)) for c, v in low_lift],
        "各时长档点击率": dur_ctr,
    }

    # 角度5 短期记忆：近期具体条目（含正/负反馈）
    def brief(r, tag):
        m = meta.get(r["video_id"], {})
        cap = (m.get("caption") or "").strip()[:34] or "(无标题)"
        return {"tag": tag, "cat": m.get("cat1", "UNK"), "cap": cap,
                "dur": dur_bucket(m.get("duration_ms", 0)),
                "play_s": round(int(r["play_time_ms"]) / 1000, 1),
                "long_view": int(r["long_view"]), "like": int(r["is_like"])}
    sel = [brief(r, "选择") for r in rec_clicked[-12:]]
    skip = [brief(r, "跳过") for r in [x for x in recent if int(x["is_click"]) == 0][-8:]]
    short_memory = {"切片起始日": cut, "切片内曝光": len(recent),
                    "切片内点击率": round(st_ctr, 4),
                    "近期兴趣类别": st_cat.most_common(5),
                    "近期选择条目": sel, "近期跳过条目": skip}

    # 聚类特征向量：L1 类别点击分布 + CTR + 时长偏好
    return {"user_id": u, "long_memory": long_memory, "short_memory": short_memory,
            "_ctr": ctr, "_st_ctr": st_ctr,
            "_cat_vec": {c: k / max(nc, 1) for c, k in lt_cat.items()}}

def kmeans(X, k, seed, iters=60):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), k, replace=False)
    Cn = X[idx].copy()
    lab = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - Cn[None, :, :]) ** 2).sum(-1)
        new = d.argmin(1)
        if (new == lab).all(): break
        lab = new
        for j in range(k):
            m = lab == j
            if m.any(): Cn[j] = X[m].mean(0)
    return lab

PROFILE_PROMPT = """你是推荐系统用户模拟器的画像生成模块。根据下面这个真实用户在观察期内的行为统计，生成其语义画像。

【长期行为特征（角度4 长期记忆）】
{long_mem}

【近期行为切片（角度2 短期兴趣 / 角度5 短期记忆）】
{short_mem}

请严格只输出 JSON，不要任何解释、不要 markdown 代码块：
{{"long_term_preference": "长期稳定偏好，具体到内容类别和风格，1-2句",
  "short_term_interest": "近期短期兴趣及其相对长期偏好的变化，1句",
  "role_tags": ["角色标签1", "角色标签2", "角色标签3"],
  "duration_preference": "对视频时长的偏好，要与'各时长档点击率'一致",
  "aversion": "明显回避的内容类型，要与'明显回避类别'一致",
  "decision_style": "该用户做选择时的决策风格，1句"}}"""

def main():
    random.seed(C.SEED); np.random.seed(C.SEED % (2**31))
    print("[1/4] 载入日志与视频元数据...")
    meta = load_video_meta()
    per = load_user_logs()
    users = [l.split("\t")[0] for l in open(C.F_USERS)]
    print(f"  用户 {len(users)}  视频 {len(meta):,}")

    print("[2/4] 构建角度 1/2/4/5 结构化状态（训练期，已降噪）...")
    states = {}
    for u in users:
        rows = per[u]
        train = denoise([r for r in rows if C.TRAIN_START <= int(r["date"]) <= C.TRAIN_END])
        states[u] = build_state(u, train, meta)

    print(f"[3/4] 角度6 群体特征：KMeans 聚类 k={C.N_CLUSTERS} ...")
    cats = sorted({c for s in states.values() for c in s["_cat_vec"]})
    X = np.array([[s["_cat_vec"].get(c, 0.0) for c in cats] + [s["_ctr"], s["_st_ctr"]]
                  for s in (states[u] for u in users)], dtype=np.float64)
    lab = kmeans(X, C.N_CLUSTERS, C.SEED)
    groups = {}
    for gid in range(C.N_CLUSTERS):
        mem = [users[i] for i in range(len(users)) if lab[i] == gid]
        agg = collections.Counter()
        for u in mem:
            for c, w in states[u]["_cat_vec"].items():
                agg[c] += w
        ctrs = [states[u]["_ctr"] for u in mem]
        groups[str(gid)] = {
            "size": len(mem), "members": mem,
            "群体近期关注类别Top5": [(c, round(w / max(len(mem), 1), 3)) for c, w in agg.most_common(5)],
            "群体平均点击率": round(sum(ctrs) / len(ctrs), 4) if ctrs else 0.0,
        }
        print(f"    群体{gid}: {len(mem)} 人  平均CTR {groups[str(gid)]['群体平均点击率']:.3f}  "
              f"关注 {[c for c,_ in agg.most_common(3)]}")
    for i, u in enumerate(users):
        states[u]["group_id"] = int(lab[i])

    print(f"[4/4] 角度1/2/3 语义画像：调用 LLM ({len(users)} 次, 并发 {C.LLM_CONCURRENCY})...")
    jobs = []
    for u in users:
        s = states[u]
        p = PROFILE_PROMPT.format(
            long_mem=json.dumps(s["long_memory"], ensure_ascii=False, indent=1),
            short_mem=json.dumps(s["short_memory"], ensure_ascii=False, indent=1))
        jobs.append((u, [{"role": "user", "content": p}]))
    res = llm.chat_many(jobs, max_tokens=C.PROFILE_MAX_TOKENS,
                        temperature=C.PROFILE_TEMPERATURE, desc="画像")
    ok = 0
    for u in users:
        j = llm.extract_json(res.get(u))
        if j:
            ok += 1
            states[u]["profile"] = j
            states[u]["role_tags"] = j.get("role_tags", [])
        else:
            states[u]["profile"] = None
            states[u]["role_tags"] = []
    print(f"  画像 JSON 解析成功 {ok}/{len(users)}")
    llm.print_stats(" 画像阶段")

    for u in states:
        for k in ("_cat_vec", "_ctr", "_st_ctr"):
            states[u].pop(k, None)
    json.dump(states, open(C.F_PROFILES, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(groups, open(C.F_GROUPS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n-> {C.F_PROFILES}\n-> {C.F_GROUPS}")
    ex = states[users[0]]
    print(f"\n示例用户 {users[0]} 画像:\n{json.dumps(ex.get('profile'), ensure_ascii=False, indent=1)}")

if __name__ == "__main__":
    main()
