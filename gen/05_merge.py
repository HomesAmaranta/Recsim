"""Step D（新用户线）：合并分片 + 7 条硬断言 + 统计。

产出最终交付：
  out/gen/sim_log_500k.csv              19 列 ← **交付表之一**
  out/gen/sim_user_features_500k.csv    31 列 ← **交付表之二**（03 已产出，这里只校验）
  out/gen/interaction_records.jsonl     内部留痕，对齐概设 §2.5.2（含 round_no）

7 条硬断言（任一失败就不许交），逐条对应 PLAN §3.6：
  1 两张表列名与顺序逐字符等于 A-README §4.1 / §4.2
  2 日志表 user_id ⊆ 特征表；特征表 user_id 唯一且 ≥1,000,000
  3 日志表 video_id ⊆ 真实 102,603 视频库；duration_ms 等于该视频真实时长
  4 date+hourmin ↔ time_ms 自洽；date ∈ [20220502, 20220508]
  5 is_click=0 的行互动字段全 0；play_time_ms ≤ duration_ms×8
  6 四个 *_range 与数值一致（用 range_maps.json 复算）
  7 is_rand 全 0；is_lowactive_period 全 0；is_live_streamer ∈ {-124, 1}
  8（用户追加要求）每个用户 ≥3 次交互，且 ≥3 次真实点击

大表用流式扫，不 load 进内存（5,000 万行 × 19 列）。

用法：
  <py> recsim/gen/05_merge.py [--plan _5000] [--keep-shards]
"""
import os, sys, json, csv, math, re, datetime, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from recsim import config as C

GEN_DATA = os.path.join(C.DATA, "gen")
OUT_GEN  = os.path.join(C.OUT, "gen")
SHARDS   = os.path.join(OUT_GEN, "shards")
TZ = datetime.timezone(datetime.timedelta(hours=8))
MIN_CLICKS = 3

LOG_COLS = ["user_id","video_id","date","hourmin","time_ms","is_click","is_like","is_follow",
            "is_comment","is_forward","is_hate","long_view","play_time_ms","duration_ms",
            "profile_stay_time","comment_stay_time","is_profile_enter","is_rand","tab"]
USER_COLS = ["user_id","user_active_degree","is_lowactive_period","is_live_streamer",
             "is_video_author","follow_user_num","follow_user_num_range",
             "fans_user_num","fans_user_num_range","friend_user_num","friend_user_num_range",
             "register_days","register_days_range"] + [f"onehot_feat{i}" for i in range(18)]
INTERACT = ["is_like","is_follow","is_comment","is_forward","is_hate","long_view",
            "is_profile_enter"]

def lookup(buckets, v):
    lab = buckets[0][2]
    for lo, _, l in buckets:
        if v >= lo: lab = l
        else: break
    return lab

def hourmin_of(dt):
    return ((dt.hour + (1 if dt.minute >= 45 else 0)) % 24) * 100

def main():
    plan_tag = sys.argv[sys.argv.index("--plan") + 1] if "--plan" in sys.argv else ""
    f_user = os.path.join(OUT_GEN, f"sim_user_features_500k{plan_tag}.csv")
    f_out  = os.path.join(OUT_GEN, f"sim_log_500k{plan_tag}.csv")
    f_rec  = os.path.join(OUT_GEN, f"interaction_records{plan_tag}.jsonl")

    print("[1/5] 校验用户特征表（§4.2, 31 列）...")
    rmaps = json.load(open(os.path.join(GEN_DATA, "range_maps.json")))
    uids, bad_range = set(), 0
    with open(f_user, encoding="utf-8") as f:
        rd = csv.reader(f)
        hdr = next(rd)
        assert hdr == USER_COLS, f"断言1 失败：特征表列名/顺序不符\n{hdr}"
        for p in rd:
            r = dict(zip(USER_COLS, p))
            u = int(r["user_id"])
            assert u >= 1000000, f"断言2 失败：user_id {u} < 1,000,000"
            assert u not in uids, f"断言2 失败：user_id {u} 重复"
            uids.add(u)
            assert r["is_lowactive_period"] == "0", "断言7 失败：is_lowactive_period 非 0"
            assert r["is_live_streamer"] in ("-124", "1"), "断言7 失败：is_live_streamer 越界"
            for rf, mp in rmaps.items():
                if lookup(mp["buckets"], int(r[mp["num_field"]])) != r[rf]: bad_range += 1
    assert bad_range == 0, f"断言6 失败：{bad_range} 处 *_range 与数值不一致"
    print(f"  {len(uids):,} 用户 × 31 列，断言 1/2/6/7 通过")

    print("[2/5] 载入真实视频库（断言 3 用）...")
    vdur = {}
    with open(C.F_VIDEO_META, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 5: vdur[p[0]] = int(p[4] or 0)
    print(f"  {len(vdur):,} 个真实视频")

    print("[3/5] 合并日志分片 + 流式断言 3/4/5/7 ...")
    # 分片名形如 log_000.csv（正式）/ log_5000_000.csv（5,000 人验收档）。
    # 用 startswith(f"log{plan_tag}_") 会让正式档把验收档的分片一并吞进来
    # （"log_5000_000.csv" 也是以 "log_" 开头的），而验收用户的 uid
    # 1,000,000-1,004,999 恰好也在 50w 特征表里，断言 2 查不出来，
    # 结果是这 5,000 人凭空多出一份日志。必须按【严格的片号格式】筛。
    pat = re.compile(rf"^log{re.escape(plan_tag)}_\d{{3}}\.csv$")
    parts = sorted(x for x in os.listdir(SHARDS) if pat.match(x))
    assert parts, f"{SHARDS} 里没有 log{plan_tag}_*.csv"
    n_row = n_clk = 0
    per_expose = collections.Counter(); per_click = collections.Counter()
    dates = collections.Counter(); tabs = collections.Counter()
    cs_nonzero = 0
    fo = open(f_out, "w", newline="", encoding="utf-8")
    w = csv.writer(fo); w.writerow(LOG_COLS)
    for pi, part in enumerate(parts):
        with open(os.path.join(SHARDS, part), encoding="utf-8") as f:
            for p in csv.reader(f):
                r = dict(zip(LOG_COLS, p))
                u, v = int(r["user_id"]), r["video_id"]
                d, hm, tm = int(r["date"]), int(r["hourmin"]), int(r["time_ms"])
                dv, pt, clk = int(r["duration_ms"]), int(r["play_time_ms"]), int(r["is_click"])
                assert u in uids, f"断言2 失败：日志里的 user_id {u} 不在特征表"
                assert v in vdur, f"断言3 失败：video_id {v} 不在真实视频库"
                assert dv == vdur[v], f"断言3 失败：{v} 时长 {dv} != 真实 {vdur[v]}"
                assert 20220502 <= d <= 20220508, f"断言4 失败：date {d} 越界"
                dt = datetime.datetime.fromtimestamp(tm / 1000, TZ)
                assert int(dt.strftime("%Y%m%d")) == d and hourmin_of(dt) == hm, \
                    f"断言4 失败：time_ms 与 date/hourmin 不自洽 {d} {hm} {tm}"
                if not clk:
                    assert all(int(r[k]) == 0 for k in INTERACT), "断言5 失败：未点击行有互动"
                    assert int(r["comment_stay_time"]) == 0, "断言5 失败：未点击行有评论停留"
                assert pt <= dv * 8, f"断言5 失败：play_time_ms {pt} > 时长×8"
                assert r["is_rand"] == "0", "断言7 失败：is_rand 非 0"
                assert r["profile_stay_time"] == "0"
                w.writerow(p)
                n_row += 1; n_clk += clk
                per_expose[u] += 1; per_click[u] += clk
                dates[d] += 1; tabs[r["tab"]] += 1
                cs_nonzero += int(r["comment_stay_time"]) > 0
        if (pi + 1) % 10 == 0: print(f"    {pi+1}/{len(parts)} 片  {n_row:,} 行", flush=True)
    fo.close()
    print(f"  {n_row:,} 行 × 19 列，断言 3/4/5/7 通过")

    print("[4/5] 断言 8：每人 ≥3 次交互 且 ≥3 次点击 ...")
    assert set(per_expose) == uids, \
        f"断言8 失败：{len(uids - set(per_expose))} 个用户没有任何日志"
    bad_e = [u for u, n in per_expose.items() if n < MIN_CLICKS]
    bad_c = [u for u, n in per_click.items() if n < MIN_CLICKS]
    assert not bad_e, f"断言8 失败：{len(bad_e)} 人曝光 <{MIN_CLICKS}，例 {bad_e[:5]}"
    assert not bad_c, f"断言8 失败：{len(bad_c)} 人点击 <{MIN_CLICKS}，例 {bad_c[:5]}"
    pe = sorted(per_expose.values()); pc = sorted(per_click.values())
    print(f"  每人曝光 min/中位/max = {pe[0]}/{pe[len(pe)//2]}/{pe[-1]}")
    print(f"  每人点击 min/中位/max = {pc[0]}/{pc[len(pc)//2]}/{pc[-1]}   （下限 {MIN_CLICKS} ✓）")

    print("[5/5] 合并交互记录（内部留痕，含 round_no）+ 统计 ...")
    nrec, rounds = 0, collections.Counter()
    with open(f_rec, "w", encoding="utf-8") as fr:
        rpat = re.compile(rf"^records{re.escape(plan_tag)}_\d{{3}}\.jsonl$")
        for part in sorted(x for x in os.listdir(SHARDS) if rpat.match(x)):
            for line in open(os.path.join(SHARDS, part), encoding="utf-8"):
                fr.write(line); nrec += 1
                rounds[json.loads(line)["round_no"]] += 1
    summ = {"n_users": len(uids), "n_log_rows": n_row, "n_click": n_clk,
            "ctr": round(n_clk / max(n_row, 1), 5),
            "expose_per_user": {"min": pe[0], "median": pe[len(pe)//2], "max": pe[-1]},
            "click_per_user": {"min": pc[0], "median": pc[len(pc)//2], "max": pc[-1]},
            "n_records": nrec, "rounds": dict(sorted(rounds.items())),
            "date_dist": dict(sorted(dates.items())),
            "tab_dist": dict(tabs.most_common()),
            "comment_stay_nonzero_rate_in_click": round(cs_nonzero / max(n_clk, 1), 5),
            "assertions_passed": 8}
    json.dump(summ, open(os.path.join(OUT_GEN, f"merge_summary{plan_tag}.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"  交互记录 {nrec:,} 条，轮次分布 {dict(sorted(rounds.items()))}")
    print(f"  整体 CTR {summ['ctr']:.4f}；点击行 comment_stay 非零率 "
          f"{summ['comment_stay_nonzero_rate_in_click']:.4f}（真实 0.1320）")
    print(f"  tab 分布 {list(tabs.most_common(5))}（真实 1:81.3% 4:8.4% 0:5.8%）")
    print(f"\n✅ 8 条硬断言全部通过")
    print(f"-> {f_out}\n-> {f_user}\n-> {f_rec}")

if __name__ == "__main__":
    main()
