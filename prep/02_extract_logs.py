"""Step 2: 抽取选定用户的完整日志 + 全库视频元数据。

日志按 config 的降噪规则（角度11）预处理：
  - 剔除 播放<DENOISE_MIN_PLAY_MS 且 无任何正反馈 的无效曝光
  - 同一用户对同一视频的重复曝光只保留首次
降噪只作用于「训练期」（用于建画像）；测试期保持原样，因为它是评测真值。
"""
import os, sys, csv, math, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from recsim import config as C

AWK_LOG = r'''
BEGIN { while ((getline u < UF) > 0) { split(u, a, "\t"); keep[a[1]] = 1 } }
NR==1 { next }
{
  split($0, f, ",")
  if (!(f[1] in keep)) next
  # user_id video_id date hourmin time_ms is_click is_like is_follow is_comment
  # is_forward is_hate long_view play_time_ms duration_ms is_profile_enter tab
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
    f[1],f[2],f[3],f[4],f[5],f[6],f[7],f[8],f[9],f[10],f[11],f[12],f[13],f[14],f[17],f[19]
}
'''

META_COLS = ["video_id","cat1","cat2","caption","duration_ms","show_cnt",      # 0-5 前6列固定不动
             "cat3","cat4","author_id","video_type","upload_dt","upload_type",   # 6-11
             "music_type","tag","show_user_num","play_progress",                 # 12-15
             "complete_rate","valid_rate","long_rate","short_rate",              # 16-19
             "like_rate","comment_rate","follow_rate","share_rate",              # 20-23
             "collect_rate","neg_rate"]                                          # 24-25

def build_video_meta():
    """全库 102,603 视频的完整元数据。不含 server_width/server_height。
    前 6 列顺序固定（video_id/cat1/cat2/caption/duration_ms/show_cnt），下游按位取。"""
    cap = {}
    for x in csv.DictReader(open(C.CAPTIONS, encoding="utf-8-sig")):
        t = (x["caption"] or "").replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()
        cap[x["final_video_id"]] = t
    cat = {}
    for x in csv.DictReader(open(C.CATEGORIES, encoding="utf-8-sig")):
        cat[x["final_video_id"]] = tuple((x[f"{lv}_level_category_name"] or "UNKNOWN")
                                         for lv in ("first", "second", "third", "fourth"))
    stat = {}
    for x in csv.DictReader(open(C.VIDEO_STAT, encoding="utf-8-sig")):
        def g(k):
            try: return float(x[k])
            except Exception: return 0.0
        sc = g("show_cnt"); den = sc + 1.0
        stat[x["video_id"]] = {
            "show_cnt": sc, "show_user_num": g("show_user_num"),
            "play_progress": g("play_progress"),
            "complete_rate": g("complete_play_cnt") / den, "valid_rate": g("valid_play_cnt") / den,
            "long_rate": g("long_time_play_cnt") / den, "short_rate": g("short_time_play_cnt") / den,
            "like_rate": g("like_cnt") / den, "comment_rate": g("comment_cnt") / den,
            "follow_rate": g("follow_cnt") / den, "share_rate": g("share_all_cnt") / den,
            "collect_rate": g("collect_cnt") / den,
            "neg_rate": (g("reduce_similar_cnt") + g("report_cnt")) / den,
        }
    Z = {k: 0.0 for k in ("show_cnt","show_user_num","play_progress","complete_rate","valid_rate",
                          "long_rate","short_rate","like_rate","comment_rate","follow_rate",
                          "share_rate","collect_rate","neg_rate")}
    n = n_cap = 0
    with open(C.F_VIDEO_META, "w", encoding="utf-8") as fo:
        fo.write("\t".join(META_COLS) + "\n")
        for x in csv.DictReader(open(C.VIDEO_BASIC, encoding="utf-8-sig")):
            v = x["video_id"]
            try: d = int(float(x["video_duration"]))
            except Exception: d = 0
            c = cat.get(v, ("UNK",) * 4)
            t = stat.get(v, Z)
            cp = cap.get(v, "")
            if cp: n_cap += 1
            def iv(k):
                try: return str(int(float(x[k])))
                except Exception: return "UNK"
            fo.write("\t".join([
                v, c[0], c[1], cp, str(d), f"{t['show_cnt']:.1f}",
                c[2], c[3], iv("author_id"), (x.get("video_type") or "UNK"),
                (x.get("upload_dt") or "UNK"), (x.get("upload_type") or "UNK"),
                iv("music_type"), (x.get("tag") or "").replace("\t", " "),
                f"{t['show_user_num']:.1f}", f"{t['play_progress']:.4f}",
                f"{t['complete_rate']:.5f}", f"{t['valid_rate']:.5f}",
                f"{t['long_rate']:.5f}", f"{t['short_rate']:.5f}",
                f"{t['like_rate']:.6f}", f"{t['comment_rate']:.6f}",
                f"{t['follow_rate']:.6f}", f"{t['share_rate']:.6f}",
                f"{t['collect_rate']:.6f}", f"{t['neg_rate']:.6f}",
            ]) + "\n")
            n += 1
    print(f"  视频元数据 {n:,} 条（{len(META_COLS)} 列，有标题 {n_cap:,} = {n_cap/n:.1%}）-> {C.F_VIDEO_META}")

def denoise(rows):
    """角度11: 序列降噪。rows 为单用户按时间排序的训练期记录。"""
    seen, out, dropped_noise, dropped_dup = set(), [], 0, 0
    for r in rows:
        pos = any(int(r[k]) for k in ("is_click", "is_like", "is_follow",
                                      "is_comment", "is_forward", "long_view"))
        if int(r["play_time_ms"]) < C.DENOISE_MIN_PLAY_MS and not pos:
            dropped_noise += 1
            continue
        if C.DENOISE_DEDUP:
            if r["video_id"] in seen:
                dropped_dup += 1
                continue
            seen.add(r["video_id"])
        out.append(r)
    return out, dropped_noise, dropped_dup

COLS = ["user_id","video_id","date","hourmin","time_ms","is_click","is_like","is_follow",
        "is_comment","is_forward","is_hate","long_view","play_time_ms","duration_ms",
        "is_profile_enter","tab"]

def main():
    if not os.path.exists(C.F_VIDEO_META):
        print("[1/2] 构建全库视频元数据...")
        build_video_meta()
    else:
        print(f"[1/2] 复用 {C.F_VIDEO_META}")

    if not os.path.exists(C.F_USER_LOGS):
        print("[2/2] 抽取 200 用户日志 (约 2-3 分钟)...")
        with open(C.F_USER_LOGS, "w") as fo:
            subprocess.run(["awk", "-v", f"UF={C.F_USERS}", AWK_LOG, C.LOG],
                           stdout=fo, check=True)
    else:
        print(f"[2/2] 复用 {C.F_USER_LOGS}")

    # 统计 + 降噪效果
    import collections
    per = collections.defaultdict(list)
    for line in open(C.F_USER_LOGS):
        p = line.rstrip("\n").split("\t")
        per[p[0]].append(dict(zip(COLS, p)))
    tot_tr = tot_te = dn = dd = 0
    for u, rows in per.items():
        rows.sort(key=lambda r: int(r["time_ms"]))
        tr = [r for r in rows if C.TRAIN_START <= int(r["date"]) <= C.TRAIN_END]
        te = [r for r in rows if C.TEST_START <= int(r["date"]) <= C.TEST_END]
        kept, a, b = denoise(tr)
        tot_tr += len(kept); tot_te += len(te); dn += a; dd += b
    print(f"\n用户数 {len(per)}")
    print(f"训练期降噪后 {tot_tr:,} 条 (剔除无效曝光 {dn:,}, 去重复曝光 {dd:,})")
    print(f"测试期 {tot_te:,} 条 (保持原样，作为评测真值)")

if __name__ == "__main__":
    main()
