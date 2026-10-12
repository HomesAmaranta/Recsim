"""Step 0（新用户线）：区间桶反推 + 自检 + 真实边际分布基准。

不依赖 LLM，也不依赖 GPU。产出两份 JSON：

  data/gen/range_maps.json     四个 *_range 字段的桶边界。50w 用户表里这四列必须由
                               数值【现算】而不是从 donor 抄——内部一致性才免费成立。
  data/gen/real_marginals.json 真实用户特征表的边际分布，供 eval/augment_check.py 做基准，
                               也供 02_users.py 按真实占比抽样。

自检：用反推出的边界重算 24,292 行的四个 *_range，必须 100% 复原原表，否则边界猜错，
      脚本直接 assert 失败——这类「看起来对」的映射错误在 50w 行里是查不出来的。

A-README §4.2 的「脏」特征在这里一并记录（漏一个就一眼假）：
  BOM 表头 / is_lowactive_period 恒 0 / is_live_streamer 用 -124 表缺失(无 0)
  / onehot_feat4,12..17 是 "1.0" 这种浮点字符串 / feat12..17 各 605 行空值
"""
import os, sys, json, csv, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from recsim import config as C

GEN_DATA = os.path.join(C.DATA, "gen")

# (数值字段, 区间字段) —— 桶边界由 derive() 从真实表反推，不硬编码
PAIRS = [("follow_user_num", "follow_user_num_range"),
         ("fans_user_num", "fans_user_num_range"),
         ("friend_user_num", "friend_user_num_range"),
         ("register_days", "register_days_range")]

ONEHOT = [f"onehot_feat{i}" for i in range(18)]

def derive(rows, num, rng):
    """反推桶边界：每个区间标签取其数值的 [min, max]，按 min 排序。

    返回 [(lo, hi, label), ...]。真实表实测四组桶都是无重叠、无空隙的，
    所以 hi 只用来自检；查表时按 lo 升序取最后一个 lo <= v 的桶。
    """
    d = collections.defaultdict(lambda: [10**18, -1, 0])
    for r in rows:
        try: v = int(float(r[num]))
        except (ValueError, TypeError): continue
        b = d[r[rng]]; b[0] = min(b[0], v); b[1] = max(b[1], v); b[2] += 1
    buckets = sorted(([lo, hi, lab] for lab, (lo, hi, _) in d.items()), key=lambda x: x[0])
    for i in range(len(buckets) - 1):                 # 无重叠
        assert buckets[i][1] < buckets[i + 1][0], f"{rng} 桶重叠: {buckets[i]} {buckets[i+1]}"
    return buckets

def lookup(buckets, v):
    """数值 -> 区间标签。最后一个 lo <= v 的桶；小于最小 lo 时归第一个桶。"""
    lab = buckets[0][2]
    for lo, _, l in buckets:
        if v >= lo: lab = l
        else: break
    return lab

def main():
    os.makedirs(GEN_DATA, exist_ok=True)
    # utf-8-sig 吃掉源表的 BOM。注意：我们【写出】的表不带 BOM，交付说明里要写明这是源表瑕疵。
    rows = list(csv.DictReader(open(C.USER_FEAT, encoding="utf-8-sig")))
    print(f"[1/3] 真实用户特征表 {len(rows):,} 行 × {len(rows[0])} 列")

    print("[2/3] 反推四个 *_range 的桶边界并自检 ...")
    maps = {}
    for num, rng in PAIRS:
        b = derive(rows, num, rng)
        ok = sum(1 for r in rows if lookup(b, int(float(r[num]))) == r[rng])
        print(f"  {rng:24s} {len(b)} 桶  复原 {ok:,}/{len(rows):,}")
        assert ok == len(rows), f"{rng} 复原失败 {ok}/{len(rows)}，桶边界猜错了"
        maps[rng] = {"num_field": num, "buckets": b}
    json.dump(maps, open(os.path.join(GEN_DATA, "range_maps.json"), "w"),
              ensure_ascii=False, indent=1)

    print("[3/3] 统计真实边际分布 ...")
    mar = {"n_users": len(rows)}
    # 类别字段：全枚举 + 计数
    for c in ("user_active_degree", "is_lowactive_period", "is_live_streamer",
              "is_video_author") + tuple(f"{p[1]}" for p in PAIRS):
        mar[c] = dict(collections.Counter(r[c] for r in rows).most_common())
    # onehot：保留【字符串原样】的计数，这样 "1.0" 与 "1"、空值 "" 都能原样复现
    mar["onehot"] = {c: dict(collections.Counter(r[c] for r in rows).most_common())
                     for c in ONEHOT}
    # 联合分布：is_live_streamer × is_video_author（两者相关，必须联合抽）
    mar["streamer_x_author"] = dict(collections.Counter(
        f'{r["is_live_streamer"]}|{r["is_video_author"]}' for r in rows).most_common())
    # 社交三元组按活跃度分层的分位数，给 02_users.py 抽样用
    soc = collections.defaultdict(list)
    for r in rows:
        soc[r["user_active_degree"]].append(
            tuple(int(float(r[n])) for n, _ in PAIRS))
    mar["social_by_active"] = {k: {"n": len(v),
                                   "p": [sorted(x[i] for x in v)[len(v) * q // 100]
                                         for i in range(4) for q in (10, 50, 90)]}
                               for k, v in soc.items()}
    # 「脏」特征留痕，交付说明直接引这里
    mar["quirks"] = {
        "header_bom": True,
        "is_lowactive_period_all_zero": len(set(r["is_lowactive_period"] for r in rows)) == 1,
        "is_live_streamer_values": sorted(set(r["is_live_streamer"] for r in rows)),
        "float_string_cols": [c for c in ONEHOT
                              if any("." in r[c] for r in rows[:2000])],
        "empty_counts": {c: sum(1 for r in rows if r[c] == "")
                         for c in ONEHOT if any(r[c] == "" for r in rows)},
    }
    json.dump(mar, open(os.path.join(GEN_DATA, "real_marginals.json"), "w"),
              ensure_ascii=False, indent=1)
    print("  脏特征:", json.dumps(mar["quirks"], ensure_ascii=False))
    print(f"-> {GEN_DATA}/range_maps.json\n-> {GEN_DATA}/real_marginals.json")

if __name__ == "__main__":
    main()
