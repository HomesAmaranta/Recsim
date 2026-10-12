"""多角度 prompt 构造 —— 每个角度一个可开关 block，对应概设 2.2/2.4.1。

开关在 config.ANGLES，便于做消融实验（逐个关掉看准确率掉多少），
这也正是概设里各技术点有效性的证据来源。
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from recsim import config as C

SYSTEM = ("你在扮演一个真实的短视频平台用户，依据下面给出的该用户画像与记忆做选择。"
          "你的任务不是评价内容好坏，而是尽可能还原【这个特定用户】会怎么选。")

def _blk_long_term(st):
    p = st.get("profile") or {}
    lines = [f"长期稳定偏好：{p.get('long_term_preference','(无)')}"]
    if p.get("duration_preference"):
        lines.append(f"时长偏好：{p['duration_preference']}")
    if p.get("aversion"):
        lines.append(f"明显回避：{p['aversion']}")
    if p.get("decision_style"):
        lines.append(f"决策风格：{p['decision_style']}")
    return "【角度1 长期偏好】\n" + "\n".join(lines)

def _blk_short_term(st):
    p = st.get("profile") or {}
    sm = st.get("short_memory") or {}
    return ("【角度2 短期兴趣】\n"
            f"近期兴趣：{p.get('short_term_interest','(无)')}\n"
            f"近期兴趣类别（切片内点击分布）：{sm.get('近期兴趣类别')}")

def _blk_role(st):
    tags = st.get("role_tags") or []
    return "【角度3 角色特征】\n该用户在推荐场景中的角色标签：" + "、".join(tags) if tags else ""

def _blk_long_mem(st):
    lm = st.get("long_memory") or {}
    keep = ["长期点击率", "长看率", "偏好类别Top6(点击占比)", "类别偏好提升Top6",
            "明显回避类别", "各时长档点击率"]
    d = {k: lm[k] for k in keep if k in lm}
    return "【角度4 长期记忆（稳定兴趣与行为特征）】\n" + json.dumps(d, ensure_ascii=False)

def _blk_short_mem(st):
    sm = st.get("short_memory") or {}
    def fmt(items):
        return "\n".join(
            f"  - [{i['cat']}] {i['cap']} ({i['dur']}, 观看{i['play_s']}s"
            + (", 长看" if i.get("long_view") else "") + (", 点赞" if i.get("like") else "") + ")"
            for i in items) or "  (无)"
    return ("【角度5 短期记忆（近期具体条目）】\n"
            "最近选择观看的内容：\n" + fmt(sm.get("近期选择条目", [])) +
            "\n最近跳过未看的内容：\n" + fmt(sm.get("近期跳过条目", [])))

def _blk_group(st, groups):
    g = groups.get(str(st.get("group_id")), {})
    if not g:
        return ""
    return ("【角度6 群体特征（所属用户群体的共同倾向，仅作补充参考，个体差异优先）】\n"
            f"所属群体 {st.get('group_id')}（{g.get('size')} 人）近期关注类别：{g.get('群体近期关注类别Top5')}\n"
            f"群体平均点击率：{g.get('群体平均点击率')}")

def _fmt_dur(ms):
    return f"{ms/1000:.1f} 秒（{_dur_label(ms)} 档）" if ms else "未知"

def _dur_label(ms):
    ms = int(ms or 0)
    if ms <= 0: return "未知"
    for lo, hi, nm in ((0,10000,"0-10s"),(10000,20000,"10-20s"),(20000,40000,"20-40s"),
                       (40000,80000,"40-80s"),(80000,10**12,"80s+")):
        if lo <= ms < hi: return nm
    return "80s+"

def _pct(x):
    return f"{float(x)*100:.2f}%"

def _blk_candidates(cands, with_pop=False, author_hist=None, aff=None):
    """尽可能完整地给出候选视频信息（不含 server_width/server_height）。"""
    out = []
    for tag, m in cands:
        cats = " / ".join(m.get(k, "UNKNOWN") for k in ("cat1", "cat2", "cat3", "cat4"))
        cap = (m.get("caption") or "").strip() or "(无标题)"
        L = [f"候选 {tag}",
             f"  类别（四级）：{cats}",
             f"  标题：{cap}",
             f"  时长：{_fmt_dur(m.get('duration_ms', 0))}"]
        au = str(m.get("author_id", "UNK"))
        h = (author_hist or {}).get(au)
        L.append(f"  作者：#{au}" + (f"（你历史上被推过该作者 {h[0]} 次、点击 {h[1]} 次）"
                                    if h else "（你历史上没接触过该作者）"))
        L.append(f"  上传：{m.get('upload_dt','UNK')}，方式 {m.get('upload_type','UNK')}，"
                 f"视频类型 {m.get('video_type','UNK')}，配乐类型 {m.get('music_type','UNK')}，"
                 f"内容标签 {m.get('tag') or '无'}")
        if with_pop:
            L.append(f"  平台热度：累计展示 {float(m.get('show_cnt',0)):.0f}，"
                     f"触达用户 {float(m.get('show_user_num',0)):.0f}")
            L.append(f"  平台播放表现：完播率 {_pct(m.get('complete_rate',0))}，"
                     f"有效播放率 {_pct(m.get('valid_rate',0))}，长播放率 {_pct(m.get('long_rate',0))}，"
                     f"短播放跳出率 {_pct(m.get('short_rate',0))}，平均播放进度 {float(m.get('play_progress',0)):.2f}")
            L.append(f"  平台互动率：点赞 {_pct(m.get('like_rate',0))}，评论 {_pct(m.get('comment_rate',0))}，"
                     f"关注 {_pct(m.get('follow_rate',0))}，分享 {_pct(m.get('share_rate',0))}，"
                     f"收藏 {_pct(m.get('collect_rate',0))}，负反馈 {_pct(m.get('neg_rate',0))}")
        af = (aff or {}).get(tag)
        if af:
            L.append("  与你历史行为的交叉统计（来自该用户训练期日志）：")
            L += [f"    - {x}" for x in af]
            L.append("  （上面标【】的是强信号：某类别被推过很多次却一次没点，"
                     "说明该用户明确不看这类；标「弱证据」的仅供微调，不要当决定性依据）")
        out.append("\n".join(L))
    return "【角度7 内容语义（候选视频完整信息）】\n" + "\n".join(out)

def build_decide_prompt(st, groups, cands, angles=None, author_hist=None, aff=None):
    """cands: [("A", meta_a), ("B", meta_b)]，位置已随机化。返回 messages。"""
    A = angles if angles is not None else C.ANGLES
    blocks = []
    if A.get("long_term_preference"): blocks.append(_blk_long_term(st))
    if A.get("short_term_interest"):  blocks.append(_blk_short_term(st))
    if A.get("role_tags"):            blocks.append(_blk_role(st))
    if A.get("long_term_memory"):     blocks.append(_blk_long_mem(st))
    if A.get("short_term_memory"):    blocks.append(_blk_short_mem(st))
    if A.get("group_feature"):        blocks.append(_blk_group(st, groups))
    body = "\n\n".join(b for b in blocks if b)

    tail = ""
    if A.get("item_semantic"):
        tail = "\n\n" + _blk_candidates(cands, with_pop=A.get("item_popularity", False),
                                 author_hist=author_hist,
                                 aff=(aff if A.get("user_item_affinity") else None))

    head = ("\n\n【任务】\n以上两个视频中，恰好有一个是该用户真实观看过的，另一个是从内容库随机取出的。"
            "请依据该用户的画像、记忆与群体倾向，判断哪一个更可能是他真实选择的。\n"
            "注意：A、B 的先后顺序是随机打乱的，不携带任何信息，不要因为某个选项排在前面就倾向它。"
            "请先各自独立评估两个选项与该用户的匹配程度，再比较。\n")
    if A.get("cot"):
        task = head + ('严格按以下格式输出，不要多余内容：\nA评估：<该选项与用户的匹配点/不匹配点，一句>\n'
                       'B评估：<同上，一句>\n答案：<A 或 B>')
    else:
        task = head + "只输出一个字母 A 或 B，不要输出任何其他内容。"
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": body + tail + task}]

def parse_choice(text):
    """抠出最终选择。有"答案："时只认它后面的字母，否则取最后出现的 A/B。"""
    if not text:
        return None
    t = text.strip().upper()
    for marker in ("答案：", "答案:", "ANSWER:", "答案"):
        if marker in t:
            seg = t.split(marker)[-1]
            for ch in seg:
                if ch in ("A", "B"):
                    return ch
    if len(t) <= 4:                      # 纯字母模式
        for ch in t:
            if ch in ("A", "B"):
                return ch
    for ch in reversed(t):               # 兜底：取最后一个 A/B
        if ch in ("A", "B"):
            return ch
    return None
