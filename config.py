"""全局配置。所有路径与超参集中在此，脚本不硬编码。"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_SRC = os.path.join(os.path.dirname(ROOT), "kuairand-10w")
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")

# ---- 源数据表 ----
LOG        = os.path.join(DATA_SRC, "log_standard_kuairand_10w.csv")
USER_FEAT  = os.path.join(DATA_SRC, "user_features_27k.csv")
VIDEO_BASIC= os.path.join(DATA_SRC, "video_features_basic_27k.csv")
VIDEO_STAT = os.path.join(DATA_SRC, "video_features_statistic_27k.csv")
CAPTIONS   = os.path.join(DATA_SRC, "kuairand_video_captions.csv")
CATEGORIES = os.path.join(DATA_SRC, "kuairand_video_categories.csv")

# ---- 时间切分（数据全域 20220408-20220508，共 31 天）----
# 训练期：建画像 / 记忆 / 群体，模拟器只能看这一段
# 测试期：正样本来源，模拟器不可见
TRAIN_START = 20220408
TRAIN_END   = 20220501
TEST_START  = 20220502
TEST_END    = 20220508
# 短期兴趣 / 短期记忆的时间切片长度（训练期末尾 N 天）
SHORT_TERM_DAYS = 3

# ---- 规模 ----
N_USERS       = 200      # 模拟用户数
N_SAMPLES     = 20000    # 二选一日志总条数（可用 generate.py --n 覆盖）
# 每用户配额不再固定：generate.py 用【水位法】按各人测试期可用正样本量分配
#   取最小的 q 使 sum_u min(可用_u, q) >= N_SAMPLES，再按 user_id 升序微调到恰好相等。
# 实测 200 人测试期(去重+有标题)可用合计 38,118 条 -> q=119，配额区间 25~119，均值 100。
# PER_USER 保留给 01_select_users 的入选门槛（测试期点击数须 >= 它）与冒烟模式使用，
# 不要跟着 N_SAMPLES 一起放大——门槛提高会把分层抽样挤向头部活跃用户，基线不再可比。
PER_USER      = 25
CTR_STRATA    = 5        # 按训练期 CTR 分层抽样，每层 N_USERS/5 人
N_CLUSTERS    = 8        # 群体特征：用户聚类簇数

# ---- 负采样协议（写死，生成与评测共用）----
# 决策：全库随机采 —— 从 102,603 视频全库均匀随机，仅排除该用户已交互过的视频。
# 注意：这是「易负样本」协议，准确率会显著高于曝光内困难负样本。
#       验收时必须与该协议一并说明，不可混用。
NEG_PROTOCOL   = "global_random"
NEG_EXCLUDE_SEEN = True   # 排除该用户在训练期+测试期出现过的 video_id
# 候选池过滤：正负样本都只从「有标题」的视频里取（全库 93,489 / 102,603 = 91.1%）。
# 无标题视频占全库 8.9%、占真实点击 3.9%，本身构成一个与用户偏好无关的捷径信号；
# 且正样本无标题时模型拿不到任何内容信息（实测该子集 LLM 0.528 ≈ 抛硬币）。
REQUIRE_CAPTION = True

# ---- LLM ----
LLM_ENDPOINT = "http://localhost:30000/v1/chat/completions"
LLM_API_KEY  = ""          # 本地 SGLang 不需要；走外部网关时填（或用环境变量覆盖）
# 环境变量覆盖，便于一条命令换模型/换供应商做对照实验，不改代码：
#   RECSIM_LLM_ENDPOINT / RECSIM_LLM_MODEL / RECSIM_LLM_KEY / RECSIM_LLM_CONC
LLM_ENDPOINT = os.environ.get("RECSIM_LLM_ENDPOINT", LLM_ENDPOINT)
LLM_MODEL    = "Qwen3-30B-A3B-Instruct-2507"
LLM_CONCURRENCY = 32
DECIDE_CONCURRENCY = 64   # 二选一只出 1 个 token，瓶颈在 prefill，可以开更大并发
LLM_TIMEOUT   = 300
PROFILE_MAX_TOKENS = 600
DECIDE_MAX_TOKENS  = 24     # cot=False 只输出一个字母；开 cot 需调到 200
PROFILE_TEMPERATURE = 0.3
DECIDE_TEMPERATURE  = 0.0

# ---- 多角度 prompt 开关（对应概设 11 个角度，便于消融实验）----
ANGLES = {
    "long_term_preference": True,   # 角度1 长期偏好
    "short_term_interest":  True,   # 角度2 短期兴趣（时间切片）
    "role_tags":            True,   # 角度3 角色特征
    "long_term_memory":     True,   # 角度4 长期记忆（稳定行为特征）
    "short_term_memory":    True,   # 角度5 短期记忆（近期具体条目）
    "group_feature":        True,   # 角度6 群体特征
    "item_semantic":        True,   # 角度7 内容语义
    "user_item_affinity": False,  # 角度7c 用户x候选交叉量（统计路特征的自然语言注入）
                                  # 实测无收益：同 100 对上 关 0.73 / 注入 0.68-0.69。原因见 README §6.5
                                  # 开关保留，作为「统计特征语义化注入」这条路的负结果证据
    "item_popularity":      True,   # 角度7b 候选平台热度/表现/互动率（易负样本下含捷径，可单列消融）
    "cot":                  False,  # 实测开 CoT 掉 7.5pt（模型转去评内容好坏而非用户匹配），关闭
    # 角度 8/9/10 为机制类，本版留接口未启用
    "self_calibration":     False,  # 角度8 自我校准（模拟 vs 真实反调）
    "group_conflict":       False,  # 角度9 群体→个体冲突调节
    "state_update_loop":    False,  # 角度10 状态更新闭环（多轮推演时启用）
}
# 角度11 序列降噪+时间切片：在 prep 阶段实现，非 prompt 开关
DENOISE_MIN_PLAY_MS = 1000   # 播放 <1s 且无任何正反馈的曝光视为无效，剔除
DENOISE_DEDUP = True         # 同一视频重复曝光只保留首次

SEED = 20260809

# ---- 中间产物 ----
F_USER_STAT   = os.path.join(DATA, "user_stat.tsv")
F_USERS       = os.path.join(DATA, "users_200.txt")
F_USER_LOGS   = os.path.join(DATA, "user_logs.tsv")
F_VIDEO_META  = os.path.join(DATA, "video_meta.tsv")
F_PROFILES    = os.path.join(DATA, "profiles.json")
F_GROUPS      = os.path.join(DATA, "groups.json")
F_SIM_LOG     = os.path.join(OUT, "sim_log.csv")       # 按 README §4.1 字段结构
F_SIM_TRACE   = os.path.join(OUT, "sim_trace.jsonl")   # 逐条决策明细（含 prompt/响应）
F_REPORT      = os.path.join(OUT, "report.md")

LLM_MODEL       = os.environ.get("RECSIM_LLM_MODEL", LLM_MODEL)
LLM_API_KEY     = os.environ.get("RECSIM_LLM_KEY", LLM_API_KEY)
DECIDE_CONCURRENCY = int(os.environ.get("RECSIM_LLM_CONC", DECIDE_CONCURRENCY))
