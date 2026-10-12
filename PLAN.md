# 交付执行计划：旧用户 2 万条日志（≥85%） + 新用户 50 万（文档框架）

> 本文是给**接手 agent 的可执行说明书**。所有数字都已在本机核过（核过的地方标了「实测」），
> 没核过的地方明确写了「需先核」。按 §7 的顺序执行，每个阶段都有冒烟检查点，不要跳。
>
> 前置：LLM 服务未起时**只能做 §2.1 的代码改动和 §4 的不依赖 LLM 的部分**，
> 判决与画像阶段必须等服务就绪（见 §6.1）。

---

## 0 目标与交付物

| | 任务一：旧用户 | 任务二：新用户 |
|---|---|---|
| 规模 | 现有 200 个真实用户，**2 万条**模拟互动日志（2 万轮二选一） | **50 万**模拟用户 + 其行为日志 |
| 硬指标 | 推荐算法模拟准确率 **≥85%**（当前 5,000 轮实测 0.8986） | 无准确率指标；只做**规模**与**真实性**（见 §5） |
| 方法 | 已跑通的 2AFC 链路（语义路 × 统计路融合），只放大规模 | 按概设文档的四件套：用户聚类 + 代表性智能体 + 轻量化模型 + 复用画像/群体/决策模板 |

**交付文件只有两张表**，严格对齐 `kuairand-10w/A-README.md`：

| 交付表 | 对齐 | 字段数 | 说明 |
|---|---|---|---|
| 模拟用户互动日志 | A-README **§4.1** `log_standard_kuairand_10w.csv` | **19** | 「==需按此表结构返回模拟用户互动日志==」 |
| 模拟用户特征表 | A-README **§4.2** `user_features_27k.csv` | **31** | 「==需按此表结构返回模拟用户特征标==」 |

**不交付视频侧任何表**——A-README §4.3 明写「==仅作参考，模拟不生成新视听内容，无需返回 4.3-4.6 的视频特征表==」。
新用户的日志里 `video_id` **一律引用真实视频库的 102,603 个 id**，一个新视频都不造。

概设文档 §2.5.2 要的四张表（用户状态表 / 用户群体状态表 / 交互记录表 / 模拟结果表）
**降级为内部中间产物**（`out/gen/interaction_records.jsonl` 等），用来证明多轮闭环成立，不作为交付。
这一点解决了之前「文档要四张表、甲方要两张表」的冲突：**两张表是交付口径，四张表是过程留痕**。

---

## 1 严格的字段口径（照抄，不要凭印象写）

### 1.1 §4.1 日志表 19 列（顺序不能变）

```
user_id,video_id,date,hourmin,time_ms,is_click,is_like,is_follow,is_comment,
is_forward,is_hate,long_view,play_time_ms,duration_ms,profile_stay_time,
comment_stay_time,is_profile_enter,is_rand,tab
```

现有 `sim/fuse.py: OUT_COLS` 与 `sim/generate.py: OUT_COLS` 已经就是这个顺序，**已核对一致，不要改**。

约定：
- `is_rand` 恒 0（**已核**：真实 36,901,927 行全为 0，A-README 也写「标准推荐日志中通常为 0」）。
- `date`/`hourmin`/`time_ms` 三者必须**互相自洽**（`time_ms` 换算回北京时间 = `date`+`hourmin`）。
  这是最容易被抽查出来的破绽，落盘前必须有断言。
- `profile_stay_time`：**已核**，99.99% 为 0（36,898,206/36,901,927），保留硬编码 0。
- `comment_stay_time`：**已核，不是全 0** —— 7.31% 非零（2,698,927 行），
  **点击行里 13.20% 非零**，非零值均值 9,970 ms、最大 347,816，量级集中在 10³–10⁴ ms。
  ⚠️ 现有 `generate.py`/`fuse.py` 把它硬编码成 0，**新用户线必须改**：
  点击行以 13.2% 概率从 donor 的非零经验分布重采样，未点击行置 0。
  （任务一的 2 万条沿用硬编码 0，因为要与 5,000 轮基线同口径；这个差异在交付说明里写明。）
- `tab`：**已核**，`1` 占 81.26%、`4` 8.39%、`0` 5.79%、`2` 3.03%、`6` 1.23%，尾部还有 10 个取值。
  现有产出硬编码 `tab=1`；新用户线按该分布（或 donor 的个人 tab 组成）抽。

### 1.2 §4.2 用户特征表 31 列（顺序不能变）

```
user_id,user_active_degree,is_lowactive_period,is_live_streamer,is_video_author,
follow_user_num,follow_user_num_range,fans_user_num,fans_user_num_range,
friend_user_num,friend_user_num_range,register_days,register_days_range,
onehot_feat0 ... onehot_feat17
```

**真实表的六个「脏」特征，必须原样复现，否则一眼假**（以下全部实测于 `user_features_27k.csv`，24,292 行）：

| 现象 | 实测 | 处理 |
|---|---|---|
| 表头带 BOM | 首列名是 `﻿user_id` | **我们写不带 BOM 的 UTF-8**，并在交付说明里写明这是源表瑕疵、我们没复制 |
| `is_lowactive_period` 恒 0 | 24,292/24,292 全 0 | 恒 0 |
| `is_live_streamer` 用 −124 表缺失 | `-124`: 18,594 / `1`: 5,698（**没有 0**） | 只能取 {−124, 1} |
| `is_video_author` | `1`: 20,126 / `0`: 4,166 | 与 `is_live_streamer` 联合抽（相关） |
| 部分 onehot 是浮点字符串 | `onehot_feat4`、`feat12..17` 是 `"1.0"` / `"0.0"` | 写出时保持 `.0` 后缀 |
| onehot 有空值 | **`feat4` 786 行**、`feat12..17` 各 605 行 | 整块复制 donor 即自动带上，**不要单列填 0** |

> `feat4` 的 786 个空值是跑 `00_range_maps.py` 时才发现的（原先只记了 `feat12..17` 的 605）。
> 以 `data/gen/real_marginals.json` 的 `quirks` 字段为准，不要凭这张表的记忆写代码。

`user_active_degree` 九个取值实测分布：
`full_active` 16,467 / `high_active` 5,126 / `middle_active` 1,850 / `2_14_day_new` 559 /
`low_active` 209 / `single_low_active` 54 / `30day_retention` 13 / `day_new` 10 / `UNKNOWN` 4。

区间字段的桶（实测枚举，`00_range_maps.py` 必须自检能 100% 复原真实表）：
- `follow_user_num_range`: `0` / `(0,10]` / `(10,50]` / `(50,100]` / `(100,150]` / `(150,250]` / `(250,500]` / `500+`
- `fans_user_num_range`: `0` / `[1,10)` / `[10,100)` / `[100,1k)` / `[1k,5k)` / `[5k,1w)` / `[1w,10w)` / `[10w,100w)` / `[100w,1000w)`
- `friend_user_num_range`: `0` / `[1,5)` / `[5,30)` / `[30,60)` / `[60,120)` / `[120,250)` / `250+`
- `register_days_range`: `8-14` / `15-30` / `31-60` / `61-90` / `91-180` / `181-365` / `366-730` / `730+`

---

## 2 任务一：旧用户 2 万条，打到 ≥85%

### 2.1 关键决策：**不换用户集，只放大每人的轮数**

两条路，选后者：

| | A：扩到 400 用户 × 50 轮 | B：**保持 200 用户，按可用量分配配额**（选这条） |
|---|---|---|
| 需重跑 | `01_select_users`（扫 36.9M 行）+ `02_extract_logs` + `03_build_profiles`（400 次 LLM） | **都不用重跑** |
| 基线可比 | 用户集变了，0.8986 不再可直接对比 | 同一批 200 人，5,000 轮 → 20,000 轮是纯规模放大 |
| 副作用 | `01` 的入选条件是「测试期点击 ≥ PER_USER」，PER_USER 从 25 提到 50 会把分层挤向头部活跃用户 | 无 |

### 2.2 配额怎么算（实测数字，可直接写死校验）

对现有 200 人，**测试期 + 有标题 + 按 `video_id` 去重**后的可用正样本：

```
合计 38,118 条    P0=25  P25=84  P50=162  P75=258  P90=393  max=744
```

用**水位法**（每人取 `min(可用, q)`，二分找最小的 q 使总量 ≥ 20,000）：

```
q = 119   →  合计 20,059   （可用 ≥119 的用户有 124 人）
削减 59 条即为 20,000：按 user_id 升序，对可用 ≥119 的前 59 个用户各减 1
```

> 注意：另一个口径（不按 `video_id` 去重）会算出 38,429 / 20,108 / 削减 108。
> **`generate.py` 现有逻辑是去重的**（`dedup` 集合），所以以 38,118 / q=119 / 削减 59 为准。
> 脚本里不要写死 q，要现算，然后**断言最终轮数恰为 20,000**。

平均每人 100 轮（vs 现在 25 轮），配额区间 25～119。

### 2.3 需要改的代码（三处，都很小）

**① `recsim/config.py`**
```python
N_SAMPLES = 20000      # 原 5000
# PER_USER 不再是「每人固定条数」，改为水位上限的兜底；实际配额由 generate.py 现算
```

**② `recsim/sim/generate.py`** —— `[2/4]` 那段，把「每人固定 `per_user` 条」改成水位配额：

- 先对所有 `users` 跑一遍，得到每人去重后的 `pos_cand` 列表（现有代码已经在构造 `pos_cand`，
  把它拆成两趟：第一趟只算长度，第二趟按配额取样）。
- 二分求 q 使 `sum(min(len_u, q)) >= N_SAMPLES`；再按 user_id 升序对 `len_u >= q` 的用户逐个 −1，
  直到 `sum(quota) == N_SAMPLES`。
- `take = rng.sample(pos_cand, quota_u)`，其余逻辑（负采样、A/B 随机化、`--dual` 互换）**一行都不用改**。
- 新增 CLI：`--n 20000`（覆盖 `C.N_SAMPLES`），保留 `--smoke a b` 不动。
- 打印必须包含：`q`、`sum(quota)`、配额的 min/median/max，方便一眼核。

**③ `recsim/sim/fuse.py`** —— 无需改动。`TRAIN_PAIRS_PER_USER = 60` **保持 60**
（与 5,000 轮基线同口径）。想看能不能再涨，另跑一次 `--tag tp120` 把它调到 120 做消融，
不要动主跑。

### 2.4 执行命令与时间账

```bash
# 0) 冒烟：3 用户 × 5 条，确认改动没把链路弄坏（1 分钟内）
python3 recsim/sim/generate.py --smoke 3 5 --dual

# 1) 主跑：20,000 轮 × A/B 双向 = 40,000 次 LLM
#    实测吞吐 19.8 req/s（SGLang 前缀缓存命中，互换 prompt 共享用户侧前缀）
#    预计 40000 / 19.8 ≈ 34 分钟
nohup python3 recsim/sim/generate.py --n 20000 --dual > recsim/out/gen_20k.log 2>&1 &

# 2) 融合 + 落盘（纯 numpy，20,000 对约 2-4 分钟）
python3 recsim/sim/fuse.py

# 3) 准确率报告
python3 recsim/eval/accuracy.py
```

产物：`out/sim_log.csv`（**40,001 行** = 表头 + 20,000×2，每轮一条 `is_click=1` 一条 `is_click=0`）、
`out/sim_trace.jsonl`（20,000 行）、`out/decisions.jsonl`（20,000 行）、`out/fuse_summary.json`。

### 2.5 验收阈值与失败处置

| 指标 | 5,000 轮实测 | 20,000 轮预期 | 红线 |
|---|---|---|---|
| 统计路（含语义增强特征） | 0.8986 | 0.89 ± 0.01 | **< 0.87 要停下诊断** |
| v2 门控堆叠融合 | 0.8980 | 与统计路持平（±0.3pt） | 低于统计路 1pt 以上说明融合器出错 |
| 语义路 A/B 去偏 | 0.7182 | 0.71 ± 0.01 | < 0.68 说明 prompt/服务有问题 |
| 位置偏置（融合路选 A 率） | 0.4986 | 0.49–0.51 | 偏离 0.5 超 0.03 要查 |

预期会**略降一点**，因为配额从 25 提到最多 119，纳入了每个用户测试期更靠尾部的点击
（按历史长度分箱实测：活跃度高的用户统计路 0.898–0.901，不算悬崖）。
只要 ≥0.85 就达标，≥0.87 就与既有结论一致。

若 < 0.87，按顺序查：
1. `fuse_summary.json` 的 `train_pair_acc`（训练期成对准确率）是否也掉了 → 掉了说明是配额引入的难样本，属真实现象，把 `TRAIN_PAIRS_PER_USER` 提到 120 再看。
2. 位置偏置是否失控 → 查 `--dual` 是否真的生效（`llm_score` 非 null 的比例应 ≈100%）。
3. `n_null`（LLM 没吐出 A/B 的条数）是否显著 > 0 → 服务过载，降 `RECSIM_LLM_CONC`。

**口径必须随报告一起写**：负采样 = 全库随机（`NEG_PROTOCOL="global_random"`，排除已见，
仅限有标题视频）。这是易负样本协议，准确率显著高于曝光内困难负样本，验收时不可混用口径。

---

## 3 任务二：50 万新用户 —— 按概设文档的框架

### 3.1 文档条款 → 实现映射（这一节直接可以进汇报）

概设文档只在三处讲「怎么扩到 50 万」，逐条落地：

| 文档原文 | 出处 | 实现 |
|---|---|---|
| 「结合**用户聚类**和轻量化模型降低大规模模拟的资源消耗」 | §1.1.2 | 复用已有 KMeans **k=8** 群（`data/groups.json`），新用户先落群，再在群内选 donor |
| 「采用用户聚类、**代表性智能体**和轻量化模型等方式提高大规模模拟效率」 | §1.2.1 | **8 群 × 200 = 1,600 个代表性智能体**，每个由 LLM 写一份画像，平均复用 ≈313 个新用户 |
| 「选取代表性用户使用大模型进行模拟，再根据实际需要采用**轻量化方式扩展用户规模**」 | §2.2 | 逐条曝光的决策走 **17 维成对 logistic + 每用户 CTR 校准**，LLM **不进单条循环** |
| 「对具有相似特征的用户**复用部分画像、群体信息和决策模板**」 | §2.2 | donor 混合（Dirichlet）+ 原型画像复用 + 群体统计共享 |
| 「每轮交互中**推荐环境提供候选内容**，用户智能体生成选择行为，系统记录交互结果并更新用户状态，更新后的用户状态继续参与下一轮交互」 | §2.2 / §2.4.1 | §3.5 的多轮闭环，`round_no` 落在内部交互记录表 |
| 「支持不低于 5 万级视听内容和 50 万级受众规模」 | §1.2.1 | 视听内容 **102,603 个真实视频（有标题 93,489）已超标**；受众 50 万由本任务提供 |

**为什么 LLM 不进单条循环，这是必须写进汇报的论证**：50M 次曝光决策 ÷ 19.8 req/s ≈ **29 天**，
物理上不可行。而我们已经用 5,000 轮实测证明**决策级 LLM 投票对准确率的增量为零**
（语义项权重 `b₁+b₂`=+0.070 vs 统计项 `a₁`=+0.957，McNemar p=0.76，见 `FUSION.md` §5 三个反证）。
所以「LLM 只做代表性智能体的画像、轻量化模型做逐条决策」不是妥协，是**有测量依据的最优分工**，
且正是文档 §2.2 点名的做法。

### 3.2 Step 0 —— 前置核查（不依赖 LLM，可以现在就做）

```bash
/home/fengyuebo/miniconda3/envs/py311/bin/python recsim/gen/00_range_maps.py
```
**已执行，全绿**：四组桶 **24,292/24,292 全部复原**，产物见 `data/gen/range_maps.json`
（桶边界）与 `data/gen/real_marginals.json`（边际分布 + `quirks` 脏特征清单）。
真实日志侧的核查结论已并入 §1.1（`comment_stay_time` 非全 0、`tab` 分布、`is_rand` 全 0）。
后续脚本一律**读这两份 JSON**，不要在代码里硬编码桶边界或分布。

### 3.3 Step A —— 代表性智能体（唯一用 LLM 的一步）

```bash
python3 recsim/gen/01_archetypes.py            # 先 --smoke 8 2 跑 16 个看输出质量
```
- 对 8 个群，各取 200 个真实用户作**原型种子**（群内按训练期活跃度分层，避免只取头部）。
- 复用 `prep/03_build_profiles.py` 的 prompt 与解析逻辑，产出 1,600 份画像
  （`long_term_preference` / `short_term_interest` / `role_tags` / `aversion` + 长短期记忆统计块）。
- 1,600 次调用 × `PROFILE_MAX_TOKENS=600`，并发 32，预计 **10–15 分钟**。
- 产出 `data/gen/archetypes.json`，每份带 `archetype_id` / `group_id` / `seed_user_id`。
- 这份画像文本同时喂给 `features.load_profile_sem()` 折出 `sem_cat_pref` 等三个语义特征——
  **语义增强在新用户这条线上是通过原型画像继承的**，不是每人重算。

### 3.4 Step B —— 50 万用户特征表（§4.2，31 字段）

```bash
python3 recsim/gen/02_users.py --n 500000
```

每个新用户的构造顺序（**先定群，再定 donor，再定字段**）：

1. `group_id` ~ 真实群体规模分布（8 群按真实占比抽）。
2. **行为 donor**：群内抽 2–3 个真实用户，Dirichlet 权重混合其类别偏好 / 时长偏好 / 互动率向量
   （温度抖动），得到「基因」。这个基因只用于 Step C 的行为生成。
3. **原型绑定**：在同群的 200 个原型里，按基因与原型种子的类别偏好余弦相似度抽一个（softmax），
   继承其画像文本。
4. **活跃度与 CTR 必须联合抽**：实测 `corr(log 活跃度, CTR) = −0.1157`，
   独立抽会造出「超高活跃 + 超高 CTR」这种真实数据里不存在的组合，是判别器最容易抓的破绽。
   做法：从真实 24,170 人的 (log 活跃度, CTR) 二维经验分布里整体抽一个点 + 局部抖动。
5. **特征表字段**（分块 bootstrap，块内保持联合分布，块间换 donor 以免整行照抄）：
   - `user_id` = `1000000 + i`（**≥1,000,000，与真实 id 空间完全不重叠**）
   - `user_active_degree`：按第 4 步抽到的活跃度水平，在真实条件分布里抽
   - `is_lowactive_period` = 0；`is_live_streamer` / `is_video_author` 联合抽（见 §1.2）
   - 社交块 (`follow_user_num`, `fans_user_num`, `friend_user_num`) **三元组整体**从同
     `user_active_degree` 层的 donor 复制，再乘 U(0.85, 1.15) 抖动取整
   - 四个 `*_range` **由数值现算**（用 `range_maps.json`），不从 donor 抄 → 内部一致性免费成立
   - `register_days` = donor 值 + 抖动，`register_days_range` 现算
   - `onehot_feat0..17` **整块 18 列从另一个 donor 原样复制**（加密特征之间强相关，
     逐列独立重采样会毁掉联合分布；空值和 `.0` 格式也靠整块复制自然带上）
6. 写 `out/gen/manifest.jsonl`：`user_id` / `group_id` / `archetype_id` / `behavior_donors` /
   `social_donor` / `onehot_donor` / 目标活跃度 / 目标 CTR。**这份 manifest 是可追溯性的全部依据，必须交。**

多样性护栏：24,170 个真实 donor → 50 万人 = **20.7 倍复用**，这是信息上限。
manifest 里必须统计**单个 donor 的最大复用次数**，超过 60 次就调整抽样（加抖动或限流）。

### 3.5 Step C —— 推荐环境 + 多轮闭环 + 决策（文档 §2.2 的闭环）

```bash
python3 recsim/gen/03_env_rollout.py --shard 0 --nshard 100     # 单片冒烟
bash recsim/gen/run_all_shards.sh                                # 32 进程并行
```

每轮 `round_no = 0..R-1`，严格按文档「推荐环境给候选 → 智能体决策 → 记录 → 更新状态 → 下一轮」：

1. **推荐环境**（可替换模块，接口留出来给后续导向性漏洞挖掘换算法）：
   给该用户出 `S` 个候选，`P(曝光 v) ∝ 热度(v)^γ · (1 + λ · 该用户对 v 类别的偏好)`
   - `γ` 由真实曝光的热度分布拟合（真实日志里视频曝光次数的幂律指数）
   - `λ` **按群拟合**：使该群合成用户「曝光落在其 Top 类别里的占比」等于真实测得值
   - 候选只从 102,603 个真实视频里取，排除该用户已曝光过的
2. **智能体决策**（轻量化模型）：用现有 17 维成对 logistic 打分，
   再做**每用户 CTR 校准** `p = σ(α_u · score + β_u)`：
   - `β_u` 使该用户实现的 CTR 等于 Step B 分配的目标 CTR
   - `α_u` 使 score→CTR 的斜率匹配该群实测曲线（防止「点击与打分无关」或「过度确定」）
3. **辅助字段**：点击行时从该用户 donor 的**同时长档经验池**重采样
   (`play_time_ms` 比例, `long_view`, `is_like`, `is_follow`, `is_comment`, `is_forward`,
   `is_profile_enter`)——沿用 `generate.py: behavior_model()` 的逻辑；未点击行播放时长取 U(0.02, 0.25)×时长。
4. **状态更新**（这一步是文档的核心要求，也是之前方案缺的）：本轮结束后
   - 更新短期兴趣：近 `SHORT_TERM_DAYS` 窗口内的类别计数、CTR、时长偏好
   - 重算该用户的统计侧特征（`features.user_state` 同一条代码路径）
   - 下一轮的推荐环境用**更新后**的类别偏好算 `λ` 项 → 曝光分布随行为漂移
   - 这样产出的是文档要的「**连续用户行为轨迹**」，不是一堆独立样本
5. **时间轴**：模拟期 `20220502–20220508`（7 天）。轮次映射到天/时段，
   `hourmin` 从 donor 的真实小时分布抽，`time_ms` 由 `date`+`hourmin` **精确换算**（落盘前断言）。
6. **内部留痕**（对齐文档 §2.5.2，不作为交付）：`out/gen/interaction_records.jsonl`
   每行 `{simulation_id, user_id, round_no, recommendation_info, behavior_info, state_info}`。

**关键：LLM 不出现在这个循环里的任何位置。** 语义信息通过原型画像折出的三个语义特征进入打分。

### 3.6 Step D —— 规模账与落盘

| 方案 | 每人曝光 | 总行数 | 体积（实测行宽 ~95 B） |
|---|---|---|---|
| **主交付** 50 万 × 5 轮 × 20 候选 | 100 | **5,000 万** | ≈ 4.8 GB |
| 高保真子集 5,000 × 1,520（真实人均） | 1,520 | 760 万 | ≈ 0.7 GB |

磁盘实测可用 **3.6 TB**，CPU **112 核**，内存 503 GB，容量上完全没问题。

- 分 **100 片**（每片 5,000 用户），**32 进程**并行（留核给别人）。
- 单片 50 万行的打分是 numpy 向量化矩阵乘，主要开销在 IO；预计整体 **1–2 小时**。
- `04_merge.py` 合并 + 校验 + 出统计；同时提供**分片保留**选项（5,000 万行单文件对下游不友好）。
- 最终交付：
  - `out/gen/sim_user_features_500k.csv`（500,001 行，31 列）
  - `out/gen/sim_log_500k.csv` 或 `sim_log_500k/part-*.csv`（5,000 万行，19 列）
  - `out/gen/manifest.jsonl`、`out/gen/interaction_records.jsonl`（内部）

落盘前的**硬断言**（任一失败就不许交）：
1. 两张表列名与顺序逐字符等于 §1.1 / §1.2。
2. 日志表 `user_id` 全部出现在特征表里；特征表 `user_id` 全部 ≥ 1,000,000 且唯一。
3. 日志表 `video_id` 全部 ∈ 真实 102,603 视频库；`duration_ms` 等于该视频的真实时长。
4. `date`+`hourmin` ↔ `time_ms` 自洽；`date` ∈ [20220502, 20220508]。
5. `is_click=0` 的行，所有互动字段（`is_like`…`is_profile_enter`）必须为 0；`play_time_ms ≤ duration_ms × 8`。
6. 四个 `*_range` 与对应数值字段一致（用 `range_maps.json` 复算校验）。
7. `is_rand` 全 0，`is_lowactive_period` 全 0，`is_live_streamer` ∈ {−124, 1}。

---

## 4 真实性验收（新用户唯一的质量指标）

```bash
python3 recsim/eval/augment_check.py
```

| 检验 | 做法 | 通过线 |
|---|---|---|
| **① 判别器** | 对每个用户算 ~24 个聚合特征（曝光数、CTR、长看率、类别熵、时段熵、互动率…），用手写 logistic 分真实 24,170 人 vs 合成同样多人 | **AUC ∈ [0.45, 0.60]**；>0.70 说明有明显破绽，看权重最大的特征去修 |
| **② 边际 + 联合分布** | ≥20 个统计量做 KS/χ²；重点核 (log 活跃度 × CTR) 二维直方图，以及 `corr(log 活跃度, CTR)` 是否落在真实值 **−0.1157** 附近 | 主要统计量的相对偏差 < 10%，相关系数偏差 < 0.05 |
| **③ 迁移测试（最强证据）** | **只用合成日志**拟合 17 维成对 logistic，拿去评测**200 个真实用户**的测试期 2AFC | **≥0.85**（真实日志自拟合 0.8986，掉 5pt 以内即认为合成数据承载了真实偏好结构） |
| **④ 多样性护栏** | manifest 里统计 donor 最大复用次数、合成用户之间的最近邻距离分布 | 单 donor 复用 ≤60；最近邻距离分布不塌缩成尖峰 |

### 一条不能破的规矩

**绝不在合成用户身上报「推荐算法模拟准确率」。**
合成用户的行为是我们自己的模型生成的，再拿同一个模型去预测，准确率会趋近 100%，
这个数字是自我实现的，一旦写进汇报就是硬伤。
**准确率只在真实 held-out 用户上报（任务一的 2 万条，0.89），规模由合成用户提供（50 万）。**
文档 §1.2.1 的措辞是「**支持**不低于…50 万级受众规模，推荐算法模拟准确率不低于 75%」，
「支持规模」是能力声明，准确率按 §2.3.1「在具备真实用户行为的情况下…比较」——
所以这两件事分开报，恰好合文档口径，不是我们打折扣。

---

## 5 新增文件清单

```
recsim/gen/
  00_range_maps.py      区间桶反推 + 自检 + 真实边际分布基准
  01_archetypes.py      8×200 代表性智能体（唯一用 LLM）
  02_users.py           50w 用户特征表（§4.2 31 字段）+ manifest
  03_env_rollout.py     推荐环境 + 多轮闭环 + 决策 + 辅助字段（分片）
  04_merge.py           合并 / 断言 / 统计
  run_all_shards.sh     32 进程并行驱动
recsim/eval/
  augment_check.py      四项真实性检验
recsim/data/gen/        range_maps.json  real_marginals.json  archetypes.json
recsim/out/gen/         sim_user_features_500k.csv  sim_log_500k*  manifest.jsonl
                        interaction_records.jsonl  augment_check.json
```

依赖约束沿用现状：**只用 numpy + 标准库**，不引 sklearn/scipy/torch/pandas。

---

## 6 执行顺序（带检查点，不要跳）

### 6.1 前置：确认 LLM 服务

```bash
gqueue                                                  # 有没有在跑的 job
curl -s -m 5 http://localhost:30000/v1/models           # 或 export RECSIM_LLM_ENDPOINT=...
```
未起时这样起（**必须走 gflow，单卡**）：
```bash
gbatch -g 1 -n sglang-qwen3 bash /home/fengyuebo/projects/start_sglang_qwen3_30b.sh
gjob log @        # 等到 "The server is fired up and ready to roll"
```
端点不是 localhost 时：`export RECSIM_LLM_ENDPOINT=http://<ip>:30000/v1/chat/completions`。

### 6.2 顺序与时间

| # | 步骤 | 依赖 LLM | 预计 | 检查点 |
|---|---|---|---|---|
| 1 | §2.3 改 config / generate.py | 否 | 30 min | `--smoke 3 5 --dual` 通过 |
| 2 | §2.4 主跑 20,000 轮 | **是** | ~34 min | `gen_20k.log` 里 `n_null` ≈ 0 |
| 3 | `fuse.py` + `accuracy.py` | 否 | 5 min | **统计路 ≥0.87，融合 ≥0.85** ← 任务一交付 |
| 4 | §3.2 Step 0 前置核查 | 否 | 30 min | 区间映射 24,292/24,292 复原 |
| 5 | §3.3 Step A 1,600 原型 | **是** | 15 min | 先 `--smoke 8 2`，人工看 16 份画像质量 |
| 6 | §3.4 Step B 用户表 | 否 | 10 min | **先造 5,000 人**跑通 §4 全部检验再放大到 50 万 |
| 7 | §3.5 Step C 单片 | 否 | 15 min | 单片 7 条硬断言全过 |
| 8 | §3.5 全 100 片 | 否 | 1–2 h | 无 |
| 9 | §3.6 合并 + §4 检验 | 否 | 30 min | **判别器 AUC ≤0.60 且迁移测试 ≥0.85** ← 任务二交付 |

**第 6 步的「先 5,000 人」是硬性要求**，不许直接上 50 万——沿用一贯的「先测 100 条看看」纪律。
5,000 人跑完 §4 四项检验，判别器 AUC 和迁移测试都过了，才允许放大。

---

## 7 已知坑（都是踩过的，别再踩）

1. **`python3` 是 3.8，会报 `TypeError: unsupported operand type(s) for |: 'dict' and 'dict'`**。
   `generate.py` 用了 3.9+ 的 `dict | dict`，而 shell 里的 `python3` 是 **3.8.10**，且沙箱里 `import numpy` 也失败。
   **一律用绝对路径 `/home/fengyuebo/miniconda3/envs/py311/bin/python`**（3.11.14 + numpy 2.4.2，
   既有 `__pycache__` 就是 `cpython-311`，与历史结果同口径）。
   同源问题在 sglang 启动脚本上是另一个版本：VS Code remote-server 把 `/usr/bin` 插到 PATH 最前，
   `/usr/bin/python` 是 **2.7.18**，`conda activate` 明明成功却报 `ImportError: No module named sglang`
   （**无引号** = python2 的报错格式）。启动脚本已改成用 `$CONDA_PREFIX/bin/python`。
   sglang 在 **py310**（0.5.3.post2 / torch2.8）；py311 是 PR 开发版，**只用来跑 recsim，不用来起服务**。
2. **本机只能单卡**：驱动 525.147.05 (CUDA 12.0) 撑不住 torch 自带 NCCL 2.27.3 (CUDA 12.8)，
   多卡 TP 会在 `ncclCommInitRank` 段错误。`gbatch -g 1`。
3. **Bash 沙箱里 `import numpy` 会失败**，但用 `python3 recsim/...` 跑脚本是好的。
   写临时分析脚本时别依赖 numpy，用标准库。
4. **`is_rand` 全 0**：真实 36.9M 行没有一条无偏曝光子集，所以**做不了无偏评测**。
   若后续需要，得去要 KuaiRand 原始的 `log_random_*.csv`。这是数据侧的硬缺口，汇报时要写明。
5. **BOM / −124 / `.0` / 605 空值**：见 §1.2，四个都是真实表的「脏」特征，漏一个就一眼假。
6. **`sim_log*.csv` 的行数是轮数 ×2**，不是轮数。2 万轮 → 40,001 行（含表头）。对外说明别写错。
7. **`--dual` 省钱不省时**：A/B 互换的 prompt 与正序共享用户侧前缀，SGLang 前缀缓存命中，
   双向 19.8 req/s vs 单向 28 req/s，不是砍半。

---

## 8 交付自查清单

任务一：
- [ ] `out/sim_log.csv` = 40,001 行 / 19 列 / 列名顺序与 §4.1 逐字一致
- [ ] `out/fuse_summary.json` 的 `n` = 20000，`stat` ≥ 0.87，`fuse` ≥ 0.85
- [ ] 位置偏置（融合路选 A 率）∈ [0.47, 0.53]
- [ ] 报告里写明负采样口径（全库随机 / 排除已见 / 仅有标题）

任务二：
- [ ] `sim_user_features_500k.csv` = 500,001 行 / 31 列 / §1.2 的六个脏特征全部复现
- [ ] `sim_log_500k*` = 5,000 万行 / 19 列 / §3.6 的 7 条硬断言全过
- [ ] `manifest.jsonl` 齐全，donor 最大复用 ≤60
- [ ] 判别器 AUC ∈ [0.45, 0.60]；迁移测试 ≥0.85；相关系数 −0.1157 附近
- [ ] `interaction_records.jsonl` 含 `round_no`，能证明多轮闭环（对齐文档 §2.2 / §2.5.2）
- [ ] 报告里**没有**任何「合成用户上的准确率」
- [ ] 视频侧一张表都没造（A-README §4.3 的要求）
