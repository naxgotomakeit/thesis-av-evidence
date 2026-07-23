# Thesis Experiment Freeze — V2.2

**Project:** Making Egocentric VQA Efficient (UCL MSc × Inferity AI)
**Revised:** 2026-07-23
**Status:** **ACTIVE — 唯一 source of truth**
**Supersedes:** V2.1 (2026-07-23)、V2 (2026-07-23)、`2026-07-21 EgoPolice 长视频主实验计划`、及其后所有口头/对话版本

**V2 → V2.1 修订摘要：** 修复 6 个 blocker（B1 候选单元唯一化、MDE 方向、frozen subset 拆分、Blind/Oracle 定义补全、cost 单位拆分、core claim 收缩）+ 3 项建议（answer model 两阶段冻结、存储策略、audit gate 第 8 条措辞）。

**Final review 修订（2026-07-23，激活前）：** §2.4 签署 Option A 并冻结 flat segmentation config；point evidence 措辞改为 provisionally valid；E3 增加 evidence-position regime 一致性规则；logging 补 online video decode / I/O；§6.3 措辞修正；answer decoding 与 LLM judge replication 分离。

**Final review 第二轮（2026-07-23）：** 新增 §2.0 **evaluation context 解耦** —— B0/B1/B2 定义不再写死 "full video"；E1/E2 冻结为 QaEgo4D canonical clip scope，仅 E3 映射至 parent video；logging 增加 `clip_uid` 与 `context_scope`。

**V2.1 → V2.2 最终修订（2026-07-23）：** 冻结 E4 的 amortization unit：正式 headline 分析 **E4a 以 canonical clip / canonical evaluation context 为摊销单位**，严格由 E2 日志解析；另设 **E4b parent-video deployment amortization** 作为后续部署分析，必须基于真实 parent-level preprocessing/index 成本，不得用 clip-level 日志假装 parent-level 一次建图。同步澄清 Ego4D 系 preprocessing 共享仅属于 deployment / E4b 扩展，不改变 E1/E2 的 canonical-clip 官方协议。

> **B0 / B1 / B2 定义自此不再修改。** 任何新想法只进入未来 B3+ 或新实验，不回头污染这条主干。

---

## 0. 引用规则

本文件标为 ACTIVE 后：

- 所有实验、代码、汇报、AI 协作（Claude / GPT / Codex）**只引用本文件**的编号与定义。
- 历史版本全部退休（见附录 A），仅作决策历史记录，不得作为当前正式编号引用。
- 代码分支名（如 `exp/b1-flat-structure`）属历史实现，不需重命名；论文与汇报的正式编号以本文件为准。
- 任何修改必须发布新版本号并注明 supersedes，不得就地口头改动。

**已知漂移事件：** 2026-07-21 至 07-23 期间，B1/B2 定义在不同对话中出现三个互不兼容版本，原因是冻结内容只存在于对话记录而非文件。

---

## 1. Thesis claim 与三张核心图

### 1.1 核心可验证 claim（V2.1 范围内，由 B0–B2 支撑）

> **Reusable offline evidence maps + question-conditioned evidence retrieval 能改善 egocentric video QA 的 quality–efficiency trade-off**：在尽量保持答案质量的前提下降低 time-to-evidence 与 cost-for-evidence（frames / tokens / model calls / latency / index storage）。

### 1.2 Extended hypothesis（尚未冻结，不写入核心 claim）

> Adaptive / question-type-aware resource allocation（Planner）是否带来额外收益，将在 B0–B2 结果暴露明确 bottleneck 后再评估。

**为何拆分：** 当前冻结的 B0–B2 全部使用 **fixed retrieval**，尚不包含任何 adaptive resource allocation。若核心 claim 从第一段就承诺 question-conditioned resource allocation，而 Planner 最终效果不佳，整个论文的立论基础会连带受损。拆分后，即使 extended hypothesis 未成立，B0–B2 的核心 claim 仍独立成立。

对应 Inferity brief：`time-to-evidence and cost-for-evidence are bottlenecks`；`Improve on VQA efficiency (+accuracy only if SOTA is too poor)`。

### 1.3 三张核心图

| Figure | 内容 | 来源 |
|---|---|---|
| **F1** **Quality**–Cost Pareto | B0 / B1 / B2 在 quality × cost 平面的位置 | E2 |
| **F2** Quality & cost vs video length | 同一批题在 2/5/10/20/40 min 的曲线 | E3 |
| **F3** Cumulative cost vs queries per **canonical clip** | benchmark-native break-even（**每种 cost 单位一张子图**） | **E4a**（由 E2 日志解析） |

> F1 使用 "Quality" 而非 "Accuracy"：主任务为 open-ended QA，accuracy 仅为 Closed 辅助信号之一。

---

## 2. 系统版本定义（FROZEN）

### 2.0 Evaluation context — 系统定义与输入作用域解耦（**关键**）

B0 / B1 / B2 的定义中一律使用 **evaluation context** 这一抽象输入，**不写死 "full video"**。evaluation context 的具体范围由**实验**规定，而非由系统版本规定：

| 实验 | evaluation context | 依据 |
|---|---|---|
| **E1 / E2** | **QaEgo4D 官方 canonical clip**（`video_id` / `clip_uid` 对应范围） | 问题在此范围内 well-posed，官方 protocol 保证 |
| **E3** | parent video (`video_uid`) 内的受控扩展窗口：2 / 5 / 10 / 20 / 40 min | 由 E0 保证扩窗后仍 well-posed |
| **E5 / E6** | 各自 benchmark 的官方输入范围 | 各 benchmark protocol |

**为什么必须解耦（这是 EgoPolice 教训的复发点）：**

QaEgo4D 的问题是基于 canonical clip 构造的；evidence timestamp 可映射回完整 parent video，**但这不等于把问题直接放进 45 分钟 parent video 后仍然 well-posed**。

```text
canonical clip (5 min)
  "Where did I put the cup?" → 唯一答案：table

full parent video (45 min)
  后续可能又把 cup 放进 cupboard
  → 原 GT 不再唯一 / 已过时
```

这与 §6.1 的 EgoPolice full-source variant 完全同构：**改变输入范围会改变问题的 well-posedness**。若 E2 直接使用 parent video，主 ablation 将建立在被污染的标签上，且污染会伪装成"方法不够好"。

**因此冻结：**

- **E1 / E2 使用 canonical clip scope。** E2 回答的是："在官方、well-posed 的 QaEgo4D protocol 下，B0 → B1 → B2 是否改善 quality–efficiency trade-off？"
- **只有 E3 才映射回 parent video**，且必须先由 E0 完成 contamination audit。
- 每次 run 必须记录 `context_scope`（见 §4.4），使任何结果都能被追溯到其输入范围。

> 附带好处：E6 换用 LongVideoBench 或其他 benchmark 时，只需规定该实验的 evaluation context，**无需改动 B0/B1/B2 定义**。

### 2.1 B0 — Dull baseline

```text
evaluation context → uniform-8 frames → answer model → answer
```

- 无 reusable index、segmentation、similarity、retrieval、hierarchy、audio、planner。
- **B0 的视觉证据 ≡ E1 的 uniform-8**：帧选择结果在 E1 与 E2 之间复用，**但 Open 与 Closed 是两次独立的 answer inference**（prompt 不同，见 §3.2）。
- 无 offline index，frame decoding + inference 全部计入 per-query online cost。

**回答：** 最简单直接的方法，quality 和 cost 是多少？

### 2.2 B1 — Flat reusable map + fixed retrieval

```text
offline: evaluation context → 1 FPS → embeddings → flat event map / index
online:  question → fixed retrieval over flat events → ≤8 representative frames → answer model
```

无 hierarchy、reranking、local refine、Router / Planner、audio、fallback。

**B0 → B1 回答：** reusable preprocessing + retrieval 这件事本身是否值得？

### 2.3 B2 — Hierarchical reusable map + SAME retrieval

```text
offline: evaluation context → Fine → Safe-Merge → Fluid Loose → Medium
online:  question → SAME retrieval over Medium events → ≤8 representative frames → answer model
```

**B1 → B2 回答：** 见 §2.4 的 claim 措辞。

### 2.4 候选单元定义（**已签署：Option A**）

**采用 Option A（与现有实现一致，2026-07-23 签署）：**

| | 候选单元 | 每单元贡献 |
|---|---|---|
| B1 | **flat event**（direct coarse flat segmentation 产出） | 1 representative keyframe |
| B2 | **Medium event**（Fine → Safe-Merge → Fluid Loose 产出） | 1 representative keyframe |

- 单元**类型相同**（均为 event + representative keyframe），差异仅在于 event 如何导出（flat 直接分段 vs 层级合并链）。
- 文档中不得再出现 "帧 **或** flat coarse event" 这类二选一表述。

**残留混淆变量（必须处理）：** flat 与 Medium 的 **event 数量与时长分布不同**，B2 可能仅因单元更少更长而覆盖更多内容取胜。

**强制缓解措施：**

1. B1 与 B2 均须报告 `event_count`、`mean_event_duration`、`event_duration_std`（per video）。
2. 若两者 event count 差异超过 ±25%，必须追加 **event-count-matched control**：调整 B1 的 flat 分段粒度，使其 event 数与 B2 的 Medium 数近似匹配，作为 B1' 重跑一次。

**Claim 措辞（因存在上述混淆，不得夸大）：**

> B1 → B2 tests the **hierarchical event-derivation package**（层级合并链 vs 直接平坦分段），**not pure hierarchy in isolation**.

**未采用的选项 B（记录备查）：** B1 与 B2 均在同一 leaf-level unit（Fine segment 或 1 FPS frame）上打分与选择，B2 仅用 hierarchy 组织/导航候选集。归因最干净，但偏离现有实现、需额外工程量，在 MSc scope 下不采用。

> **签署：** Option A ✅ — 2026-07-23。理由：与现有代码一致；claim 已限制为 hierarchical event-derivation package；event-count-matched control 已覆盖最主要的粒度混淆。

### 2.4.1 B1 flat segmentation config（**必须冻结**）

"direct coarse flat segmentation" 不得停留在描述层面。以下参数须在 E2 开跑前写入配置文件并纳入版本控制，任何变更需发布新版本号：

```text
embedding_backbone            # 与 B2 相同
input_fps                     # 与 B2 相同
change_criterion              # similarity / distance 度量定义
threshold_or_quantile         # 绝对阈值或 per-video quantile，二选一并写死
min_segment_duration_s
merge_rule                    # flat 分段的合并逻辑（若有）
representative_frame_rule     # 与 B2 相同
config_hash
```

> 未冻结这些参数，则更换一个 threshold 即可让 B1 本身漂移，B1 → B2 的比较随之失效。B2 的 hierarchy config（Safe-Merge / Fluid Loose 参数）同样要求写入配置文件并记录 hash。

### 2.5 "SAME retrieval" 的精确定义

B1 与 B2 必须严格相同：

- same embedding backbone
- same similarity / scorer
- same aggregation 方式
- same Top-K rule
- same final model-facing budget：**≤ 8 frames**
- same representative-frame 规则：每个命中单元贡献 1 张 representative frame

**唯一允许不同：** 候选单元如何导出（§2.4）。

### 2.6 暂不冻结的部分

**B3 / B4 编号现在不定。** Audio、Planner / adaptive routing、reranking、local refine、fallback / re-retrieval 均**不得**塞入 B0–B2，待 E2 结果暴露明确 bottleneck 后再决定加什么、如何编号。

---

## 3. 实验矩阵 E0–E6（FROZEN）

| ID | 名称 | 数据集 | GPU 成本 | 前置依赖 |
|---|---|---|---|---|
| E0 | Well-posedness audit | QaEgo4D + Ego4D parent videos | ~0 | 无 |
| E1 | Headroom gate | QaEgo4D Closed + Open | 低 | 无 |
| E2 | Incremental ablation B0→B1→B2 | QaEgo4D | 主要成本 | E1 通过 |
| E3 | Controlled length scaling | E0 clean common subset | 中 | E0 + E2 |
| E4 | Break-even / amortization（E4a 主分析 + E4b deployment） | E4a: —；E4b: parent video | E4a **0**；E4b 待实测 | E4a: E2 日志；E4b: parent-level index 实测 |
| E5 | Specialized branches | EgoSound / EgoSchema / long-video | 中 | E2 |
| E6 | Method comparability | LongVideoBench 或同类（**待定**） | 低 | E2 |

---

### E0 — Well-posedness audit

**目的：** 确认"从 canonical clip 切换到 parent video 并围绕 evidence 自然扩展 context"这一**作用域变更**不会破坏 GT。E0 只服务 E3；E1/E2 因使用官方 canonical clip scope 而不受此风险影响（§2.0）。

**风险：** QaEgo4D 多为 episodic memory 问题。窗口从 2 min 扩到 40 min 后，同类事件可能再次发生 → GT answer 不唯一或过时。若不处理，会系统性表现为"长度越长质量越低"，被误读为 length scaling 效应，实为 label 失效率上升。

> 此风险与 EgoPolice distractor 污染同构：**改变输入范围会改变问题的 well-posedness**。

**步骤：**

1. 筛出 parent video 足以支撑最长窗口（40 min）的问题。
2. 用 narration / annotation 自动标记"扩展区间内出现同类事件"的 at-risk 问题。
3. 人工核验子集，估计**污染率**。
4. 优先保留 GT 事件在 parent video 中唯一出现的问题，产出 clean common subset。

**同时产出（point-evidence 统计）：** 统计 `start == end` 的比例（当前观测约 8–9%）及其分布，供 §3.2 Oracle 与 localization metric 使用。

**交付物：** `e3_clean_common_subset.json`、污染率报告（作为 E3 的 known limitation 正式写入论文）、point-evidence 统计。

**约束：** E3 所有 length bin **只跑这同一批题**（nested common subset），保证 paired 比较。若 subset 过小，报两条曲线（严格 common subset 版 + 全量宽松版）并说明差异。

---

### E1 — Headroom gate

**目的：** 在投入 E2 主算力前，确认 benchmark × answer model 组合真的能测出 evidence selection 的作用。

**Evaluation context：QaEgo4D 官方 canonical clip（`clip_uid` 范围）。** 不使用 parent video（§2.0）。

**运行条件（四条）：**

```text
Blind        无视觉输入
Uniform-8    = B0 的帧选择
Uniform-32   budget anchor
Oracle       GT evidence window 内 ≤8 frames
```

**Blind 的两种形式（分别定义，不可混用）：**

```text
Closed Blind = question + options,  no video
Open  Blind  = question only,       no video
```

**Oracle 定义（含 point evidence 规则）：**

```text
interval GT (start < end)
  → 在 GT window 内选择最多 8 个 unique frames

point GT (start == end)
  → 使用对应帧；若该时间戳无解码帧，取时间上最近的一帧
  → 不得因 duration == 0 自动剔除该样本
```

> **point evidence 的处理状态（provisional）：** 当前仅知 `start == end`（约占 8–9%），且数据中确实存在极短的正常 interval。因此暂按 **potentially valid instantaneous evidence** 处理——**不自动排除**，但**也不断言其全部为合法单帧证据**。最终 protocol 待 E0 与 source annotation semantics 核验后确定并写入本文件。剔除会无故损失近十分之一数据并可能引入系统性偏差，故默认保留。

**Localization metric 对应规则：**

```text
interval GT → IoU / temporal overlap
point GT    → distance-to-GT + tolerance-based hit（tolerance 需在冻结时写死）
```

**Gate 判据（结果出来之前即生效，不得事后调整）：**

| 结果 | 行动 |
|---|---|
| Oracle 明显 > Uniform-8（paired test 显著 + practical gap） | 冻结 answer model，继续 E2 |
| 有 gap 但过小 / 不稳定 | **不硬跑 E2**；先换更强 answer model 或重新评估 benchmark |
| 基本无 gap | 该 benchmark × answer model 组合无法承载 evidence-selection claim；回到 benchmark 选择 |

**不硬编码百分比阈值**（"15–20 points" 仅为经验参考）；以 paired statistical evidence + practical gap 共同判定。

**Blind 结果的额外用途：** 记录答案分布与选项位置偏置，作为所有后续质量指标的解读基线。

**Answer model 两阶段冻结（见 §4.2）：** E1 阶段 answer model 为 *candidate*，允许在 gate 未通过时更换；gate 通过的那一刻起冻结，E2 及之后不得再换。

---

### E2 — Incremental ablation（thesis 核心）

**Evaluation context：QaEgo4D 官方 canonical clip（`clip_uid` 范围）。** 不使用 parent video（§2.0）——E2 是**官方 protocol 下的干净 ablation**，标签 well-posedness 由官方构造保证。

**E2 回答的问题（措辞需在论文中保持一致）：** 在官方、well-posed 的 QaEgo4D protocol 下，B0 → B1 → B2 是否改善 quality–efficiency trade-off？

**运行：** B0 → B1 → B2，全部在 QaEgo4D，使用 §4.1 冻结 manifest。

**Quality 指标：**

- Open-ended answer quality（**主任务**）
- Closed MCQ accuracy（辅助，低噪声信号）
- Evidence localization / recall（按 §3.2 的 interval / point 规则分别计算）

**结构指标（§2.4 要求）：** `event_count`、`mean_event_duration`、`event_duration_std`

**Efficiency 指标：** 见 §4.4 logging schema。

**Open-ended 评估：** 不得只依赖单一 LLM judge。同时报告 ①官方 lexical/semantic metric（若 benchmark 提供）②固定 LLM judge ③小规模人工 sanity check。

**⚠ Answer generation 与 LLM judging 是两套独立配置，不可混淆：**

```text
Answer generation（§4.3）
  deterministic / frozen decoding
  单次生成，不做 repeated sampling

LLM judging
  judge model 版本 + judge prompt 单独冻结
  若做 repeated judging，须明确 repetition_count / seed / aggregation 规则
  并报告 judge 内部一致性
```

> 即 "多次采样" 只发生在 judging 环节，**answer model 本身始终是确定性单次生成**。

**统计检验：** B0/B1/B2 回答**同一批题**，一律使用 **paired test（McNemar 或 paired bootstrap）**。

**统计功效判据（V2 中方向写反，此处为正确版本）：**

```text
若 MDE  >  预期/观察到的方法增益   → underpowered → 只报趋势，不报显著性
若 MDE  <  预期/观察到的方法增益   → 有足够功效  → 可报显著性

例：预期增益 3%，MDE 8%  → 测不出来
    预期增益 8%，MDE 3%  → 可以测出
```

每个报告单元需附其 MDE。

**注意：** MCQ 与 open-ended 结果的相关性本身值得报告——若某方法 MCQ 涨而 open-ended 不涨，说明 MCQ 选项脚手架效应在起作用。

---

### E3 — Controlled length scaling

**协议：Natural context expansion**（优于合成拼接：不换 domain、不换摄像人、不换场景分布、无人为 edit boundary）

**Evaluation context：此处才从 canonical clip 切换到 parent video（§2.0）。** 映射路径：

```text
canonical QA (clip_uid)
  ↓ absolute evidence timestamp mapping
parent video (video_uid)
  ↓ E0 contamination audit 通过的题目
2 / 5 / 10 / 20 / 40 min 受控窗口
```

**前置硬约束：** 未经 E0 audit 的题目**不得**进入 E3。E0 正是为这一次作用域切换而设。

```text
同一 question / 同一 answer / 同一 evidence / 同一 parent video
围绕 evidence 扩展 context window：

2min          ────█────
5min        ───────█───────
10min    ───────────█───────────
20min ─────────────────█─────────────────
40min
```

**运行：** B0 vs B1 vs B2，题目来自 `e3_clean_common_subset.json`。

**第二维度：evidence 在窗口中的相对位置（beginning / middle / end）。** 长上下文存在 serial position 效应（lost-in-the-middle）。若 B0 uniform 在证据靠边时显著崩溃而 retrieval 不受影响，这是独立的强卖点。**不得将 evidence 一律固定在窗口正中。**

**位置一致性规则（FROZEN，防止长度与位置同时变化）：**

> 在主 length-scaling analysis 中，同一道 question 跨 2/5/10/20/40 min 时**必须保持相同的 evidence-relative-position regime**。

两种合法实现，二选一并写入配置：

```text
方案 1（推荐，成本低）
  每道题在 E0 阶段被分配一个固定 regime
  Q1 → beginning across all lengths
  Q2 → middle    across all lengths
  Q3 → end       across all lengths
  position 作为 between-question 因子分析

方案 2（成本 ×3）
  同一道题分别跑 beginning / middle / end 三套
  每套内部跨长度保持位置一致
  position 作为 within-question 因子分析
```

**禁止：** 同一道题在 2min 时 evidence 居中、5min 时靠前、10min 时靠后——此时长度与位置同时变化，曲线无法归因。

**备用协议（stress test，非主实验）：** synthetic distractor padding，控制 padding 与 evidence 的语义相似度（random unrelated vs same-scene similar）。

---

### E4 — Break-even / amortization

E4 是一个 umbrella experiment，分为 **E4a（正式 benchmark-native headline）** 与 **E4b（后续 deployment analysis）**。两者的 amortization unit 必须分开，禁止混算。

#### E4a — Benchmark-native break-even（**正式 F3，零额外 GPU**）

**摊销单位固定为：QaEgo4D canonical evaluation context，即 `clip_uid` / canonical clip。**

原因：E2 的 quality evaluation 与 offline preprocessing 都在 canonical clip scope 内执行，因此 E2 日志真实测到的是：

```text
canonical clip
  → offline map/index cost（B1/B2）
  → repeated queries within the same canonical clip
```

因此 E4a 严格回答：

> **同一个 canonical clip 被查询多少次以后，B1/B2 的 reusable map 相比每次重新执行 B0 更划算？**

**完全由 E2 run logs 解析得出，无独立 evaluation subset、无额外模型运行。**

```text
C_B1_clip(N) = C_offline_B1_clip + N · C_online_B1_query
C_B2_clip(N) = C_offline_B2_clip + N · C_online_B2_query
C_B0_clip(N) = N · C_B0_per_query

Break-even N* = 两条对应 cost 曲线相交时的查询数
```

> **禁止**把 E2 的多个 clip-level offline cost 事后合并，然后声称“一个 parent video 只建了一次 index”。如果实际实验是 clip-by-clip 建图，就只能报告 clip-level amortization。

**关键要求：cost 不得混成单一数字。** 不同单位不可相加，F3 为一组子图：

| 子图 | 单位 | 说明 |
|---|---|---|
| Compute break-even | GPU-seconds | offline + online 分开后按相同单位累加 |
| Latency break-even | wall-clock seconds | 需说明串行/并行假设 |
| Monetary break-even | £/$ | 若可估；必须注明硬件/API 单价假设 |
| Token / call curve | tokens、model calls | 分别作曲线，不折算 |

**Storage 单独报告：**

- `index_size_per_video_minute`
- **B0 = 0 additional index storage**（不是 “0 storage”）。原始视频存储是所有方法共同成本。

**报告点：** N = 1 / 2 / 5 / 10 / 20 / 50，B0 / B1 / B2 各一条曲线。

**可能的结论形态：** query 少时 B0 赢；随着同一 clip 被反复查询，B1/B2 的一次性 offline cost 被摊薄。B1 与 B2 的交点也可单独报告，说明 hierarchy 的额外预处理成本需要多少次查询才能回本。

**为什么不能直接使用 benchmark 的天然 queries-per-video 作为单点结论：** benchmark 的问题密度是数据集构造属性，不应替方法选择摊销系数。E4a 把 N 显式作为自变量。

#### E4b — Parent-video deployment amortization（**后续部署分析，不替代 E4a**）

E4b 面向 Inferity / bodycam 的真实使用逻辑：

```text
parent video / long bodycam recording
  → build reusable index ONCE
  → Q1
  → Q2
  → Q3
  → ...
```

**摊销单位：`video_uid` / parent video。**

它回答：

> **一段完整长视频被查询多少次后，parent-level reusable index 开始值得？**

但 E4b 必须满足以下约束：

1. **必须真实测量 parent-level preprocessing/index cost。** 不得从 clip-level E2 日志推算成“整段只建一次”。
2. 若 parent-level map 会改变 segmentation / hierarchy / retrieval 结果，则它已不再是 E2 的同一个系统实例；**quality 结论必须另行验证**，不能直接继承 E2 accuracy/quality。
3. 若仍使用 QaEgo4D canonical QA 做 online query，retrieval/answer 的可见证据范围必须明确限制在对应官方 scope，除非另有经过 E0-style well-posedness audit 的扩窗协议。
4. E4b 是 **deployment / external-validity analysis**，不用于替换 E2/E4a 的官方 benchmark 主结论。

因此当前冻结关系为：

```text
E2   = canonical clip 上的干净 quality–efficiency ablation
E4a  = canonical clip 上的正式 break-even（直接解析 E2 logs）
E3   = parent video 上的受控 length scaling
E4b  = 后续真实 parent-level index-once 的 deployment amortization
```

---

### E5 — Specialized branches

**前置成本提醒：** 每条支线都需在该数据集上**先建 map 并跑 B0/B2 作为锚点**，不是"只跑一个新 stage"。

| 支线 | 数据集 | 前置 runs | 比较 |
|---|---|---|---|
| Audio | EgoSound（2,346 QA / 260 videos，已处理） | B0, B2 @ EgoSound | B2 visual-only vs +audio |
| Audio no-harm | QaEgo4D | 已有 E2 | 加 audio branch 是否徒增成本 / 误导 retrieval |
| Planner | QaEgo4D (LOCAL-ish) + EgoSchema (GLOBAL-ish) + long-video | B0, B2 @ EgoSchema 及 long-video | fixed vs adaptive |

**Planner 报告要求：** 主表必须是 **question type × method 矩阵**（LOCAL / GLOBAL / MIXED），聚合数字仅作附注。一个"帮了 GLOBAL 但伤了 LOCAL"的 planner 在聚合下会抵消为 0，而这恰是最有信息量的结果。

**Scope 限定：** modality decision 只能在有多模态信号处检验（EgoSound / EgoPolice）。在 EgoSchema / long-video 上，Planner 实际只测 scope + budget allocation，必须写明。

**Audio 的必要性：** brief 明确写 `egocentric videos (with audio)`，audio 不得降级为 future work。EgoSound 为正式落点。

---

### E6 — Method comparability

**目的：** 回答 "你的方法和已有选帧 / long-video QA 方法比怎么样？"

**背景：** 与本工作最接近的一批方法（AKS、Q-Frame、FOCUS、GIFT、MDP3、VideoTree、ReQuest、ToolMerge）几乎全部在 **Video-MME / LongVideoBench** 上评估，而非 egocentric benchmark。不做此实验则 efficiency 数字只能与自己的 B0 比较。

**Benchmark：待定**（LongVideoBench 或 Video-MME long split）。选定后写入本节并发布新版本号。

**设计：** 小规模、标准化。单一 length split、固定 frame budget、2 个开源 baseline（AKS、Q-Frame 均有代码）。

**对齐 literature 的报告约定（沿用 FOCUS/AKS 协议）：**

- 关闭字幕、zero-shot、answer model 参数冻结，仅变化 frame selection
- 固定 MLLM 与选帧数量
- 报告 **Frames Seen (%)**
- 明确声明 **pre-filtering（1 fps 下采样）是否计入成本**（AKS 官方 pipeline 默认包含此步，不声明则不可比）

**必读直接竞品：** ReQuest（uncertainty-driven question-adaptive selection + re-thinking routing；差异化在于其为 online per-query，本工作为 offline reusable map + 摊销）、ToolMerge（LLM planner 分解 query 为 tool calls；自建 M2M，每题按构造锚定时间区间）、VideoTree（hierarchical query-adaptive frame pyramid）、KFS-Bench（选帧准确率的指标定义）。

> **注意：** 主任务转为 open-ended 后，与该条线（全为 MCQ + fixed frame budget）的直接可比性下降。E6 应使用 Closed / MCQ 形式进行，以保持可比。

---

## 4. 全局冻结项

### 4.1 Frozen question manifests（**三份，不是一份**）

```text
qaego4d_open_eval_ids.json      → E1 / E2 的 Open 主结果
qaego4d_closed_eval_ids.json    → E1 / E2 的 Closed 辅助结果
e3_clean_common_subset.json     → 由 open eval set 进一步筛出（E0 产出）
                                   E3 的 2/5/10/20/40 min 全部共用
```

- Open 与 Closed 的题量与覆盖范围不同（Closed 通常为 Open 的子集），**不可能是同一份 list**；实际数量以 release schema 为准，填入 manifest 头部。
- **E4 无独立 manifest**：`derived from the exact E2 run logs; no independent evaluation subset.`
- 三份 manifest 均纳入版本控制，并在每次 run 中记录其 hash。

### 4.2 Answer model 两阶段冻结

```text
E1 阶段：candidate answer model = Qwen2.5-VL-7B-Instruct（BF16，优先无量化）
   ↓ gate pass
FREEZE：选定的 answer model 版本写入本文件并锁定
   ↓
E2 及之后：绝不再更换
```

**Qwen2-VL-2B 的角色（明确限定）：** 仅作为 **cheap diagnostic / engineering model**，用于 pipeline 调试、dry-run、快速迭代。**不得**出现在 E1–E6 的任何正式报告数字中。

### 4.3 Deterministic config

**(a) Answer generation（确定性，单次生成）：**

```text
prompt template（Open 与 Closed 各一份）
seed
temperature / decoding params
max tokens
max_pixels
frame preprocessing
answer model version
```

**(b) LLM judge（独立冻结）：**

```text
judge model version
judge prompt
repetition_count / judge seed（若做 repeated judging）
aggregation 规则
```

> 两套配置分别记录 hash。answer model 不做 repeated sampling；repetition 只可能出现在 judging 环节。

### 4.4 Logging schema

```text
offline (per video):
  preprocessing_time_s
  compute_gpu_s
  frames_decoded
  frames_embedded
  index_size_bytes
  index_size_per_video_minute
  event_count                    # §2.4 要求
  mean_event_duration_s
  event_duration_std_s

online (per query):
  video_decode_time_s            # 含 seek / I/O；B0 的主要隐藏成本
  frames_decoded_online
  retrieval_time_s
  answer_model_time_s
  total_latency_s
  frames_shown
  visual_tokens
  text_tokens
  model_calls / api_calls
  peak_vram_gib

IDs:
  video_uid
  clip_uid
  context_scope        # canonical_clip | parent_window_{2,5,10,20,40}min | benchmark_default
  question_id
  stage            (B0 / B1 / B1' / B2)
  task             (open / closed)
  run_id
  config_hash
  manifest_hash
```

**强制要求：**

- offline 与 online cost **必须分开记录**，per-video 与 per-query 两级粒度。E4 完全依赖此结构，混记则需全部重跑。
- **B0 无 offline stage，其 video seek / decode / I/O 成本必须完整计入 online**（`video_decode_time_s`、`frames_decoded_online`）。若 B1/B2 的解码成本认真计入 offline 而 B0 的解码成本消失，break-even 曲线会系统性偏向 B0。B1/B2 在 online 阶段若需回读原始帧，同样计入这两个字段。
- **E2 批量开跑前，先用 1 个视频完整 dry-run**，确认所有字段真的能记录。

---

## 5. Benchmark 角色（当前状态）

| Dataset | 角色 | 状态 |
|---|---|---|
| **QaEgo4D (Open + Closed)** | 主干：E0–E4 | 已有权限；主 open-ended benchmark |
| **EgoSound** | Audio ablation（E5） | 已处理 2,346 QA / 260 videos；**正式引用前需核官方名称、来源、citation** |
| **EgoSchema** | GLOBAL / Router sanity（E5） | 已有 pipeline，边际成本近零 |
| **LongVideoBench 或同类** | Method comparability（E6） | **待定** |
| HourVideo | 外部长视频验证 | 500 videos / 20–120 min / **12,976** 五选一（MM-Ego 论文转述的 121,976 为 typo）；需 audit |
| EgoMemoria (MM-Ego) | 长度缩放候选 | 629 videos / 7,026 QA / 0.5–60 min；annotation gate 未过，**不阻塞前四步** |
| EgoPolice | Bodycam domain case study | 见 §6 |
| CASTLE / X-LeBench | 后期 external case study | 不阻塞；CASTLE 约 8.22 TB，challenge 仅 185 closed-form questions |

### 5.1 Benchmark audit gate 清单

1. evidence / key frame 标注是否在 release 中、粒度如何
2. debias 协议形式（官方 biased-question 列表 vs 需自行复现）
3. distractor 构造方式 + 抽 20 题人工查 well-posedness
4. 视频 UID 映射与下载可达性
5. question type 分布（Planner 需要 scope 多样性）
6. 各 length bin 的视频数与题目数，并反推 MDE
7. 与 QaEgo4D 的 UID 重叠（既是 leakage 风险，也是 preprocessing 复用红利）
8. **Answer-model headroom（最早的 compute gate）**

**关于第 8 条的正确措辞：** headroom 检查是**最早可执行的 compute gate**，但并非"不需要任何 schema"。它需要：

```text
最低要求：question 文本、answer / options、video mapping   （Blind / Uniform 条件）
Oracle 条件额外需要：evidence annotation
```

因此正确流程为：

```text
minimal schema / access sanity
  → headroom 作为最早的 compute gate
  → 完整七项 audit 并行推进
```

### 5.2 Ego4D 系 preprocessing 共享

QaEgo4D、EgoMemoria、EgoSchema、HourVideo 均基于 Ego4D，**从 deployment 角度**具备共享 parent-level 1 FPS + embedding index 的潜力，这正是 reusable map 的长期价值之一。

但必须区分两层：

- **E1 / E2 官方主实验：** preprocessing/index scope 与 evaluation context 一致，固定为 QaEgo4D canonical clip；不得因为“未来可共享”而偷偷使用 clip 外信息。
- **E4b / deployment extension：** 可实现 `parent video → index once → multiple queries / multiple clip scopes`，但必须真实测量 parent-level offline cost，并在 representation 发生变化时单独验证 quality。

因此，“Ego4D 系 preprocessing 可共享”是 **deployment hypothesis / engineering opportunity**，不是 E2 已经默认实现的事实。

### 5.3 存储策略（V2 已修正）

**QaEgo4D raw videos：保留至 E0–E3 全部完成。** 原因：E0 contamination check、E3 context 扩展、B0 原视频采样、failure analysis、以及后续可能的 local refine 均需要原始视频。

**大规模 external benchmark（EgoMemoria / HourVideo 等）：** 才采用 `下载 → 1 FPS 解码 → 建 index → 验证通过后按 UID evict` 策略，需要时 on-demand re-fetch。

---

## 6. 已知问题与决策记录

### 6.1 EgoPolice protocol mismatch

**官方 protocol（已核实论文）：** 12,000 题 MCQ，基于 1s / 10s / 60s **预先切好的 clips**；五选一，末项恒为 "None of the above"；问题文本全部相同（`Which action is happening in this video clip?`）——定位信息由视觉输入承载。Qwen2.5-VL-7B 官方：1s 52.5 / 10s 55.8 / 1min 55.5，chance = 20%。

**改造为完整 source video 输入后出现的问题（属本工作的 variant，非官方 bug）：**

1. **任务欠定**：generic question 缺少 temporal referent，同一视频多题无法区分
2. **Distractor 在其他时段可能成立**：错误选项仅排除"该 clip 内未发生"，未排除"全视频未发生" → full-source variant 可能 ill-posed，**比例需单独量化**
3. 官方所谓 needle-in-haystack 的 haystack 仅 1 分钟

**已有结果处理：** Blind 17.35% / Uniform-8 26.53%（GT Hit@8 24.49%）保留为 diagnostic，引用时必须注明为 "ill-posed full-video variant"，**不得与官方 55.5% 直接比较**。

**当前定位：** 官方 clip protocol → external bodycam sanity check + Oracle upper bound；完整 source videos + 自建 QA → domain case study。

### 6.2 EgoPolice 自建 QA protocol（case study，非 benchmark）

- 规模 30–100 题，人工核查，类型：existence / counting / ordering / temporal grounding
- **Distractor 规则 = query-conditioned well-posedness**：distractor 不得满足当前问题的时间 / 实体 / 关系 / 事件约束。**不采用**"全视频从未发生"这一过强规则（会使题目退化为存在性判断）；对 existence / counting 类 GLOBAL 问题，二者自然重合。
- **刻意保留并标记一批"在别处真实发生但违反时间约束"的 distractor**——检索到错误时间窗的系统会被其精准欺骗，是测试 retrieval precision 的高价值样本，错误分析时单独统计。
- 生成规则在查看任何模型输出**之前**冻结。
- 保留一小批 audio-relevant 问题。
- 指标：time-to-evidence、evidence recall、incident timeline reconstruction、cost。

### 6.3 来自 Gabe / Inferity 会议的约束

- **目标 deliverable 是 open-ended incident summarization**，非固定类别 event classification。**除 QaEgo4D（Open）外，大多数候选 long-video / method-comparability benchmark 以 MCQ 为主**，与 open-ended incident summarization 存在结构性错配——这是 QaEgo4D 被选为主 open-ended benchmark 的核心理由。
- Gabe 指引：先用约 **10 个** EgoPolice development videos 起步；旧计划中的"冻结 50 个视频"作废。
- **Summarization 判分方式尚未设计**（开放项）：需在 case study protocol 阶段确定，避免落入 LLM-judge 噪声问题。

### 6.4 待处理的技术问题

- **1515s hub 现象**：C-RADIO 检索中所有 query 的 Top-1 异常集中于同一帧，疑为 hubness（黑帧 / 过曝 / logo 帧 / embedding norm 离群）。正式 retrieval 实验前需：肉眼核查该帧、检查 embedding norm、试 centering / QB-norm。
- **C-RADIO vs DINOv2 threshold**：C-RADIO 的 smoothed MAD 比 DINOv2 窄约 8.49×，绝对阈值不可直接迁移。若后续比较，改用 per-video quantile / relative threshold。此项不属 B0–B2，暂不动 frozen pipeline。

---

## 7. 执行顺序

```text
现在并行：

Line A ─ E0 Well-posedness audit          （零 GPU；产出 E3 subset + point-evidence 统计）
Line B ─ E1 Headroom gate  ──通过──▶ freeze answer model ──▶ E2 B0 → B1 → B2
Line C ─ 全局冻结项生成                     （3 份 manifest / config / logging schema + 单视频 dry-run）
Line D ─ Benchmark audits（并行，不阻塞）     （先拉 annotation，先跑 headroom）

E2 完成后：
  ├── E3 Length scaling      （需 E0 交付物）
  └── E4a Benchmark-native break-even
      （canonical clip；从 E2 日志解析；零额外算力）

之后 / 并行扩展：
  E4b Parent-video deployment amortization
      （必须真实测 parent-level index-once 成本）
  E5 Audio / Planner specialized
  E6 Existing-method comparability
```

**关键原则：** benchmark audit 是**并行任务，不是前置依赖**。E0–E4 全部可在手上已有的 QaEgo4D 上完成；真实长视频 benchmark 的作用是 external validity。

---

## 8. 开放项清单（不阻塞 ACTIVE，但需在对应实验前关闭）

| # | 开放项 | 类型 | 位置 | 阻塞 |
|---|---|---|---|---|
| 1 | B1 flat segmentation config 参数写入配置文件 | 待冻结 | §2.4.1 | **E2** |
| 2 | E3 position regime 方案 1 / 2 选定 | 待定 | E3 | **E3** |
| 3 | point GT 的 tolerance 数值写死 + annotation semantics 核验 | 待定 | E0 / E1 | E1 |
| 4 | 三份 manifest 的实际题量（以 release 为准） | 待填 | §4.1 | E1 |
| 5 | E4b parent-level index-once 实现与真实成本测量 | 待实现 | E4b / §5.2 | E4b（**不阻塞 E4a**） |
| 6 | E6 benchmark 选定 | 待定 | E6 | E6 |
| 7 | EgoSound 官方名称 / 来源 / citation 核实 | 待核 | §5 | 论文写作 |
| 8 | Summarization 判分方式设计 | 待设计 | §6.3 | EgoPolice case study |

**已关闭：** B1 候选单元选择（§2.4，Option A，2026-07-23 签署）。

---

## 附录 A — 退休版本记录

| 版本 | 时间 | B 编号定义 | 退休原因 |
|---|---|---|---|
| 7-21 freeze | 2026-07-21 | B1 retrieval / B2 Router / B3 representation / B4 Planner | 建立在 EgoPolice 98Q 之上；question degeneracy 使 Router 无输入信号 |
| 工作版 | ~07-22 | B1 flat structure / B2 hierarchical Medium / B2+SC | EgoPolice query 崩溃后的应急版；benchmark pivot 后失效 |
| 讨论版 | 07-23 | B1 map only / B2 retrieval+rerank+refine | B1 不可运行（无检索定义）；B2 一步含三个能力；hierarchy 变量丢失 |
| V2 | 07-23 | 同 V2.1 但 B1 单元二义、MDE 方向写反、manifest 单一化、Blind/Oracle 定义不全、cost 混单位、claim 过度承诺 | 由 V2.1 修订 |
| V2.1 | 07-23 | evaluation-context 解耦后版本；E4 仍写作模糊的 “queries per video” | E4 amortization unit 未区分 canonical clip 与 parent video；由 V2.2 冻结为 E4a/E4b |

**V2.2 的吸收关系：** flat → hierarchy 的比较（工作版）与 retrieval / Planner 的链条（7-21）均被保留，重排为单变量 incremental chain，Planner / Audio 移出编号待定。

## 附录 B — 术语

- **time-to-evidence**：从问题到达至定位相关证据所需时间
- **cost-for-evidence**：获得该证据所消耗的 frames / tokens / model calls / storage
- **Frames Seen (%)**：frame-level forward pass 数量占"对全部帧打分"的比例（literature 通用效率指标）
- **MDE**：Minimum Detectable Effect，当前样本量下能检测出的最小差异
- **MDA**：Mean Debiased Accuracy（EgoMemoria）
- **nested common subset**：所有 length bin 共用的、能支撑最长窗口的同一批题
- **point evidence**：`start == end` 的瞬时 GT 标注
