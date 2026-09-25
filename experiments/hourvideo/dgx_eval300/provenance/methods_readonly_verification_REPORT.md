# HourVideo R1 / Staged Methods 只读核实报告

核查日期：2026-09-15（Europe/Athens）  
当前主机：`dgx01`  
边界：只读取现有代码、配置、manifest、资料清单及归档文件目录；未运行实验、模型或 API，未重新计算准确率，未修改任何既有代码或实验资产。

## 结论摘要

1. **R1 分段类别分布：正式实现目前不能确认。** 本机没有正式 HourVideo Eval300 R1 builder 的冻结源码、构建配置、manifest 或正式 R1 索引 lineage。因此，`p_j` 究竟取检测实例次数、含该类的帧数、`observed_frame_ratio` 还是其他统计，如何归一化，同帧多实例如何计数，无检测 Medium 与空/零分布的 JSD 如何处理，以及边界统计是否与标签摘要统计相同，均没有足够证据。不得补写公式或伪代码。

2. **Staged 的 Fine 选择与再次选区：Local/Planner-only 对应本地下游及 Full Staged paired100 最终版均不能确认。** 本机没有 accepted/frozen Staged 或 Planner-only API bundle，也未找到 `paired100` 的正式版本绑定记录。因而不能确认 Fine 候选来源、排序公式或权重，不能确认 Medium/区域能否重选及其条件，也不能确认程序按区域、Fine ID 还是实际帧去重；prompt 与 controller/core 是否一致同样未知。

3. **Staged 高层流程：所提 `Planner → Shared 判断 → 按需 Fine 检查 → 反馈 Shared → Final` 不能由本机正式证据确认。** 第一次 Fine 前是否已有 Shared 判断、何时可跳过 Fine、循环/终止由谁控制、格式重试是否独立于新增视觉检查，以及 Local 与 Full 的系统图差异，均须等学校端正式冻结包恢复后再写。现有 VideoSEAL Dense-H、Direct 或辅助 Fine-reranking 代码不能代替 Staged。

## 一、R1 分段中的检测类别分布

### 可确认内容

- 2026-09-14 的本机结构化盘点已把正式 R1 相关来源标为缺失：`DATA_INVENTORY.md` 第 49、51–55 行；`experiment_registry.csv` 第 11、14、22 行。
- 盘点明确记录未找到三个历史线索：`run_hourvideo_dev50_r1_offline_v1.py`、`outputs/experiments/hourvideo_dev50_r1_offline_v1`、`hourvideo_r1_r3_offline_processing_cost_eval300_v1.md`。
- 本轮再次检查 `/home/naxucl/data/HourVideo`、`/home/naxucl/projects`、`/home/naxucl/migration_unpack_20260812_0213`，并只读查看 `HourVideo.zip` 与 `videoseal_generated_20260812_0213.tar.zst` 的文件目录；仍未定位到正式 R1 builder、R1 构建配置/manifest，或包含 `observed_frame_ratio`、JSD/Jensen–Shannon 边界实现的候选源码。
- 现有冻结 Eval300 manifest 只能证明 Eval300 题目/视频 population；现有 `shared_dense_r3_eval300_index` 是 R3/Dense-H 资产，不能证明 R1 的构建统计。

### 无法确认的公式/伪代码

下列正式量均为 **unknown**，不能从冻结资产恢复：

```text
stat_j[c] = UNKNOWN              # instance count / frame occurrence / ratio / other
p_j[c] = UNKNOWN                 # denominator and smoothing unknown
JSD(p_j, p_{j+1}) = UNKNOWN      # empty/zero handling and log base unknown
boundary(j, j+1) = UNKNOWN       # threshold/tie handling unknown
label_summary_j[c] = UNKNOWN     # relationship to boundary statistic unknown
```

因此不能声称同帧同类多个 detection 计一次或多次，也不能声称无检测 Medium 被赋零向量、均匀分布、跳过或合并。边界统计与标签摘要统计是否复用同一计数器/字段也不能确认。

## 二、Staged 的 Fine 选择、重选与去重

### 正式 Local / Planner-only 对应本地下游

状态：**source missing / identity unconfirmed**。`experiment_registry.csv` 第 12–13 行分别把 Planner-only API 和 Staged API 标为 accepted formal source not found。缺少：运行 acceptance/freeze、配置、prompt、controller/core、下游实现及其 SHA/manifest 绑定。

所以正式结论只能是：

- Fine 候选集合及排序信号：未知；不能断言视觉与 lexical/关键词融合，也不能给权重。
- 再次选择已选 Medium/区域：未知；不能断言由“尚有未见 Fine”、excluded regions 或其他条件控制。
- 去重层级：未知；区域、Fine ID、时间戳/实际帧三者均未获得正式代码证明。
- prompt 与程序强制规则：未知；没有 prompt/core 对照，不能把自然语言要求写成 controller 保证。

### Full Staged paired100 正式最终版

状态：**未定位到任何 `paired100` 正式版本绑定记录**。缺少 final/accepted 说明、UID manifest、正式 config、冻结 prompt、controller/core snapshot 及运行产物之间的 lineage。已有盘点提到的是另一个缺失项 `R1/R3 paired-150`，不能用它替代或证明 paired100。

因此 Full Staged 的 Fine 评分、重选条件和去重规则也全部保持 unknown；不能从 Local 原型、Planner-only 输出、VideoSEAL Dense-H 或 Direct 推断。

### 找到但明确排除的相似代码

`runtime_v4/runtime_support/src/experiments/fine_reranking/core.py` 是名为 `fine_reranking_v1` 的独立辅助模块，不是已绑定的 Staged 正式实现：

- `validate_frozen_inputs` 固定要求 1 个视频、6 个问题以及 88/30/10/3 的节点规模（第 87–135 行），`build_frozen_manifest` 也明确写入 `experiment="fine_reranking_v1"` 和单一 video ID（第 138–164 行）。
- 该模块的 Fine 分数为 `s_f = e_f^T q`（第 174–210 行），实际按 raw SigLIP score 降序排序；其 min–max 值被记录，但不是排序融合项。
- 去重/多样性使用 Fine ID 与时间/区间条件（第 246–307 行）；第 638–639 行另行统计重复 frame path。它不证明 Staged 使用同样规则。
- `planner_medium_retrieval/core.py` 第 21–44 行出现 `0.60 visual + 0.30 lexical + 0.10 coarse prior` 配置，但这是 **Medium retrieval** 的独立模块配置，不是上述 Fine 排序公式，更不能当作 Staged paired100 的正式权重。

这些内容仅解释为什么本机可能出现“视觉/lexical/Fine”相关搜索命中；本报告不把它们写入 Staged Methods。

## 三、Staged 高层控制流程

对建议流程逐项核实如下：

| 待核实项 | 正式结论 | 缺失证据 |
|---|---|---|
| 第一次 Fine 前是否已有 Shared 判断 | 未确认 | 冻结 controller、Shared prompt/response schema、正式 trajectory |
| 不调用 Fine 直接 Final 的条件 | 未确认 | controller 分支与正式配置 |
| 循环和终止由谁控制 | 未确认 | state machine/core、step/budget 配置 |
| 格式校验重试是否与新增视觉检查分开 | 未确认 | parser/validator/retry 与 inspection 调度代码 |
| Local 与 Full 的系统图差异 | 未确认 | 两者各自的 accepted manifest 与版本绑定源码 |

不能把 prompt 中可能存在的“先判断、必要时检查、不要重复区域”等文字自动解释成程序强制。必须分别找到：prompt 要求；controller 的状态变量与分支；candidate/exclusion 集合；实际 frame/Fine 去重键；格式 repair 计数器；视觉检查计数器。

## 来源与版本关联依据

1. `/home/naxucl/projects/thesis-av-evidence/thesis_appendix_materials_dgx_v1/README.md` 第 3–5 行：资料收集主机为 `dgx01`，学校端材料“稍后另行合并”；第 11–13 行限定“本机未定位”的含义和已检查根。
2. `/home/naxucl/projects/thesis-av-evidence/thesis_appendix_materials_dgx_v1/APPENDIX_MATERIALS_INVENTORY.md` 第 58–64 行：正式 Planner-only API、Staged API、Direct R1/R3、R1/R3 paired-150 等仍待学校端包对接，不能凭相似名称认定。
3. `/home/naxucl/projects/thesis-av-evidence/thesis_data_inventory_20260914T014645Z/DATA_INVENTORY.md` 第 49、51–55 行：R1 build ledger、R1/Planner-only/Staged 正式包及历史 R1 builder 线索未找到。
4. `/home/naxucl/projects/thesis-av-evidence/thesis_data_inventory_20260914T014645Z/experiment_registry.csv` 第 11–14、22 行：相关记录状态均为 `source_missing_identity_unconfirmed`。
5. `/home/naxucl/data/HourVideo/experiments/hourvideo_v7_4_variant_c_budgets_v1/runtime_v4/MANIFEST.json`：现有冻结 runtime 属于 Variant-C/VideoSEAL lineage；不能作为 R1 或 Staged 的版本绑定依据。
6. 当前 `/home/naxucl/projects/VideoSEAL` checkout 有未提交修改，且其正式资料 lineage 只覆盖 VideoSEAL 条件；本报告没有用该 checkout 推断 R1/Staged。

## 可直接用于 Methods 的英文说明

在学校端冻结包恢复前，建议只使用以下保守说明，不填入未经证实的算法细节：

> Implementation-specific details of the R1 boundary statistic and the Staged fine-inspection controller were intended to be reported from the accepted school-side frozen snapshots. Those snapshots were not present in the DGX archive inspected for this revision. We therefore do not infer the category-distribution estimator, empty-distribution handling, Fine-ranking weights, reselection or deduplication rules, or controller state transitions from the separate VideoSEAL Dense-H, Direct, or auxiliary fine-reranking implementations.

这段文字是档案限制声明，不是对正式方法的替代描述。正式 Methods 仍需从学校端包补入实际公式和控制规则。

## 尚不能确认及所需最小补件

- **R1**：正式 builder 源码及 commit/SHA、构建 config、Eval300 R1 manifest/index metadata；至少应能定位类别聚合函数、归一化/JSD 函数、空分布分支和标签摘要函数。
- **Local Staged / Planner-only 下游**：accepted config/manifest、冻结 prompt、controller/core、Fine candidate/ranking 与 exclusion/dedup 实现。
- **Full Staged paired100**：final acceptance/freeze、paired100 UID manifest、正式 config、冻结 controller/core/prompt、版本 lineage；若有 superseded 版本还需明确最终采用关系。
- **流程核实**：Shared 判断记录或 schema、视觉检查调度、format-repair 计数与 inspection 计数、终止/预算配置。

在这些文件缺失时，三组问题均只能报告为“未确认”，不能根据相似代码、目录名、prompt 文案或其他方法的规则猜测补齐。
