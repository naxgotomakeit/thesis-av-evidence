# HourVideo Eval300：Flat-30、Dense Semantic H-8、H-15 与 H-30 数据档案

## 1. 档案状态

本文档记录冻结 Eval300 上正式 Flat-30 与新版 Dense Semantic Beam-B H-15 的最终数据，用于后续报告、论文制表和结果追溯。

- 数据集：HourVideo Eval300
- UID 数量：300 个唯一 UID
- 视频数量：12
- UID manifest SHA-256：`6db7b3cc672919a60c4b0c4a84f9f2e039e1673bdeddcfe619fdce92f1cb6fa1`
- Flat：正式 first pass 与独立 retry 的权威 merged 结果
- H-15：Dense Coarse Top-15 → Dense Medium Top-15 → 局部 Fine SigLIP → Flat-compatible segment Top-15
- Planner：原生 Qwen3-8B
- Inspector：Qwen2.5-VL-7B-Instruct
- 正式并发：`concurrency=1`
- H-15 backend：`dense_semantic_beam_b`
- H-15 runtime adapter SHA-256：`008efa173913759a7713209b649ab3947e99349a2385afd017995778460fc4b0`
- H-15与Flat完整tools schema SHA-256：`71346effd5acd8d567e16921316e0f6cb0e94b422f42c4d55edb24a86d4a1863`

## 2. 核心结果

| 指标 | Flat-30 | Dense H-15 | H-15 − Flat |
|---|---:|---:|---:|
| Strict completed / 300 | 254（84.67%） | 270（90.00%） | +16题；+5.33个百分点 |
| Correct / 300 | 82（27.33%） | 81（27.00%） | −1题；−0.33个百分点 |
| Correct / strict completed | 32.28% | 30.00% | −2.28个百分点 |
| Timeout / 300 | 36（12.00%） | 26（8.67%） | −10题；−3.33个百分点 |
| `status=success`但strict invalid | 10（3.33%） | 4（1.33%） | −6题 |
| 最终未strict completed | 46（15.33%） | 30（10.00%） | −16题 |

### 指标含义

**Strict completed**：最终选中 attempt 同时满足：metric `status=success`、prediction 是单个合法选项字母 A–E、trajectory 已结束且具有非空答案。它比简单的 `status=success` 更严格。

**Correct / 300**：以全部300题为分母；未完成题按未答对处理。这是两种方法总体准确率的主要口径。

**Correct / strict completed**：只在严格完成题中计算答对比例。该值会受到完成题集合差异影响，不能替代 `correct/300`。

**Timeout**：最终选中 attempt 的metric状态为timeout。这里使用最终terminal attempt，不把first pass和retry的timeout重复相加。

**Success但strict invalid**：runner报告success，但prediction或完整trajectory不满足strict completed条件。这些题不能算作正式完成。

## 3. 延迟效率

E2E分布只统计各方法最终strict completed的题：Flat 254题，H-15 270题。

| E2E指标 | Flat-30 | Dense H-15 | 相对变化 |
|---|---:|---:|---:|
| Mean | 318.99秒 | 250.27秒 | H-15降低21.5% |
| Median | 252.53秒 | 219.25秒 | H-15降低13.2% |
| P90 | 631.40秒 | 437.66秒 | H-15降低30.7% |
| P95 | 810.00秒 | 540.14秒 | H-15降低33.3% |

### 含义

- Mean反映平均每个严格完成题的端到端耗时，容易受长题和接近timeout的题影响。
- Median反映典型题耗时；H-15的典型完成题约快33秒。
- P90/P95反映长尾运行成本。H-15在长尾处的降幅大于中位数，说明主要效率收益之一是减少极慢轨迹。
- 延迟来自各题最终选中attempt的`metric.elapsed_sec`，不把被替换的旧attempt耗时加入“方法本身的最终效率”。

## 4. 共同完成题的严格配对延迟

Flat和H-15双方均strict completed的UID共232题。

| 配对指标 | Flat-30 | Dense H-15 |
|---|---:|---:|
| 配对题mean | 317.35秒 | 247.71秒 |
| 配对题median | 246.93秒 | 217.71秒 |

逐UID计算`H-15 − Flat`：

- 平均差：`−69.63秒/题`
- 差值中位数：`−24.27秒/题`

### 含义

配对比较只使用相同UID且双方都完成的题，因此不会因为两种方法完成了不同难度的题而直接混淆延迟。负值表示H-15更快。该结果说明，在可直接比较的232题上，H-15平均每题少用约70秒。

## 5. 调用量与视觉输入

下表统计300个UID各自最终terminal attempt：若UID执行过正式retry，则由retry替换对应first pass；新旧attempt不重复相加。

| 指标 | Flat-30 | Dense H-15 | H-15相对变化 |
|---|---:|---:|---:|
| Planner调用 | 1,300 | 1,154 | −11.2% |
| Retrieval调用 | 566 | 486 | −14.1% |
| Inspector调用 | 354 | 302 | −14.7% |
| Inspector实际发送图片 | 21,492 | 18,336 | −14.7% |
| Final-selected attempt E2E总量 | 32.62小时 | 26.10小时 | −20.0% |

### 含义与边界

- **Planner调用**：所有最终terminal trajectory中的Planner请求次数，包括最终回答前的搜索、检查及合法protocol恢复步骤。
- **Retrieval调用**：Planner调用`visual_retrieve`的次数。一次H-15 retrieval内部包含一次query text embedding、Dense Coarse/Medium路由和一次本地SigLIP query encoding；不能把一次retrieval理解为只做一次简单相似度计算。
- **Inspector调用**：Planner调用`visual_inspect`的次数。
- **Inspector实际发送图片**：metric中的`sent_images`总和，只表示发送给Inspector的图片。它不包括H-15在本地Fine排序中评分的canonical 1fps帧，也不包括以文本caption形式送入Summarizer的segment证据。
- **Final-selected attempt E2E总量**：用于描述最终方法表现；被retry替换的first-pass成本不在这一列重复计算。

H-15不仅单次完成题延迟更低，而且最终轨迹中的Planner、retrieval、Inspector和Inspector图片数量也更少。因此观察到的E2E下降不是通过增加更多下游调用换来的。

## 6. 实际实验总成本

实际实验成本保留first pass及所有正式retry，因为这些计算资源确实已经消耗。

| 指标 | Flat-30 | Dense H-15 | H-15相对变化 |
|---|---:|---:|---:|
| First pass题数 | 300 | 300 | 相同 |
| 正式retry题数 | 72 | 51 | −21次attempt |
| 实际attempt总数 | 372 | 351 | −5.6% |
| 全部attempt `elapsed_sec`总和 | 51.41小时 | 38.60小时 | −24.9% |

H-15正式运行墙钟时间：

- 开始：`2026-08-27T22:46:48Z`
- 结束：`2026-08-29T14:39:10Z`
- 包含模型服务启动、first pass、retry、finalize和少量阶段开销，约39小时52分钟。

### 方法效率与实验成本的区别

- 报告最终方法效率时，应使用最终selected attempt指标。
- 报告实际消耗资源时，应使用first pass＋retry全部attempt。
- 不能把旧attempt和替换后的retry同时加入最终方法E2E分布；也不能在报告实际算力消耗时丢弃旧attempt。

## 7. 准确率配对结果

全部300题中，未完成视为未答对：

| 配对结果 | UID数 |
|---|---:|
| 双方都正确 | 44 |
| Flat正确、H-15错误 | 38 |
| H-15正确、Flat错误 | 37 |
| 双方都未答对 | 181 |

### 含义

- 两个方向的discordance几乎完全对称：38对37。
- Flat总体只比H-15多答对1题，即0.33个百分点。
- 这一结果不支持声称H-15准确率优于Flat；更准确的表述是两者在Eval300上的总体准确率近似，而H-15完成率和效率更好。
- McNemar比较只依赖38和37两个discordant计数；两侧exact p值为1.0，没有准确率差异的统计证据。

## 8. 推荐报告表述

可以使用以下保守结论：

> 在冻结的HourVideo Eval300上，Dense Semantic Beam-B H-15取得81/300正确，Flat-30取得82/300正确，总体准确率相差1题。H-15的strict completion从84.67%提高到90.00%，timeout从36题降至26题。在双方共同完成的232题上，H-15平均每题比Flat快69.63秒；其最终selected-attempt总E2E降低20.0%，first pass加正式retry的实际总运行成本降低24.9%。因此，本实验支持H-15在基本保持总体准确率的同时改善完成率和运行效率，但不支持其准确率显著优于Flat。

## 9. 不应使用或混入的数据

- 不得使用H-15自动`final_report.json`内的`paired_with_flat`作为权威最终配对；该字段仍指向219-valid的旧Flat first-pass引用，而不是254-completed最终merged Flat。
- 不得把旧lexical-gated H-6/H-15或旧global Fine SigLIP H-30结果混入新版Dense Semantic Beam-B结果。
- 不得把smoke、readiness、fallback修复额外计算或被替换attempt加入最终方法效率。
- Parser/protocol事件应作为control-flow事件统计；被拒绝的plain-text答案不计为prediction，最终合法`<final>`答案才进入评分。

## 10. 权威来源与血缘

### Flat

- First pass：`data/HourVideo/videoseal_original/runs_dgx_eval300_v1`
- Retry：`data/HourVideo/videoseal_original/eval300_timeout_retry_20260819T084545Z/original_retry`
- Reconciliation：`data/HourVideo/videoseal_original/eval300_reconciliation_audit_20260822T144251Z/summary.json`
- Reconciliation SHA-256：`c08d896e565e4e589a11a3e92a9c0a22a0f2ef0306f6a4ac757eb58b16f6cec1`

### Dense H-15

- Formal root：`data/HourVideo/experiments/hourvideo_v7_4_variant_c_budgets_v1/outputs/dense_semantic_beam_b_h15_eval300_formal_v1_20260827T224330Z`
- Final report：`h15/status/final_report.json`
- Final report SHA-256：`1e0831758bb9e7736d086aa6e73fbb8c531f5742b25e515e59954efc247159d9`
- Prestart manifest SHA-256：`82bde90ca64298f9908b76f3f039968bbe40c9222f8249f324460c60e7a83413`

### 计算说明

本文档中的Flat/H最终配对和效率比较由现有artifact只读重建，没有调用模型、API或重新评分。H-15 raw结果、Flat结果及冻结manifest均未被修改。

## 11. Dense Semantic H-30正式结果

H-30使用与新版H-15相同的Dense Semantic Beam-B实现，唯一区别是层级beam及最终证据预算由15改为30：Dense Coarse global Top-30 → Dense Medium global Top-30 → 所选Medium内局部Fine SigLIP → Flat-compatible caption segment Top-30。以下结果来自正式H-30 first pass与一次冻结retry形成的merged视图。

| 指标 | Flat-30 | Dense H-15 | Dense H-30 |
|---|---:|---:|---:|
| Strict completed / 300 | 254（84.67%） | 270（90.00%） | 249（83.00%） |
| Correct / 300 | 82（27.33%） | 81（27.00%） | 78（26.00%） |
| Correct / strict completed | 32.28% | 30.00% | 31.33% |
| Timeout / 300 | 36（12.00%） | 26（8.67%） | 45（15.00%） |
| 最终selected E2E mean | 318.99秒 | 250.27秒 | 305.70秒 |
| 最终selected E2E median | 252.53秒 | 219.25秒 | 260.24秒 |
| 最终selected E2E P90 | 631.40秒 | 437.66秒 | 541.58秒 |
| 最终selected E2E P95 | 810.00秒 | 540.14秒 | 694.40秒 |

H-30 first pass严格完成234题；66题进入正式retry，其中15题恢复为strict completed，最终严格完成249题。修复后的方法效率采用每个UID最终选中的attempt，不将被retry替换的first-pass attempt重复加入E2E分布。

## 12. Flat-30与H-30严格UID配对

### 12.1 全部300题正确性

未完成统一视为未答对：

| 配对结果 | UID数 |
|---|---:|
| 双方都正确 | 43 |
| Flat正确、H-30错误 | 39 |
| H-30正确、Flat错误 | 35 |
| 双方都未答对 | 183 |

- 准确率差`H-30 − Flat`：`−1.33`个百分点。
- 固定seed UID bootstrap 95% CI：`[−7.00, +4.33]`个百分点。
- McNemar exact双侧检验：`p=0.7275`。

两个方向的discordance为39对35，没有证据表明Flat与H-30的总体准确率存在统计显著差异。这不证明两种方法等价，只表示Eval300没有检测到稳定的准确率差异。

### 12.2 Strict completion状态

| 配对结果 | UID数 |
|---|---:|
| 双方都完成 | 217 |
| Flat完成、H-30未完成 | 37 |
| H-30完成、Flat未完成 | 32 |
| 双方都未完成 | 14 |

- 完成率差`H-30 − Flat`：`−1.67`个百分点。
- 固定seed UID bootstrap 95% CI：`[−7.00, +3.67]`个百分点。
- McNemar exact双侧检验：`p=0.6305`。

H-30总体比Flat少完成5题，但配对结果没有显示显著完成率差异。

### 12.3 双方共同完成题延迟

双方共同strict completed的UID为217题：

| 配对指标 | Flat-30 | Dense H-30 |
|---|---:|---:|
| Mean | 308.44秒 | 287.83秒 |
| Median | 245.95秒 | 254.08秒 |
| P90 | 599.91秒 | 492.23秒 |
| P95 | 800.12秒 | 608.46秒 |

逐UID差值`H-30 − Flat`：

- 平均差：`−20.61秒/题`；固定seed paired bootstrap 95% CI为`[−52.17, +10.10]`秒。
- 差值中位数：`−6.30秒/题`。
- H-30更快115题，Flat更快102题，无相同耗时题。
- Wilcoxon signed-rank双侧检验：`p=0.4539`。

因此，在共同完成题上，H-30的平均延迟点估计低于Flat，但置信区间跨越0且Wilcoxon不显著，不能声称H-30相对Flat具有可靠的端到端提速。

## 13. H-15与H-30预算消融配对

### 13.1 全部300题正确性与完成率

| 指标 | 双方均为真 | H-15 only | H-30 only | 双方均为假 | McNemar exact p |
|---|---:|---:|---:|---:|---:|
| Correct | 43 | 38 | 35 | 184 | 0.8151 |
| Strict completed | 233 | 37 | 16 | 14 | 0.00549 |

- 准确率差`H-30 − H-15`：`−1.00`个百分点；固定seed bootstrap 95% CI为`[−6.67, +4.67]`个百分点。
- 完成率差`H-30 − H-15`：`−7.00`个百分点；固定seed bootstrap 95% CI为`[−11.67, −2.33]`个百分点。

准确率没有显著差异，但H-30的完成率显著低于H-15。增加最终证据预算没有在Eval300上产生可检测的准确率收益，却增加了未完成风险。

### 13.2 双方共同完成题延迟

H-15与H-30共同strict completed的UID为233题：

| 配对指标 | Dense H-15 | Dense H-30 |
|---|---:|---:|
| Mean | 242.21秒 | 297.86秒 |
| Median | 214.25秒 | 257.52秒 |
| P90 | 387.81秒 | 518.53秒 |
| P95 | 522.43秒 | 623.62秒 |

逐UID差值`H-30 − H-15`：

- 平均差：`+55.65秒/题`；固定seed paired bootstrap 95% CI为`[+31.29, +80.29]`秒。
- 差值中位数：`+30.49秒/题`。
- H-30更快82题，H-15更快151题。
- Wilcoxon signed-rank双侧检验：`p=7.10×10⁻⁶`。

该配对结果支持H-30显著慢于H-15。结合完成率结果，H-15是当前VideoSEAL下游中更好的预算折中；H-30的作用是说明同为30条最终segment时，完整层级Indexer仍可保持与Flat接近的总体准确率，而不是证明其端到端速度更快。

## 14. H-30结果的报告边界与权威来源

推荐表述：

> 在冻结的HourVideo Eval300上，Dense Semantic H-30取得78/300正确，Flat-30取得82/300正确。逐UID配对的McNemar exact检验未发现准确率差异（p=0.7275）。双方共同完成的217题上，H-30平均快20.61秒，但paired bootstrap区间跨0且Wilcoxon检验不显著，因此不能声称H-30相对Flat具有可靠端到端提速。相较H-15，H-30准确率没有显著变化，但完成率低7个百分点且共同完成题平均慢55.65秒，说明在当前VideoSEAL下游中，将预算从15增加至30没有带来可检测的准确率收益，并显著增加运行成本。

权威H-30来源：

- Formal root：`data/HourVideo/experiments/hourvideo_v7_4_variant_c_budgets_v1/outputs/dense_semantic_beam_b_h30_eval300_formal_v2_20260829T155639Z`
- Final report：`h30/status/final_report.json`
- Final merged per-UID manifest：`h30/merged/per_question_manifest.json`

本节配对计算使用完整300 UID：Flat来自最终254-completed reconciliation，H-15与H-30来自各自正式merged per-question manifest。Bootstrap固定seed为`20260831`，重复100,000次。所有统计均为现有artifact的只读计算，没有运行模型、API或重新评分。

## 15. Dense Semantic H-8正式结果

H-8使用与新版H-15/H-30相同的Dense Semantic Beam-B链路：Dense Coarse global Top-8 → 所选Coarse下Medium候选并集global Top-8 → 所选Medium内local Fine SigLIP → Flat-compatible caption segment Top-8。H-8取代未正式运行的H-6，属于预先登记的低预算正式配置；Inspector图片独立于B=8统计。

权威H-8来源：

- Formal root：`data/HourVideo/experiments/hourvideo_v7_4_variant_c_budgets_v1/outputs/dense_semantic_beam_b_h8_eval300_formal_v3_20260831T201510Z`
- Final report：`h8/status/final_report.json`
- Final merged per-UID manifest：`h8/merged/per_question_manifest.json`
- 正式运行：原生Qwen3-8B Planner，`concurrency=1`，backend=`dense_semantic_beam_b`。
- 旧H-8 v2的600条基础设施失败不属于本结果，未用于resume、retry或任何统计。

### 15.1 最终结果与指标含义

| 指标 | Flat-30 | Dense H-8 | Dense H-15 | Dense H-30 |
|---|---:|---:|---:|---:|
| Strict completed / 300 | 254（84.67%） | 270（90.00%） | 270（90.00%） | 249（83.00%） |
| Correct / 300 | 82（27.33%） | 78（26.00%） | 81（27.00%） | 78（26.00%） |
| Correct / strict completed | 32.28% | 28.89% | 30.00% | 31.33% |
| Timeout / 300 | 36（12.00%） | 29（9.67%） | 26（8.67%） | 45（15.00%） |
| `status=success`但strict invalid | 10（3.33%） | 1（0.33%） | 4（1.33%） | 6（2.00%） |
| 最终未strict completed | 46（15.33%） | 30（10.00%） | 30（10.00%） | 51（17.00%） |

指标定义：

- **Strict completed / 300**：最终选中attempt必须同时满足metric成功、prediction为单个合法选项A–E、trajectory正常结束且答案非空。分母固定为冻结的300 UID。
- **Correct / 300**：以全部300题为分母，未完成统一视为未答对。这是准确率比较的主要口径。
- **Correct / strict completed**：只描述已经严格完成的题，受各方法完成题集合影响，不能代替`correct/300`。
- **Timeout**与**strict invalid**均使用每个UID最终选中的terminal attempt；执行过retry的UID由retry替换first pass，不重复计数。

H-8 first pass严格完成254题，46题进入一次冻结正式retry；其中16题恢复为strict completed。最终270题完成、29题timeout、1题strict invalid。Launch-gate首题只保留了一次正式记录，没有重复计入300题。

### 15.2 最终selected-attempt延迟

E2E分布只统计各方法最终strict completed的UID：

| E2E指标 | Flat-30 | Dense H-8 | Dense H-15 | Dense H-30 |
|---|---:|---:|---:|---:|
| Mean | 318.99秒 | 247.03秒 | 250.27秒 | 305.70秒 |
| Median | 252.53秒 | 205.99秒 | 219.25秒 | 260.24秒 |
| P90 | 631.40秒 | 432.76秒 | 437.66秒 | 541.58秒 |
| P95 | 810.00秒 | 580.99秒 | 540.14秒 | 694.40秒 |

这些数字描述“最终方法输出的完成题延迟”，不包含已被retry替换的first-pass attempt。Mean受长尾影响；median表示典型完成题；P90/P95表示慢题尾部。H-8相对Flat的mean降低22.6%、median降低18.4%；H-8与H-15的总体E2E非常接近。

### 15.3 调用量、视觉输入与真实实验成本

下表使用300个UID的final-selected attempt：

| 指标 | Flat-30 | Dense H-8 | Dense H-15 | Dense H-30 | H-8相对Flat | H-8相对H-15 |
|---|---:|---:|---:|---:|---:|---:|
| Planner调用 | 1,300 | 1,251 | 1,154 | 1,307 | −3.8% | +8.4% |
| Retrieval调用 | 566 | 442 | 486 | 413 | −21.9% | −9.1% |
| Inspector调用 | 354 | 220 | 302 | 245 | −37.9% | −27.2% |
| Inspector实际发送图片 | 21,492 | 13,348 | 18,336 | 14,892 | −37.9% | −27.2% |
| Final-selected attempt E2E总量 | 32.62小时 | 26.19小时 | 26.10小时 | 33.45小时 | −19.7% | +0.3% |

- **Planner调用**包括搜索、视觉确认、最终回答和合法protocol recovery涉及的Planner请求。
- **Retrieval调用**是Planner实际调用`visual_retrieve`的次数。H-8每次正常retrieval返回不超过8个Flat-compatible caption segments；最终视图共442次retrieval，因此最多向Summarizer提供3,536条segment证据。
- **Inspector调用/图片**独立于retrieval B预算。13,348是实际送入Inspector的图片总数，不包括Summarizer使用的文本segment，也不包括local Fine阶段仅在本机评分的canonical 1fps帧。
- **Final-selected E2E总量**用于描述最终方法表现；它不能表示整次实验实际消耗，因为被retry替换的first-pass attempt也真实消耗了资源。

H-30列由其正式merged `per_question_manifest.json`的300条final-selected记录逐字段求和得到：Planner 1,307次、Retrieval 413次、Inspector 245次、Inspector图片14,892张，累计E2E为120,412.54秒（33.45小时）。该累计E2E包含每个UID最终选中的timeout/invalid attempt耗时，但不重复加入被retry替换的first-pass attempt；它不同于只在249个strict-completed UID上计算的305.70秒E2E mean。

实际运行成本（first pass加全部正式retry）：

| 指标 | Flat-30 | Dense H-8 | Dense H-15 |
|---|---:|---:|---:|
| First pass题数 | 300 | 300 | 300 |
| 正式retry题数 | 72 | 46 | 51 |
| 实际attempt总数 | 372 | 346 | 351 |
| 全部attempt `elapsed_sec`总和 | 51.41小时 | 37.62小时 | 38.60小时 |

因此H-8相对Flat的实际总attempt成本降低约26.8%，相对H-15降低约2.5%。虽然H-8减少了retrieval、Inspector和视觉图片量，但Planner调用比H-15多8.4%，抵消了部分低预算收益；不能仅依据B从15降到8，推断E2E也应按相同比例下降。

当前finalizer没有把Dense Retriever内部耗时写入统一的`retrieval_latency_sec`数组，因此本档案不报告H-8独立retrieval mean/P95。Coarse/Medium/Fine候选裁剪与本地SigLIP评分量应从逐次retrieval telemetry另行汇总，不能用E2E差值反推。

## 16. Flat-30与H-8严格UID配对

### 16.1 全部300题正确性

未完成统一视为未答对：

| 配对结果 | UID数 |
|---|---:|
| 双方都正确 | 41 |
| Flat正确、H-8错误 | 41 |
| H-8正确、Flat错误 | 37 |
| 双方都未答对 | 181 |

- 准确率差`H-8 − Flat`：`−1.33`个百分点。
- 固定seed UID bootstrap 95% CI：`[−7.00, +4.33]`个百分点。
- McNemar exact双侧检验：`p=0.7343`。

H-8比Flat少答对4题，但两个方向的discordance为41对37，没有检测到稳定的准确率差异。该结果不证明两种方法等价；它表示Eval300没有提供H-8准确率不同于Flat的显著证据。

### 16.2 Strict completion状态

| 配对结果 | UID数 |
|---|---:|
| 双方都完成 | 233 |
| Flat完成、H-8未完成 | 21 |
| H-8完成、Flat未完成 | 37 |
| 双方都未完成 | 9 |

- 完成率差`H-8 − Flat`：`+5.33`个百分点。
- 固定seed UID bootstrap 95% CI：`[+0.33, +10.33]`个百分点。
- McNemar exact双侧检验：`p=0.04794`。

H-8多完成16题。该结果在未作多重比较校正的单项双侧检验中刚达到0.05阈值，应表述为边界显著，而不是强效应或因果结论。

### 16.3 双方共同完成题延迟

Flat与H-8共同strict completed的UID为233题：

| 配对指标 | Flat-30 | Dense H-8 |
|---|---:|---:|
| Mean | 313.35秒 | 236.99秒 |
| Median | 245.95秒 | 197.95秒 |
| P90 | 610.13秒 | 397.73秒 |
| P95 | 811.25秒 | 579.48秒 |

逐UID差值`H-8 − Flat`：

- 平均差：`−76.36秒/题`；固定seed paired bootstrap 95% CI为`[−108.61, −44.52]`秒。
- 差值中位数：`−48.94秒/题`。
- H-8更快155题，Flat更快78题，无相同耗时题。
- Wilcoxon signed-rank双侧检验：`p=7.35×10⁻⁷`。

该配对结果支持H-8在双方均完成的相同题目上具有更低E2E延迟。这里比较的是完整VideoSEAL端到端流程，并非纯Retriever耗时。

## 17. H-8与H-15/H-30预算消融配对

### 17.1 正确性与完成率

| 比较/指标 | 双方均为真 | 较高预算only | H-8 only | 双方均为假 | 差值（H-8−较高预算） | McNemar exact p |
|---|---:|---:|---:|---:|---:|---:|
| H-15 vs H-8 Correct | 45 | 36 | 33 | 186 | −1.00个百分点 | 0.8099 |
| H-15 vs H-8 Strict completed | 250 | 20 | 20 | 10 | 0.00个百分点 | 1.0000 |
| H-30 vs H-8 Correct | 41 | 37 | 37 | 185 | 0.00个百分点 | 1.0000 |
| H-30 vs H-8 Strict completed | 232 | 17 | 38 | 13 | +7.00个百分点 | 0.00646 |

固定seed bootstrap 95% CI：

- H-8与H-15正确率差：`[−6.33, +4.33]`个百分点；完成率差：`[−4.00, +4.00]`个百分点。
- H-8与H-30正确率差：`[−5.67, +5.67]`个百分点；完成率差：`[+2.33, +11.67]`个百分点。

H-8与H-15在准确率和完成率上均未检测到差异。H-8与H-30答对数完全相同，但H-8完成率显著更高。结合B的定义，这支持“增加到30条最终segment没有在当前Eval300/VideoSEAL下游带来可检测准确率收益”，但不能脱离该下游推广为Retriever本身的普遍结论。

### 17.2 共同完成题延迟

| 配对比较 | 配对数 | 较高预算mean | H-8 mean | H-8差值mean | 差值median | Paired bootstrap 95% CI | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|---:|
| H-15 vs H-8 | 250 | 247.77秒 | 239.77秒 | −8.00秒 | −12.64秒 | [−31.88, +15.95]秒 | 0.2342 |
| H-30 vs H-8 | 232 | 300.18秒 | 243.47秒 | −56.71秒 | −39.04秒 | [−84.70, −28.67]秒 | 2.46×10⁻⁶ |

H-8相对H-15的延迟差异不显著，因此不能声称H-8可靠快于H-15。H-8相对H-30则表现出明确的配对E2E优势。

## 18. H-8正式报告结论与边界

推荐表述：

> 在冻结的HourVideo Eval300上，Dense Semantic Beam-B H-8取得78/300正确，Flat-30取得82/300正确；逐UID McNemar exact检验未检测到准确率差异（p=0.7343）。H-8严格完成270/300题，高于Flat的254/300题；在双方共同完成的233题上，H-8平均快76.36秒，paired bootstrap 95% CI为[−108.61, −44.52]秒。H-8与H-15具有相同完成数，准确率相差3题且配对检验不显著；H-8显著减少Inspector调用与图片量，但由于Planner调用增加，其整体E2E与H-15接近。H-8与H-30答对数相同，但完成率高7个百分点，并在共同完成题上平均快56.71秒。因此，H-8显示低证据预算可以在当前VideoSEAL下游中维持近似总体准确率并降低视觉处理负担；它不证明低预算Retriever在所有下游上普遍更优。

统计边界：

- 300题来自12个视频，不应解释为300个完全独立视频样本；本节UID级检验没有替代按视频cluster的稳健性分析。
- 多组预算比较产生多重检验问题；本文保留原始p值并明确效应量与置信区间，不以单个p值宣称普遍优越性。
- H-8 final report中的旧式`parser_failures`累计字段不是重建control-flow事件数，本节不据此作因果解释。
- `state.txt`在任务完成后仍保留`FORMAL_RUNNING`，但`finished_utc.txt`、46/46 retry覆盖、final report和300条merged manifest均已生成，服务及18082/18083端口已释放。这是收尾状态标签遗漏，不改变结果选择或统计。
- H-8目录中的`paired_with_flat`是旧finalizer兼容字段，本节未使用它；Flat比较来自最终254-completed reconciliation，并按完整300 UID重新配对。

本节配对计算使用完整300 UID：Flat来自最终reconciliation，H-8/H-15/H-30来自各自正式merged per-question manifest。Bootstrap固定seed为`20260831`，重复100,000次。所有统计均由已有artifact只读重建，没有运行模型、API、retry或重新评分。
