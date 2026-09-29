# R1 Direct visual-only 证据支持审计更新

## 1. 结论

本次更新按旧 R1 最终表实际采用的 `codex_evidence_audit_r1_r3_gens_v3_v1` criteria、材料投影、图片顺序、批次上限和 schema 执行。新版 R1 Eval300 中，174 条有最终答案的补跑路线全部重新判断；1 条失败路线按旧定义记录为 `no_final_answer`；另 125 条经完整审计输入投影等价核验后复用旧冻结标签。

最终 300 条 UID 唯一且无遗漏。新版标签分布为：supported 58、partially supported 118、unsupported 86、contradicted 37、unreviewable 0、no final answer 1。该列可替换论文旧 R1 审计列。

## 2. 协议身份与差异

- 旧冻结标准：`AUDIT_CRITERIA_FROZEN.md`，SHA-256 `7e9e25909a97ade16b86c896e5b086f5148b7d3f1bda8a8118db8c07f318f4e3`。
- 旧最终 R1 标签来源：`evidence_audit_frozen.jsonl`，SHA-256 `120fa48bba3fdba4dc53a71e160c4def946910e9e018764a9df777b32bc8c329`。
- 本次完整执行指令：`REVIEW_EXECUTION_PROTOCOL.md`。当前可说明为 Codex、GPT-5 系列执行环境；精确 build、temperature、工具封装和旧运行对应值均没有可验证冻结记录，因此不声称运行环境完全相同。
- 本次仅处理 R1，保持 UID、gold/correctness、待重审旧标签和新旧答案变化隐藏；不声称实现旧 900 条 R1/R3/GenS 的跨方法混排盲化。
- `direct_evidence_source` 是审计依据来源的描述字段，不参与四类支持标签汇总。旧 R1 的 `not_applicable=106` 是 parser 与 criteria 的历史不一致；本次有答案记录仅使用 `map/images/map_and_images/none`，未改变支持标签定义或指标。
- 旧工作流没有 independent reviewer 或 adjudication；batch judgment 经 freezer 结构校验后即为最终记录。本次同样未叠加新复核流程。

## 3. 输入与执行

- 准备包 `SHA256SUMS.txt` 的 416 项全部通过；其清单 SHA-256 为 `ca374f35505e2c5800ed2a51ce64825de066c5241889155a4c3600e1f24c6494`。
- 新版论文数据包自身 4 项校验全部通过；其清单 SHA-256 为 `160dc3d5ebc1f472d24312e7fd0ee070c3e82833c752078f0a99b7f6503edd63`。
- 新审计只使用中性 review package、visual-only map、实际发送帧及其 requested/resolved time、SHA、工具反馈、问题/选项、最终答案和 action reason。
- 审计判断完成并形成 300 条冻结标签前，没有加载 gold/correctness，也没有读取待重审 175 条的旧标签。
- contact sheet 只用于批次导航；判断结合了原图展示、时间顺序、map 和工具反馈，不把模型理由本身当成证据。

## 4. 新版 R1 Eval300 审计列

| 标签 | 数量 | 占固定分母 300 | 其中回答正确 | 类内正确率 |
|---|---:|---:|---:|---:|
| Supported | 58 | 19.33% | 19 | 32.76% |
| Partially supported | 118 | 39.33% | 42 | 35.59% |
| Unsupported | 86 | 28.67% | 17 | 19.77% |
| Contradicted | 37 | 12.33% | 7 | 18.92% |
| Unreviewable | 0 | 0.00% | 0 | — |
| No final answer | 1 | 0.33% | 0 | 0.00% |
| **Total** | **300** | **100.00%** | **85** | **28.33%** |

`correctness` 只在标签文件完成结构校验并冻结后由新版论文数据包关联。支持标签评估的是所选答案是否得到模型实际证据输入支持，不等同于 gold correctness，因此类别正确率不应解释为标签定义。

## 5. 来源拆分

| 来源 | Supported | Partial | Unsupported | Contradicted | No final answer | 合计 |
|---|---:|---:|---:|---:|---:|---:|
| 新 visual-only 路线重审 | 28 | 74 | 47 | 25 | 1 | 175 |
| 已核验旧标签复用 | 30 | 44 | 39 | 12 | 0 | 125 |
| 合计 | 58 | 118 | 86 | 37 | 1 | 300 |

无最终答案路线为 `70f2a750-f403-41b8-aabb-480eb3ab4ed4_17_16`（审计中性 ID `E0001`）；没有伪造支持标签。

## 6. 产物与限制

- `r1_visual_only_evidence_labels_frozen.jsonl`：300 条最终标签，逐条保留新审计或旧复用来源。
- `r1_visual_only_evidence_labels_with_outcomes.csv`：冻结后关联的回答、correctness 与 route artifact provenance。
- `r1_visual_only_evidence_audit_summary.json`：结构计数及各标签正确数。
- `reviews/B001.jsonl`–`B054.jsonl`：175 条本次原始 batch judgment。
- `SOURCE_AND_PROTOCOL_PROVENANCE.md`：关键来源和差异说明。

限制：旧 Codex 精确 build/运行设置不可恢复；本次不具备跨方法混排盲化；历史逐帧全量 SHA 仍有既有证据缺口。本次没有修改旧标签、旧 parser、旧实验或 QA 资产，也没有调用模型/API。
