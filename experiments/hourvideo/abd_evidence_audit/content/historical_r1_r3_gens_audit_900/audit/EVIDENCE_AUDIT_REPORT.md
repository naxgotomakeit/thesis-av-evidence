# R1 Direct / R3 Direct / GenS v3 分批证据审计

本报告是 **Codex辅助证据审计**，不是人工复核，也没有调用独立评审模型。审阅阶段按中性 ID 分成192批，每批不超过10条（图片多时进一步缩小），覆盖全部900条路线。每条均以回答时实际可见的 map、已传输图片和工具反馈为证据；未发送的帧不用于判断。

## 冻结与完整性

- 冻结标准：`AUDIT_CRITERIA_FROZEN.md`
- 逐批记录：`reviews/B001.jsonl` 至 `reviews/B192.jsonl`
- 覆盖：900/900；重复0；缺失0；结构校验问题0
- 冻结逐题审计：`evidence_audit_frozen.jsonl`
- 冻结审计 SHA-256：`120fa48bba3fdba4dc53a71e160c4def946910e9e018764a9df777b32bc8c329`
- 分类先于评分关联完成；评分关联不修改分类。
- Direct 原始闭包当前重算 SHA 与正式记录相同：`a83e53138ff945efb96ba87b1e80cb9c8ed6dc35449898514b9ccab869c37d29`（1,507文件）。
- GenS v3 的604个原始闭包文件逐一重算均与冻结 inventory 相同；闭包 SHA 为 `e60ded47ee7c5feea51ede9798f9f0c79ab99083f20f1c1e2df2bf9ebbd4c863`。

审阅完整后有一次宽泛文件搜索在合并 SHA 写入前意外打印了少量既有逐题 gold 字段。此时192个批次文件已经全部写完，`progress.json` 已验证900/900；此后没有修改任何批次审计记录。`evidence_audit_freeze.json` 保存了全部192个批次文件的 SHA。该时序偏差必须作为审计限制保留，不能把本审计称为严格的全流程 blind review。

## 原正式结果与完成率

| 方法 | 正确/300 | 准确率 | 有最终答案 | 完成率 |
|---|---:|---:|---:|---:|
| R1 Direct | 88/300 | 29.33% | 300 | 100.00% |
| R3 Direct | 103/300 | 34.33% | 298 | 99.33% |
| GenS v3 | 93/300 | 31.00% | 300 | 100.00% |

R3 的2条失败均保持 `no_final_answer`，固定分母仍为300。本审计标签不改分、不剔除题目。

冻结审计后，`independent_gold_validation.json` 直接从正式 annotation 源重新计算三组正确性：R1=88、R3=103、GenS=93，与既有逐题评分的差异为0，缺失gold身份为0。

## 证据支持分类与正确性

单元格为“总数（正确/错误）”。

| 方法 | supported | partially_supported | unsupported | contradicted | unreviewable | no_final_answer |
|---|---:|---:|---:|---:|---:|---:|
| R1 | 67（24/43） | 133（48/85） | 69（11/58） | 28（4/24） | 3（1/2） | 0（0/0） |
| R3 | 78（35/43） | 127（43/84） | 48（11/37） | 42（13/29） | 3（1/2） | 2（0/2） |
| GenS v3 | 36（10/26） | 90（33/57） | 156（47/109） | 12（1/11） | 6（2/4） | 0（0/0） |

这里的 `supported` 是“实际输入中存在可区分候选项的明确证据”，并不意味着模型必然选中 gold；`partially_supported` 表示有相关线索但关键判断仍缺证据。

## 正确但证据不足，以及有支持但答错

本报告把“证据不足或被证据反驳”操作化为 `unsupported + contradicted`，不把 `partially_supported` 混入该指标。

| 方法 | 答对但 unsupported/contradicted | 占全部300 | 占该方法正确题 | supported 但答错 |
|---|---:|---:|---:|---:|
| R1 | 15 | 5.00% | 17.05% | 43 |
| R3 | 24 | 8.00% | 23.30% | 43 |
| GenS v3 | 48 | 16.00% | 51.61% | 26 |

因此，GenS v3 的93个正确答案中有48个在其实际输入证据下仍属无支持或被反驳；这说明强制选择带来的正确数中包含较多证据不足的命中。它不证明答案在现实视频中错误，只说明**模型当时实际收到的帧不足以支撑该选择**。同理，`supported but wrong` 可能来自选项歧义、证据并非唯一、审计误差或与 gold 的不一致，不能用审计标签替代正式评分。

## 理由依据

理由标签允许多选。

| 方法 | 与输入对应 | 合理但未证实推断 | 常识/措辞/猜测 | 无依据事实断言 | 与输入矛盾 | 输入无法核验 |
|---|---:|---:|---:|---:|---:|---:|
| R1 | 213 | 62 | 12 | 65 | 33 | 119 |
| R3 | 229 | 48 | 8 | 53 | 48 | 105 |
| GenS v3 | 136 | 52 | 29 | 58 | 14 | 166 |

这些标签描述理由文本与输入证据的对应关系，不证明模型内部真正采用了何种推理过程。

## Direct 的 map / 图片证据

客观的输入通道可用性为：

- R1：300条均同时有 map 和至少一张已观察图片。
- R3：298条有 map+图片；2条只有 map，且这2条没有最终答案。

冻结审计中的 `direct_evidence_source` 早期批次存在标签使用不一致，不能在看过评分后回写修正。按原样记录：

- R1：`map_and_images=121`、`images=60`、`none=13`、`not_applicable=106`。
- R3：`map_and_images=163`、`map=3`、`images=16`、`none=3`、`not_applicable=115`。

后两组中的 `not_applicable` 显然不适用于 Direct，故不能把上述数字当作完整可靠的因果依赖分布。客观通道可用性可信，但“答案究竟依赖 map 还是图片”的精确总量仍需针对这221条进行独立、预注册的 source-label 复核。本轮不在评分后修补该字段。

## GenS v2 → v3

- v2拒答91题：v3答对30、答错61。
- v2已回答209题：对→对52、对→错7、错→对11、错→错139。
- v2正确59，v3正确93，净增34。
- v3新增正确41题，同时丢失原正确7题，因此净增34。
- 41个新增正确答案的冻结证据标签：`supported=1`、`partially_supported=9`、`unsupported=29`、`unreviewable=2`。

新增正确中只有10题达到 supported/partially_supported，29题的实际输入不足以支持所选项，另2题无法审阅。因此准确率上升不能简单解释为证据利用改善。v2→v3同时改变了 prompt、接口和强制作答策略，绝不是纯 parser 效果。

## 固定案例选择

案例没有按是否正确或是否有利挑选。`fixed_examples.json` 对每个方法、每个支持类别，按 canonical Eval300 顺序固定取前2条，保存中性ID、question ID、预测、正确性、审计说明和证据引用。完整逐题记录见 `scored_evidence_audit.jsonl`。

典型现象包括：

- 有明确视觉/地图支持：工具身份、清晰空间关系或完整前后事件序列直接对应选项。
- 合理但未证实：更换工具会更快/更方便等反事实推断，输入只能证明原方法，不能证明反事实结果。
- 证据不足：单帧回答全程计数、持续时间、完整枚举或“从未出现”。
- 明确矛盾：理由自己的时长计算或空间描述与最终所选项方向相反。

## 结论

三组结果都同时包含有证据支持的回答和依赖未证实推断的回答。R1/R3 的 supported+partial 比例分别为200/300与205/300；GenS v3为126/300。GenS v3更多路线只有少量冻结帧，156/300被评为 unsupported，符合其输入覆盖受限的风险；但不能据此预设某方法“更真实”或改写正式准确率。

R3的正式准确率最高，但其 `contradicted` 数量也最高（42），且正确题中24题属于 unsupported/contradicted。R1的对应数量为15；GenS v3为48。由此能得出的稳健结论是：**准确率与输入证据充分性不是同一指标，三者都需要同时报告。**

仍有12条 `unreviewable`（R1 3、R3 3、GenS 6），另有190条标记 `needs_further_review`（R1 73、R3 58、GenS 59）。这些项目不应被猜测填补；如需更高置信度，应由独立审阅者按同一冻结证据包复核，而不是增加未发送视频内容。

## 输出文件

- `evidence_audit_frozen.jsonl`：冻结的900条无方法/无评分审计
- `evidence_audit_freeze.json`：冻结 SHA 与192个批次哈希
- `scored_evidence_audit.jsonl`：评分后身份关联逐题表
- `evidence_score_cross_stats.json`：三方法交叉统计
- `gens_v2_to_v3_transition.json`：v2→v3逐题转换
- `unified_summary.csv`：统一方法汇总
- `fixed_examples.json`：固定规则案例
- `raw_integrity.json`：原始闭包未变化核验
- `independent_gold_validation.json`：冻结后独立gold重算核验
- `scored_analysis_freeze.json`：派生统计文件 SHA

本轮新增 API 调用：0；新增模型调用：0；未重跑任何实验。
