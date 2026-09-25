# R3 Full Staged vs R3 Direct paired100：论文结果资料

## 结论摘要

本资料包分析固定的 paired100（100 道题，来自 12 个视频），所有主要准确率均以固定 100 题为分母，失败不删除。R3 Full Staged 在 100 题中答对 **24 题（24%）**，产生 97 个预测；同题 R3 Direct 答对 **32 题（32%）**，产生 99 个预测。两者相差 8 个百分点，但题目级双侧 exact McNemar 检验为 **p=0.1686375439**；该结果在常用 0.05 阈值下不显著，也不能解释为两种方法等价。

## 实验与冻结身份

- Full Staged：复用冻结的 R3 Planner 输出，新增调用 Shared、Fine、Final 三个 Haiku 阶段。
- Direct：冻结的 Direct-v1.2 3/16 R3 路线，同一批 100 题，不重跑。
- paired100 manifest SHA-256：`963c3c30dfbe73d7602a54558fa866cc0f74ddad35e12ee7c3fb268f78cacf8b`
- Full Staged runtime fingerprint：`06dc90fb8652a8a8cd392bf654040c64446fca246a1bc800d2b45a53a40820e2`
- 原始正式结果闭包 SHA-256：`fec945bfcf788b4cdf3b709794e6797e2a5bf974c83005cb86ef58b2bcdc3a17`
- 无 gold 结构核验：PASS；随后才由独立统计阶段读取 gold。
- gold 源 SHA-256：`e1af087df035d524ee64d92d34d2e81f29461fdeb5fa72cfda679d7c9909daf3`
- 统计范围：fixed denominator = 100；Full Staged 的 3 个失败与 Direct 的 1 个历史失败均按错误计入。

## 主结果与资源

| 口径 | 正确/100 | 预测/成功 | 逻辑调用 | 物理请求 | Validation retry | Transport retry | 图片传输 | 每题去重图总和/均值 | API 费用 |
|---|---|---|---|---|---|---|---|---|---|
| Staged 新增下游 | 24/100 | 97/100 | 478 | 662 | 184 | 0 | 1,424（含验证重传） | 1,312 / 13.12 | $11.195644600 |
| 历史 R3 Planner | N/A | 100/100 stage outputs | 100 | 100 | N/A（总 retry=0） | N/A（总 retry=0） | 0 原图 | 0 / 0 | $3.299245000（记录估算） |
| 重建完整 Staged | 24/100 | 97/100 | 578 | 762 | 184 + Planner subtype N/A | 0 + Planner subtype N/A | 1,424 | 1,312 / 13.12 | $14.494889600 |
| R3 Direct | 32/100 | 99/100 | 487 | 487 | N/A（结构纠错=0） | 0 | 928 次新图传输 | 928 / 9.28 | $4.301863350 |

Full Staged 新增下游费用比 Direct 高 **$6.893781250（160.251051%）**；把历史 Planner 记录估算加入后，重建完整 Full Staged 费用比 Direct 高 **$10.193026250（236.944445%）**。百分比均以未四舍五入数值计算。

### Token 明细

| 口径 | Ordinary input | 历史未拆分 input | Cache creation | Cache read | Output |
|---|---|---|---|---|---|
| Staged 新增下游 | 7,621,330 | N/A | 382,270 | 209,771 | 615,100 |
| 历史 R3 Planner | N/A | 2,853,515 | N/A | N/A | 89,146 |
| 重建完整 Staged | N/A（Planner 未拆分） | 2,853,515 | 382,270（仅下游已知） | 209,771（仅下游已知） | 704,246 |
| R3 Direct | 2,118,602 | N/A | 367,819 | 14,824,626 | 48,205 |

历史 Planner 的已选 100 题日志仅保留总 input/output 与 usage-based cost estimate，没有可用于本资料包的 ordinary/cache creation/cache read 分拆。因此重建完整流程的这些 token 类不能伪装成精确总数；表中只列下游已知部分并将未知项标为 N/A。

### 延迟

单位均为秒；mean、median、P90 均按全部 100 题计算，P90 使用排序后 `(n-1)*0.9` 的线性插值。

| 口径 | API latency sum | mean | median | P90 | route wall sum | mean | median | P90 |
|---|---|---|---|---|---|---|---|---|
| Staged 新增下游（实测） | 6,432.660 | 64.327 | 40.771 | 135.720 | 6,511.081 | 65.111 | 41.127 | 136.903 |
| 历史 R3 Planner | 1,725.258 | 17.253 | 17.682 | 20.186 | N/A | N/A | N/A | N/A |
| 重建完整 Staged（跨运行相加） | 8,157.918 | 81.579 | 58.788 | 153.403 | 8,236.339 | 82.363 | 59.145 | 154.551 |
| R3 Direct（历史实测） | 1,043.289 | 10.433 | 9.181 | 15.506 | 1,116.593 | 11.166 | 9.796 | 16.852 |

历史 Planner 与本次新增下游是在不同时间分别运行。表中的“重建完整 Staged”延迟是逐题历史 Planner API latency 与本次下游 latency/wall time 的算术相加，**不是一次实测端到端 wall time**。本次下游正式运行的观测整体 wall-clock 为 6,602.582 秒；它不与历史 Planner wall time混称。

## 配对统计

| 配对结果 | 题数 |
|---|---|
| 两者都正确 | 15 |
| 两者都错误（含无预测失败） | 59 |
| 仅 Direct 正确 | 17 |
| 仅 Full Staged 正确 | 9 |

双侧 exact McNemar 使用 26 个 discordant pairs（Direct-only 17，Staged-only 9），在零假设下计算 `X ~ Binomial(26, 0.5)`，p 值为 `2 * P(X <= 9) = 0.1686375439`。100 题来自 12 个视频，同一视频中的题目可能相关，因此题目级 McNemar 的独立性假设并不完全成立；该检验应视为描述性/探索性结果，不能据此追加选题或夸大统计结论。

## Final validation attempts 矛盾核对

冻结配置字段是 `max_validation_retries=2`。实际冻结 V6.6.2 代码 `_max_attempts` 返回 `1 + retries`，Final 循环执行 `range(1, attempts_allowed + 1)`；因此真实规则是：**初次 validation attempt 后最多重试 2 次，共最多 3 次 validation attempts**。`legacy_bridge.py` 将 legacy attempt 1/2/3 映射为 provider telemetry 的 `validation_retry_index` 0/1/2。

旧 aligned_v2 `PROTOCOL_REVIEW.md` 中的“Final allows 2 under the actual legacy loops”是文档措辞错误。它与实际冻结代码、paired100 manifest 中的配置以及正式 attempt 日志均矛盾。正式运行没有偏离其冻结代码：两条 Final exhaustion 路线均真实记录了 attempt 1、2、3，三次都因 `answer_text` 与所选 option 文本不一致而失败。新报告纠正文档表述，但没有覆盖旧文件。

冻结规则还允许每个 validation attempt 最多 1 次 transport retry，因此理论上每个逻辑阶段调用最多 3×2=6 个物理请求；本次正式 paired100 的 transport retry 为 0。Final 共 100 个逻辑调用、118 个物理请求，其中 18 个是 validation retry 请求。

## 失败与验证层级

Full Staged 有 3 条正式失败：

- `4572b198-2c1c-4920-bcf0-95fcebe12261_4_3`：RuntimeError: Direct Final contract exhausted after 3 attempts; errors=['Direct Final answer_text does not match selected option', 'Direct Final answer_text does not match selected option', 'Direct Final answer_text does not match selected option']
- `db3f7933-dfa0-4678-9d4f-393b628ded45_11_2`：RuntimeError: Direct Final contract exhausted after 3 attempts; errors=['Direct Final answer_text does not match selected option', 'Direct Final answer_text does not match selected option', 'Direct Final answer_text does not match selected option']
- `819c8af7-851f-434f-ab32-318285bc54b1_7_17`：KeyError: 'uncertainty'

- 两条 `Direct Final contract exhausted after 3 attempts` 属于旧 V6.6.2 **科学语义验证失败**；三次 provider tool envelope 均已通过，但 `answer_text` 与 `selected_option_id` 对应的原选项文本不一致。
- `KeyError: 'uncertainty'` 路线的 Final provider envelope 与 `_validate_direct_final_result` 均记录为成功，随后在构造最终答案时访问缺失字段而触发运行错误。这是本次冻结实现中暴露的 post-validation 字段覆盖缺口，不是拒答、预算停止或 SSH 中断。
- 全部 662 个物理请求中，provider tool envelope 层 661 accepted、1 rejected。唯一 envelope rejection 是 Fine 阶段“exactly one tool_use is required”，随后按冻结 validation retry 恢复，不是终态失败。
- V6.6.2 科学 attempt 日志共有 445 success、217 validation_failed；其中 184 个失败触发了下一次 validation 请求，其余为最后一次失败后进入既定降级或终止。正式结果没有 transport retry、未知 provider outcome、预算停止或 interrupted-active 路线。
- Direct 的唯一历史失败为 `runtime_failure:turn_limit_exhausted`，达到 32 turn 上限；并非 schema/语义验证失败或中断。

## 图片口径

- Full Staged 的“每批最多 16 图”是 **Fine 单次批处理上限**。每个 claim 最多 3 个调查轮，且 validation retry 会重传该批图片；它不是每题全局 16 张唯一图上限。因此本次共 1,424 次图片传输，最终每题去重后的 reviewed Fine 图总和为 1,312（均值 13.12）。
- Direct 的 16 是 **每题全局唯一图片上限**，每次 inspect 最多 3 张新图。本批 100 题新传输的唯一图总和为 928（均值 9.28）。Direct 多轮请求中历史图像块随会话再次序列化的总次数没有作为独立指标保存在冻结 route artifact 中，因此不把 928 误称为所有 HTTP payload 内 image blocks 的累计次数。

## 方法差异与解释限制

Full Staged 与 Direct 不是仅更换提示词的同构系统。Full Staged 复用 per-option Planner，经过确定性 Coarse→Fine 检索，并由 Shared/Fine/Final 阶段及各阶段 validation retry 组成；Direct 是一个读取完整 R3 map、自己选择检查时间点的单 agent，多轮对话共享每题 16 张唯一图预算。因而准确率、成本、图片数和完成率差异同时包含推理拓扑、检索责任、证据预算语义及重试协议差异。

paired100 是从 Eval300 固定抽取的 100 题而非完整 Eval300。样本覆盖 12 个视频，但视频内题目相关；本结果不外推为全体 300 题的确定结论。失败按固定分母保留，未做失败剔除、补跑或事后择优。

## 指标定义

- **逻辑调用**：Full Staged 中一次 Shared/Fine/Final 科学阶段调用；Direct 中一次模型决策 turn。
- **物理请求**：实际发生并记账的 provider 请求，包括 validation/transport retry。
- **Validation retry**：同一科学阶段因工具外壳或 V6.6.2 语义验证失败而追加的 provider 请求；不等同于新的科学调查轮。
- **图片传输**：Full Staged 为 Fine provider attempts 实际携带的图片数，验证重试重传会重复计数；Direct 表中为 controller 记录的新唯一原图传输。
- **完成率**：产生合法 A–E 最终预测的题数除以固定 100。
- **API latency**：各题已记录 provider latency 的总和；**route wall**：该路线的实测墙钟时间。跨历史运行相加一律标为 reconstructed。

## 源文件追溯

以下绝对路径只用于学校机器上的审计追溯；理解本资料包的表格和结论不依赖这些路径：

- `${DIRECT_WORKSPACE}/outputs/full_staged_api_preflight/full_staged_api_r3_vs_direct_r3_paired100_final_v2/paired100_manifest.json`  
  SHA-256: `963c3c30dfbe73d7602a54558fa866cc0f74ddad35e12ee7c3fb268f78cacf8b`
- `${DIRECT_WORKSPACE}/outputs/full_staged_api_analysis/full_staged_api_r3_vs_direct_r3_paired100_final_v2/structural_validation.json`  
  SHA-256: `f44faa76d0b0c43085820c22b88cb00d3e3098d35784571bc93b59bb5e40ccab`
- `${DIRECT_WORKSPACE}/outputs/full_staged_api_preflight/full_staged_api_r3_vs_direct_r3_paired100_final_v2/planner_cost_summary.json`  
  SHA-256: `1cb81dcf08f30b0d79b7538ea482cc9b72b13cca6987533d2bac09a9efb6897e`
- `${DIRECT_WORKSPACE}/outputs/full_staged_api_preflight/full_staged_api_r3_vs_direct_r3_paired100_final_v2/legacy_downstream_config.json`  
  SHA-256: `e71b5e6064d91fde950429bdf6bc91c2dbcbaad48fe554c9954cb34680f766bc`
- `${DIRECT_WORKSPACE}/outputs/full_staged_api_preflight/full_staged_api_r3_eval300_v1_aligned_v2/PROTOCOL_REVIEW.md`  
  SHA-256: `bd77a8a2a1db17b766b70219dac3ca3b430b38428fbf47405ad31196a50b7f0b`
- `${DIRECT_WORKSPACE}/src/staged_api_v1/config.py`  
  SHA-256: `0e52444d4b6d142af5ecbefc8a66cfecfd1a54c127974ffa83a6600e0644a24a`
- `${DIRECT_WORKSPACE}/src/staged_api_v1/legacy_bridge.py`  
  SHA-256: `fc70bb9cd37e63d8b6608c0563e00d5dcaa5ac9192ad0d91f35860667d73000d`
- `${DIRECT_WORKSPACE}/src/staged_api_v1/provider.py`  
  SHA-256: `7d0af2285d412d19a860d6be6b9ae9075e1110d498e0fd4ea078aa8c782883c5`
- `${LEGACY_RUNTIME}/src/experiments/hourvideo_v6_6_1_contract_telemetry_v1/core.py`  
  SHA-256: `01e0810d46fa041368c4eda0b704c38e8321e954bc0c9f46930fa209391dedf7`
- `${DIRECT_WORKSPACE}/outputs/direct_v1_formal/direct_v1_2_3x16_r1_r3_eval300_formal_v1/canonical_summary_v1/structural_validation.json`  
  SHA-256: `d1cc7c08ec701f22612e81866a125fa553be09f66e004b10fe0bdd2505674fc8`
- `${HOURVIDEO_ROOT}/benchmark/v1.0_release/json/dev_v1.0_annotations.json`  
  SHA-256: `e1af087df035d524ee64d92d34d2e81f29461fdeb5fa72cfda679d7c9909daf3`

## 完整性声明

生成资料包前后均重新计算 `structural_validation.json` 所列原始文件的 SHA/大小闭包，结果保持 `fec945bfcf788b4cdf3b709794e6797e2a5bf974c83005cb86ef58b2bcdc3a17`。本资料包不含 API key、认证头、环境变量秘密、provider base64 图片或原始响应正文；只包含派生统计和追溯 SHA。本次资料整理 API 调用数为 0，模型调用数为 0。
