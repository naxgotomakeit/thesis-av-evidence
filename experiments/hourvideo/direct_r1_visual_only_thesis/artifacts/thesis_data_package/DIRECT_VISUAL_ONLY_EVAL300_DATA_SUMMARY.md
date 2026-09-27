# Direct Visual-Only Eval300 数据摘要

## 1. 结论与实验身份

本数据包依据 v13 最终冻结资产重新离线统计，未调用模型/API，也未重跑任何路线。v13 最终报告 SHA-256 已验证为 `6aaada06175b6945063ef90c3ddd47c134fc31317b9cb8b488d3cdb0b10818e6`；其 `FINAL_AUDIT_SHA256.txt` 中 19 个引用文件全部通过校验。

论文主结果的组成是：

- **R1 visual-only**：对原地图含非空音频派生内容的 175 条路线，使用 `coarse_regions[*].audio_channel=[]` 的地图从初始状态补跑；另外复用原地图本来没有非空 ASR、且已通过输入与协议复用核验的 125 条路线。
- **R3**：复用原冻结 Direct Eval300 的 visual-only 300 条路线。
- 旧的、被替换的 R1 175 条路线和全部 smoke 均未进入新 R1 Eval300。
- R1 的 175+125 恰好覆盖 300 个 question UID；R3 是同一组 300 个 UID。逐 UID 检查无重复、无遗漏，R1/R3 严格配对。

对 7 个受影响视频，递归地图差异白名单仅允许把 `coarse_regions[*].audio_channel` 从非空列表置为空列表；视觉/结构字段未改变。其余 5 个视频的新旧地图逐字节一致。保留的 `map_type=r1_av_structural_audio_dual_channel` 是静态 schema/type 标签，不是音频观察内容。详见 map transform manifest。

### 1.1 冻结身份

| 身份 | 路径 | SHA-256 |
|---|---|---|
| 原 Direct 正式 candidate v3 | `${PROJECT_MSC_ROOT}/msc_thesis/main_system/isolated_workspaces/hourvideo_direct_api_eval300_v1/outputs/direct_v1_formal/direct_v1_2_3x16_r1_r3_eval300_formal_v1/formal_manifest_final_candidate_no_api_v3.json` | `c73aeaf41d5bf284ade54198525d8396bf3a9fa5e45587aedd49dd3744dc557d` |
| 原 Direct launch lock v3 | 同一正式运行目录下 `formal_launch_lock_v3.json` | `ebd2014fcd952779444064510d39147149c174bc1ff2bc7fa231dfd589c09d5c` |
| R1 correction v8 正式 manifest | `.../direct_r1_visual_only_correction_v13_unified_recovery/runs/direct_r1_visual_only_correction_v8_formal_175/formal_manifest_visual_only_correction_v8.json` | `d556d817cc0ba6b49f0dfa7b6e4cf291387f9a5f55aa86ec02d7cb0b80517c61` |
| 最终 v13 recovery manifest | `.../runs/direct_r1_visual_only_correction_v13_unified_recovery_remaining_12/recovery_manifest_v13.json` | `629870ab6c17d7648becd868c8468d40f712466f05101d5833f2d1c975375d97` |
| 地图转换 manifest | `.../manifests/map_transform_manifest.json` | `9da0d49ee94a5c0685b7548bce41312c5a071b3fe1684f924757ffb549049993` |
| 175/125/R3 复用 manifest | `.../manifests/route_reuse_manifest_v8.json` | `bbefd954706b539fa4fa03b6b5d4b0b3c78813ca81e8b5b6071a790e7ba0f72a` |
| correction code-freeze manifest | `.../manifests/code_freeze_manifest.json` | `c4930eb61c09d0022f2c9f586c05d5464029aa26afe13d35e5c1ec5a6998274b` |
| 无 gold 结构合并 | `.../outputs/r1_visual_only_recovered_eval300_v1/merged_results_no_gold.json` | `57425a522b4bd0bde9de077d96c19149fa1965c1d04913bbb97fc96b82dd1da0` |
| 带 gold 的逐题冻结结果 | 同目录 `scored_results_with_gold.json` | `17550fb4f0bd244fd49c220728a6bf668402b31ead56979eaf23bd5d6fca2cd1` |

以上省略号均展开为 `${PROJECT_MSC_ROOT}/direct_r1_visual_only_correction_v13_unified_recovery`。

### 1.2 QA 协议和运行管理变化

冻结 QA 协议为：模型 `claude-haiku-4-5-20251001`、prompt version `direct_v1.2`、temperature 0、max output 512、timeout 120 秒、每轮最多 3 张新图、每题最多 16 张唯一图、最多 32 turns、1 次 provider transport retry、1 次 structural correction retry，以及相同的 system-prompt、action-schema、map-loader 和 final-answer parser 身份。

| 核心身份 | 原正式 / correction 冻结值 |
|---|---|
| provider config | `581f7bb119660096820011bea6c03970c6af3391f3a2a3bff428377ff77f4c8a` |
| system prompt text | `6e938f9fc09af7d3ce94d75ee7c6d1588b053aeb79f6b0855137b581e1b3a22d` |
| action schema | `4ab64f5a27dcd9a5b6431b161479f89eeb077c757d9ec5beebc40312c626396f` |
| map loader | `70d727f0bbc05ad2c8581c1c07796f76a0f43629886aaeca7b78836d573352a4` |
| final-answer parser/actions | `0ad965d5f0fe6f4278a864245a283d2ddeeb2532beec99bad727ef739ca63f24` |

这些共同哈希由原正式 candidate 与 v13 manifest 交叉确认。运行实现的差异如下：原正式 `controller/provider/frame_resolver/formal_runtime` 哈希分别为 `abeaf634… / dba18032… / 080ceecb… / 99074127…`；correction 对应为 `4dfd33d8… / cbef7056… / df7c6cb4… / 42ce6eac…`。完整值可在上述两个 manifest 的 `fingerprints` / `protocol_identity` 字段复核。

对补跑路线，**唯一预期的模型可见输入变化**是移除 R1 地图中的音频派生内容；没有向 prompt 加入 correction 标签或旧答案/旧对话。与此同时必须披露：

- 旧正式运行与 correction 在不同日期执行，无法证明服务端模型跨时间字节级一致。
- correction 强制使用历史 `urllib_fallback` transport，并增加了 writer lock、输入 closure、发送前帧 SHA 检查、预算预留/结算、pending 恢复与统一网络故障恢复记录。
- 原 candidate 与 correction 的 provider、controller、frame-resolver 和 formal-runtime 文件哈希并不相同；冻结测试证明正常 QA payload/参数不变，但这些运行管理和完整性实现不是字节级相同代码。
- 现有 frame manifest 是 correction 前冻结的当前快照；缺少原正式实验对所有逐帧文件的历史全量 SHA。因此它能证明补跑使用冻结快照及发送时校验，不能补充证明旧运行时每一张缓存帧的历史 SHA。

## 2. 论文主结果

以下数字由本数据包再次从 600 个 source route artifact 与正式 gold 逐题复算，和 v13 冻结结果一致。失败保留为错误，主分母固定为 300。

| 指标 | R1 visual-only | R3 visual-only |
|---|---:|---:|
| 题数 | 300 | 300 |
| 正确数 | 85 | 103 |
| 主准确率 | **28.33%** | **34.33%** |
| 合法最终答案 | 299 | 298 |
| 完成率 | **99.67%** | **99.33%** |
| 失败数 | 1 | 2 |
| completed-only accuracy | **85/299 = 28.43%** | **103/298 = 34.56%** |

失败原因：

- R1：1 条 `invalid_action:frame_resolution:requested timestamp outside video duration: 1680.0`。
- R3：1 条 `runtime_failure:turn_limit_exhausted`；1 条 `structural_action_correction_exhausted:frame_resolution:requested timestamp outside video duration: 1900.0`。

### 2.1 R1/R3 配对四格表

| R1 | R3 | 数量 |
|---|---|---:|
| 对 | 对 | 51 |
| 对 | 错 | 34 |
| 错 | 对 | 52 |
| 错 | 错 | 163 |

合计 300。R1 比 R3 少 18 条正确答案，即固定分母下低 **6.00 percentage points**。

## 3. 完整 Eval300 效率与成本

### 3.1 调用、图片、tokens 与延迟

| 指标 | R1 visual-only 300 | R3 300 |
|---|---:|---:|
| 已记录 provider sends/attempts | 1,956 | 1,444 |
| 收到并保存响应 | 1,951 | 1,444 |
| 冻结协议内 retry sends | 5 | 0 |
| 结果未知的请求 | 5 | 0 |
| structural-correction attempts（已含在 provider attempts） | 15 | 7 |
| 有确认响应的 input tokens | 10,790,322 | 6,188,889 |
| 有确认响应的 cache-creation input tokens | 718,375 | 1,415,926 |
| 有确认响应的 cache-read input tokens | 24,229,721 | 43,169,630 |
| 有确认响应的 output tokens | 183,730 | 144,978 |
| 实际发送的唯一图片，逐题求和 | 4,217 | 2,838 |
| 每题 provider sends：均值 / 中位数 | 6.520 / 7 | 4.813 / 4 |
| 每题确认响应：均值 / 中位数 | 6.503 / 7 | 4.813 / 4 |
| 每题唯一图片：均值 / 中位数 | 14.057 / 15 | 9.460 / 9 |
| 每题已记录 API latency：均值 / 中位数（秒） | 13.259 / 13.730 | 10.451 / 9.345 |
| 每题 route duration：均值 / 中位数（秒） | 16.517 / 15.201 | 11.175 / 9.956 |

`provider attempts` 是实际记录的发送次数，包含正常多轮工具交互、结构纠正调用及网络 retry；不能用 `attempts − 300` 表示重试。这里的 retry 数严格按同一 turn 中 `attempt_index > 1` 统计。R1 五个首发请求在网络错误后结果未知，其后冻结协议内 retry 获得了完整响应。

R1 的 token、已知 cost 和 API latency 仅覆盖 1,951 个确认响应；五个未知请求的 token/usage/latency 均未记录，未按零补齐。R3 覆盖全部 1,444 个已记录请求。逐题图片字段为 `unique_images_transmitted`；逐 turn 的 `images_transmitted` 加总与其一致，没有另行补造“请求 payload 重复携带历史图像”的统计。

路线 duration 跨 v8–v13 多个运行片段拼接。其 300 条求和是 route-level durations 的总和，**不是一次连续运行的 Eval300 wall-clock**。

### 3.2 A：论文方法评估成本

| 指标 | R1 visual-only 300 | R3 300 |
|---|---:|---:|
| 已知 route cost 总额 | **$15.02991285** | **$13.00064950** |
| 每题已知 route cost 均值 | $0.05009971 | $0.04333550 |
| 每题已知 route cost 中位数 | $0.05343400 | $0.02986930 |
| 未知请求数 | 5 | 0 |
| 正式未知费用预留（不是实际支出） | $1.26280000 | $0 |

R1 的 `$15.02991285` 是最终采用的 175 条新路线与 125 条复用路线的已知 route cost 总和；`$8.53059750` 只对应新 175 条，不能作为完整 R1 Eval300 成本。五个未知请求是否被 provider 计费无法由本地资产确认，因此 `$1.26280000` 单独作为保守预留披露，不并入“已知实际成本”。

### 3.3 B：本次修正项目账本

| 项目 | 金额 |
|---|---:|
| Correction 总 cap | $15.00000000 |
| 已结算总额（补跑 + 成功 smoke） | **$8.61269850** |
| 其中补跑 175 条已知 route cost | $8.53059750 |
| 其中已结算 smoke | $0.08210100 |
| 全部未决预留（不是实际支出） | **$2.52560000** |
| 其中正式运行五个未知请求 | $1.26280000 |
| 其中历史 smoke 五个未知请求 | $1.26280000 |
| 账本剩余额度 | $3.86170150 |

方法评估成本描述最终采用的路线；修正项目账本描述这次 correction 工作实际产生/保留的现金责任。二者范围不同，不能相加或互相替代。

## 4. 新旧 R1 逐题变化

旧 R1 正式结果为 88/300；新 visual-only R1 为 85/300。逐题转移如下：

| 旧 R1 → 新 R1 | 全部 300 | 补跑 175 |
|---|---:|---:|
| 对 → 对 | 71 | 33 |
| 对 → 错 | 17 | 17 |
| 错 → 对 | 14 | 14 |
| 错 → 错 | 198 | 111 |
| 合计 | 300 | 175 |

净变化为 `14 − 17 = −3`，所以正确数从 88 降至 85。共有 60 条最终答案发生变化，全部来自补跑 175 条；125 条复用路线的答案、source artifact 路径和 SHA 逐条一致。

新增失败 1 条：`70f2a750-f403-41b8-aabb-480eb3ab4ed4_17_16`。旧 R1 对该题给出 `D` 且与 gold 一致；新路线因请求 1680 秒的越界帧而失败，纳入“对→错”。

这些是受控输入修正后的**观测变化**，不能把全部 60 条答案变化或净 -3 直接归因于 ASR。除 ASR 内容移除外，还存在跨时间服务端非确定性以及运行管理/transport 实现差异；本数据不支持单因素因果估计。

## 5. 论文更新范围

### 5.1 可直接替换

- Direct 主结果表中的 R1：替换为 85/300、28.33%、299/300 completion、completed-only 85/299。
- Direct R1/R3 配对四格表：替换为 51 / 34 / 52 / 163。
- Direct 效率表中的 R1 调用、tokens、图片、延迟与完整 300 条已知成本，采用本报告第 3 节数字，同时披露五个未知请求。
- 所有把旧 R1 88/300 当作 visual-only 主结果的正文、表格、图注和 appendix 引用。

### 5.2 仍可复用

- R3 的 300 条冻结结果、103/300 主准确率及本报告复算的 R3 效率/成本。
- R1 中通过逐条来源核验的 125 条原本无非空 ASR 路线。
- 不依赖被替换 R1 答案、选帧或 evidence chain 的静态实验身份材料。

### 5.3 必须复核、不能直接继承

- 依赖旧 R1 175 条的回答、工具轨迹、选帧或证据引用的 evidence-support audit。
- 旧 R1 的 evidence-support / correctness-support 标签；即便答案未改变，补跑后的取帧和证据链也可能不同。
- 使用旧 R1 选帧作为输入的 frames-only 实验、selection ablation、case illustrations 和 frame-count 图表。
- 由旧 R1 逐题 correctness 或旧 88/300 派生的 paired significance、类别分解、错误案例与图表。
- 任何把 correction 多段路线 duration 拼成一次连续 wall-clock 的文字。

保守更新范围是复核全部补跑 175 条相关的下游 evidence/selection 资产，而不是只检查 60 条答案变化。本轮未执行这些后续审计或实验。

## 6. 统计口径与限制

- 主准确率分母固定为 300；失败按错误保留。completed-only accuracy 另列，并明确分母 299/298。
- Provider sends 包括失败及 retry；结构纠正调用也计入。未知请求的 token、实际费用和 latency 未记录，均未用零补齐。
- 已知 route cost 使用冻结 cache-aware pricing 字段，是最终 300 条 source artifact 中所有有 usage 记录 attempts 的总和。
- Gold 仅在无 gold 结构合并通过后加载；它没有参与路线选择、恢复或答案生成。
- 逐题详情见 `direct_visual_only_eval300_per_question.csv`；机器可读汇总及全部来源身份见 `direct_visual_only_eval300_statistics.json`。
- 仍缺少：五个 R1 未知请求的 provider 端处理/计费结果、对应 token 与 latency；原正式运行所有逐帧文件的历史全量 SHA；跨时间服务端模型字节级版本证明；一次连续执行的 correction Eval300 wall-clock。
