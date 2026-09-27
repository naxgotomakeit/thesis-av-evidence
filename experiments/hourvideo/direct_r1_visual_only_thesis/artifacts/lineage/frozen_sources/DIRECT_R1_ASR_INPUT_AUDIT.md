# Direct R1 ASR 正式输入核查

核查日期：2026-09-13  
核查范围：Direct-v1.2 3/16 R1/R3 Eval300 正式运行，12 个视频、300 个问题、600 条 method routes。  
核查方式：只读；未运行实验、QA、ASR、模型或 API，未读取 gold 内容，未修改任何旧资产。本文只新增报告。  
前置报告：`R1_R3_FORMAL_INDEX_METHOD_IMPLEMENTATION_AUDIT.md`。

## 1. 结论先行

1. **存在非空且可被 Direct 模型看到的 ASR。** 12 个正式 R1 map 中 7 个含非空 `audio_channel`，合计 1,007 个不同 `audio_id`、1,026 次 Coarse 引用、21,897 个去重后文本字符。正式 Direct provider 每一轮逐字节读取整个 map 并放入 system input，没有 ASR 删除或字段截断逻辑。
2. 7 个视频各有 25 条 R1 routes，因此 **175/300 条 R1 routes 暴露于非空 ASR 文本**；其余 125 条 R1 routes 的 `audio_channel` 为空。300 条 R3 routes 的正式 Variant-C maps 均为 `exact_source_asr=[]`，没有非空 ASR。
3. 按本轮新增的论文主实验目标——所有比较组均应 visual-only，且禁止地图中的 ASR——**这组正式 Direct R1/R3 输入不符合目标条件**：R1 有 175 条 routes 带音频派生文本，而 R3 为 audio-free。这是输入条件不对称；本文不推断答案是否或如何受到影响。
4. 但当时的冻结实验说明并未规定两侧 visual-only。实验专属 thesis data report 把 R1 明确写成 `Structural/ASR-oriented navigation representation`，并把 native map representation 定义为 intended method difference。因此实际输入**符合当时冻结的 R1-with-ASR 设计**；它与现在明确的全组 visual-only 论文条件冲突。现有证据更支持“冻结设计与当前目标条件不同”，不支持“运行时偶然注入 ASR”。
5. ASR provenance 能追到对应 UID 的 HourVideo MP4、音轨、抽取 WAV、WAV SHA 和本地 Whisper-small/CPU 记录。没有发现跨视频 UID/path 混配证据。不过本轮无法进行可靠人工听辨，故只能确认“来自对应视频的真实音轨”，**不能确认每条转录均有对应真实语音，也不能把异常文本直接定性为 Whisper 幻觉**。
6. 文本存在需要披露的机械异常：独立 segment 之间有 208 条 exact-text duplicate extras；`7dd...` 中 `you` 出现于 107 个不同 segment；`70f...` 的最后两个 segment 超出视频/WAV 尾部，其中 `A0162` 未被 map 附着；还存在标点-only、极短文本和混合文字。它们证明转录质量需补证，不证明答案受影响。

## 2. 证据口径与路径简称

以下结论按证据等级区分：

- **E1：正式产物直接证明**——candidate/launch lock/canonical completion、逐文件 SHA、map/ASR/WAV 实际字段。
- **E2：冻结输入与精确代码指纹联合证明**——candidate 锁定 map SHA 和 provider/loader SHA，代码显示 map 如何进入请求。
- **E3：实现与 lineage 推断**——生成代码、case config、source audit、audio cost 共同恢复来源，但没有保存原始命令行或 request body。
- **E4：未确认**——需要人工听辨或当前视频全文件重新哈希；本轮为避免干扰正在运行的任务未执行。

路径简称：

- `DIRECT` = `${PROJECT_MSC_ROOT}/msc_thesis/main_system/isolated_workspaces/hourvideo_direct_api_eval300_v1`
- `DRUN` = `DIRECT/outputs/direct_v1_formal/direct_v1_2_3x16_r1_r3_eval300_formal_v1`
- `V6` = `${PROJECT_MSC_ROOT}/msc_thesis/main_system/thesis-av-evidence-hourvideo/hourvideo_v6_1_runtime`
- `FSRC` = `V6/outputs/experiments/hourvideo_v6_6_2_formal_eval300_v2/source`
- `ARCH20` = `V6/outputs/archive/2026-08-12/contaminated_or_superseded_pilots/hourvideo_dev20_v6_1_local_av_v1/source`
- `PILOT10` = `V6/outputs/experiments/hourvideo_pilot10_api_maps_v6_3_sparse_official_full_v1/source`
- `HVVID` = `${PROJECT_MSC_ROOT}/HourVideo/videos/v2/video_540ss`

正式身份锚点：

| Artifact | SHA-256 / 状态 |
|---|---|
| `DRUN/formal_manifest_final_candidate_no_api_v3.json` | `c73aeaf41d5bf284ade54198525d8396bf3a9fa5e45587aedd49dd3744dc557d` |
| `DRUN/formal_launch_lock_v3.json` | `ebd2014fcd952779444064510d39147149c174bc1ff2bc7fa231dfd589c09d5c` |
| `DRUN/canonical_summary_v1/canonical_manifest.json` | `08685a33eaaf7c9824e426364f0ce3521fadcdcb2bb1b3729f57b2d866f57cc7`；label=`FINAL CANONICAL DIRECT R1/R3 EVAL300 RESULT`；600 route artifacts；raw closure unchanged；validation PASS |
| 正式 provider 指纹 | candidate `provider_hash=dba180326d726dddaaf6880e4cca03575881aabd7b08622d53be39f81d9699cf` |
| 正式 map loader 指纹 | candidate `maps_loader_hash=70d727f0bbc05ad2c8581c1c07796f76a0f43629886aaeca7b78836d573352a4` |

## 3. 12 视频正式 map 身份与 SHA

下表 SHA 来自 candidate，并已对当前 path 的文件内容重新核对；24/24 个 R1/R3 map 均匹配。每个视频在 candidate 中均有 25 个问题和 25 R1 + 25 R3 routes。

| Video UID | R1 map SHA-256 | R3 map SHA-256 |
|---|---|---|
| `6fd90f8d-7a4d-425d-a812-3268db0b0342` | `b7657ad79880f4eb10fd5e3221044d7a2e4dc29d5d1332bc190556d98195a0ff` | `14a5e59a5a8a4a3530d378e215c91e784b3105a56993401837b7e728113ca664` |
| `4572b198-2c1c-4920-bcf0-95fcebe12261` | `3ce668babbe2b106f8d1652895bd3cdc3723bb81b7ea10797c94d0295e1de424` | `582ac72405398fc0d3a85fd39f181f72e7f88f80fe550378818a821fcafd7806` |
| `71fbc5bf-7e2a-415d-86bc-3a948742e904` | `1f9ef7a11bd71ac5c52b67f8af16cbb121d670d51a61b4112b85cc3ee51b5872` | `2ede3e0c9a069f412ee7250fd88f6cbb1ee8671357e5189c23c6e0256d0616c2` |
| `115774b6-534d-444f-b7aa-d1b834eb0ee7` | `62147202409b7a2f4ddd9fee0ca90205ab1e590deeae55ea8bd69c15a39bd099` | `3b54111f311fa9e6dd699f9ed25f3405942eb3989d2c71dc02d47680784690b4` |
| `7e512589-aa97-41e8-83d3-af2e83e4fd06` | `3315c1d7d681315236f79e9a87eaccfb4b4b555b010db3b14d7a45a2e0706542` | `67aa4f36319a017e9906556d39281183620f901a2b37f74c84d6becab4ce5279` |
| `d3a0899e-2093-454c-9f65-30087883193a` | `4da4b06501ff4519fba189875ed424cecaa18f1bce16312d62c0127af9d313b9` | `cd078a1905c884b0afa48a2a08ebe0a3c3eb62a44cd8667247a791a5e6526342` |
| `41a86310-2cc1-48f9-b5b5-6b495a95fbac` | `dba922b3f0c68a3199bf5084c33e5956d4979f837718bcf336699584ae33f8d1` | `9cc8bdf2c41d3e7acebb8d109ff000b4c6647440038613d2634239ab37888d7f` |
| `db3f7933-dfa0-4678-9d4f-393b628ded45` | `184e6445fe7801346b55f496e45c07ab0465d5037f85ae77a0b04f2ecf408098` | `c49312f89dab336b691861b8be8089a6d0fdd76abc34ae82cda7039fc35b11ac` |
| `7ddbf8a2-5b3e-44cd-b9dc-db17ec06831b` | `7386e169139a3ce9756303b48531ac1c515d0914f34c99feacce4465a1f7cdfd` | `836d499723974850b87d98becd22e2409c6b57db24cde8a8009fd80533f58373` |
| `819c8af7-851f-434f-ab32-318285bc54b1` | `c8569d6c02067c3a495d0e1528edc5f9537d9d9cf1a8567887a18307b4156347` | `6d5777f050298fce9b4f3328791ef2ee5abea4f61c0f24c9cdfefc4f4f41d493` |
| `70f2a750-f403-41b8-aabb-480eb3ab4ed4` | `626b8cc8112708f4e7ac788cea9a7b7fa08b1f43a8b751547f22a961f06eada6` | `5c99bb29944d76d20fd2f4bc6909468702d85b71bc9bfbdf73c1414894edbc12` |
| `a6d45e95-8dc0-4932-83bf-ec53e265a16a` | `0a8c4634de4a5d5f33277c737bed2d6d64a53989ad8cd3d01bfaba2327c9813c` | `8d08900826b9317db6370c3b5b6b75f52b0dab8fc5d263b66bd248154fa88426` |

正式 map 路径分别为：

- R1：`FSRC/video_assets/<uid>/r1_av_navigation_map.json`；
- R3：candidate 指向 `.../hourvideo_v7_4_variant_c_budgets_v1/work_index/cases/<uid>/r3_2_navigation_map.json`。

## 4. 12 视频 ASR 覆盖统计

计数定义：`源 segments` 是 `audio_asr.json.segments`；`map refs` 是所有 Coarse `audio_channel` 行数之和；`map unique` 按 `audio_id` 去重；`跨 Coarse 重引` 是同一个 `audio_id` 因时间跨边界而在多个 Coarse 出现的额外引用；`文本重复 extras` 是不同 `audio_id` 具有完全相同、去首尾并折叠空白后的文本所产生的额外条数。字符数按同样规范化后的 Unicode 字符计。它们不涉及 gold 或答案。

| Video UID | 音轨 | 源 segments / 非空 | map refs / unique | 源字符 / map unique 字符 | 跨 Coarse 重引 | 文本重复 extras | 源时间范围 (s) | R3 ASR refs | 正式 routes |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| `6fd90...0342` | 有 | 359 / 359 | 366 / 359 | 9,440 / 9,440 | 7 | 25 | 0.00–3598.24 | 0 | R1 25 ASR+；R3 25 ASR0 |
| `4572b...2261` | 有 | 80 / 80 | 83 / 80 | 1,777 / 1,777 | 3 | 6 | 0.00–3003.98 | 0 | R1 25 ASR+；R3 25 ASR0 |
| `71fbc...e904` | 无 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 0 | — | 0 | R1 25 ASR0；R3 25 ASR0 |
| `11577...0ee7` | 有 | 34 / 34 | 35 / 34 | 682 / 682 | 1 | 1 | 0.00–4319.90 | 0 | R1 25 ASR+；R3 25 ASR0 |
| `7e512...fd06` | 无 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 0 | — | 0 | R1 25 ASR0；R3 25 ASR0 |
| `d3a08...193a` | 无 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 0 | — | 0 | R1 25 ASR0；R3 25 ASR0 |
| `41a86...fbac` | 无 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 0 | — | 0 | R1 25 ASR0；R3 25 ASR0 |
| `db3f7...ed45` | 有 | 39 / 39 | 40 / 39 | 240 / 240 | 1 | 22 | 0.00–1565.42 | 0 | R1 25 ASR+；R3 25 ASR0 |
| `7ddbf...831b` | 有 | 205 / 205 | 207 / 205 | 1,970 / 1,970 | 2 | 125 | 0.00–5640.22 | 0 | R1 25 ASR+；R3 25 ASR0 |
| `819c8...54b1` | 有 | 129 / 129 | 131 / 129 | 2,445 / 2,445 | 2 | 23 | 0.00–1779.48 | 0 | R1 25 ASR+；R3 25 ASR0 |
| `70f2a...ed4` | 有 | 162 / 162 | 164 / 161 | 5,349 / 5,343 | 3 | 6 | 30.00–1685.00 | 0 | R1 25 ASR+；R3 25 ASR0 |
| `a6d45...16a` | 无 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 0 | — | 0 | R1 25 ASR0；R3 25 ASR0 |
| **合计** | **7 有 / 5 无** | **1,008 / 1,008** | **1,026 / 1,007** | **21,903 / 21,897** | **19** | **208** | — | **0** | **R1 ASR+ 175；R1 ASR0 125；R3 ASR0 300** |

补充核对：

- 12/12 `audio_asr.json.video_uid` 与 candidate video UID、case-config UID、MP4 文件名 stem 一致。
- 12/12 `audio_asr.json` 当前 SHA 与 `FSRC/source_manifest.json.video_assets.<uid>.r1_audio_source_sha256` 一致。
- 12/12 R1 map 中没有 source 外的 `audio_id`。
- 若不消除 19 次跨 Coarse 引用，map 中所有 ASR rows 的文本字符总量为 22,282；这不是 22,282 个独立文本字符。
- 只有 `70f...` 的 source `A0162` 未进入 map：它从 1681.24 s 开始，已完全位于 hierarchy/map 的 1675.00 s 末端之后。`A0161` 从 1674.82 s 开始，因与尾段相交而进入 map，但其 end=1681.24 s 超出视频尾部。
- R3 的 12 个正式 map 所有 Coarse 都是 `exact_source_asr=[]`，不是仅顶层声明 visual-only。

## 5. ASR 来源链

### 5.1 正式构建链

实际 lineage 为：

```text
Direct candidate route
  -> exact R1 map path + SHA
  -> FSRC/source_manifest.json
  -> r1_audio_source path + SHA
  -> audio_asr.json (video_uid, audio_id, timestamps, exact_transcript)
  -> audio_cost.json (Whisper-small, CPU, WAV SHA, segment count, API calls=0)
  -> audio_16khz_mono.wav
  -> case_configs/<uid>.json (exact HVVID/<uid>.mp4)
  -> HourVideo MP4 + frozen source_artifact_audit video SHA
```

正式 adapter 明确执行以下动作，而不是靠文件名猜测：

- `V6/src/experiments/hourvideo_v6_6_2_shared_coarse_contract_v1/formal_eval300.py:74-82`：先找 Pilot10 `audio_asr.json`，否则找归档 Dev20；找不到就失败。
- 同文件 `:177-188`：读取 `segments`，传入 `_build_r1_map`，并验证生成的 R1 map。
- 同文件 `:201-209`：将实际音源 path/SHA 写入 `source_manifest`。
- `V6/src/experiments/hourvideo_r1_av_r3_2_single_video_smoke/local_prepare.py:109-149`：从 `cfg.video_path` 用 ffmpeg `-vn -ac 1 -ar 16000` 抽取 WAV；读取 16 kHz mono waveform；以 Whisper small、指定语言、`condition_on_previous_text=False` 转录；写出 `audio_asr.json` 和含 WAV SHA 的 `audio_cost.json`。
- 同文件 `:152-155`：按时间 overlap 把 segment 附着到 Medium。
- `V6/src/experiments/hourvideo_r1_av_r3_2_single_video_smoke/live_runner.py:377-395`：汇总到 Coarse，并仅在该 Coarse 内按 `audio_id` 去重。这解释了同一 segment 跨 Coarse 时的 19 次额外引用。该 `live_runner.py` 当前整文件 SHA 与冻结 closure 中旧 SHA 不同，因此这里只把行号作为与正式产物一致的实现解释；正式 map bytes 由 candidate SHA 直接锁定。

### 5.2 source path 与归档状态

下表给出正式 `r1_audio_source` 的精确 SHA；`ARCH20`/`PILOT10` 的完整展开见第 2 节。

| UID | Source | `audio_asr.json` SHA-256 |
|---|---|---|
| `6fd90...0342` | `ARCH20/cases/<uid>/audio_asr.json` | `88d6b9a3bd975a1f04230ead02a9ba042f6527569c031f4f319bbc1ab9a74e12` |
| `4572b...2261` | `ARCH20/cases/<uid>/audio_asr.json` | `e58916a8c834bc48372f8a11f5c15455a8b2155495361581ce8b61e0ce568151` |
| `71fbc...e904` | `ARCH20/cases/<uid>/audio_asr.json` | `ca02625d211eb58e71c4331f1d1feea1493940965712d71b73672c4c8a73409b` |
| `11577...0ee7` | `ARCH20/cases/<uid>/audio_asr.json` | `3c4da1f1d27d402b10194e54a29b6475cccf1b0fcee4848762473f78a7e83ac1` |
| `7e512...fd06` | `ARCH20/cases/<uid>/audio_asr.json` | `54af807256b8a4c51ac4d9d6f319cffe8de3902e0f3740f0b0fcd8110c98f756` |
| `d3a08...193a` | `ARCH20/cases/<uid>/audio_asr.json` | `ee098ea32e9b0a13a68247a269bef90134eeeafa3f510a1839e56da426a19d31` |
| `41a86...fbac` | `ARCH20/cases/<uid>/audio_asr.json` | `8b69944dd3f29a5d97be712ae2187b38f88edb4935faee07a20f5129498c2995` |
| `db3f7...ed45` | `ARCH20/cases/<uid>/audio_asr.json` | `3eb68b4ede45e275e51f654a268ce21918a930b61bf2ac935843d652b81d5f66` |
| `7ddbf...831b` | `ARCH20/cases/<uid>/audio_asr.json` | `53e323d6a239993ee8e5019656fd156ec5a9c0e9ed348cf06bde3a91c4f02d7a` |
| `819c8...54b1` | `PILOT10/cases/<uid>/audio_asr.json` | `ca0703079589d134a504e8895ccad7f148f463b7dc788536df167ce1853f6476` |
| `70f2a...ed4` | `PILOT10/cases/<uid>/audio_asr.json` | `5184af2d9fa825239203e23a0e3053c7de4a74fd2604762d717fb76a1cff12af` |
| `a6d45...16a` | `PILOT10/cases/<uid>/audio_asr.json` | `9212bc436194bcd8091bcf56e7f2c148c2919db7a2406123f99070c66b04946e` |

- 9 个 UID 的正式 `r1_audio_source` 直接位于 `ARCH20/cases/<uid>/audio_asr.json`；其中 5 个非空、4 个为无音轨的空 segments。
- 其余 3 个位于 `PILOT10/cases/<uid>/audio_asr.json`；其中 `819...`、`70f...` 非空，`a6d...` 为空。
- `819...` 与 `70f...` 的 Pilot10 `audio_asr.json` 分别与 ARCH20 副本逐字节同 SHA（`ca070307...`、`5184af2d...`），相邻 `audio_cost.json` 也分别逐字节同 SHA（`55714b34...`、`48ef5fd7...`）。Pilot10 目录没有 WAV，但 ARCH20 的同 UID WAV SHA 与这两份 cost 记录完全一致。因此它们是历史 ASR/WAV 的 byte-identical reuse，不是新的转录。
- `V6/outputs/archive/2026-08-12/ARCHIVE_MANIFEST.json:21-25` 给该组的理由是：旧 V6.1 assets/experiments 已被 lineage-validated Pilot10 supersede，`not valid for current comparisons`。这是一条组级归档政策，不是逐 ASR 的错视频或内容污染判定。
- 后来的正式 V6.6.2 adapter 又在源码中显式把该归档目录作为 fallback。故这里存在**provenance/设计张力**：归档政策说不用于 current comparisons，正式 adapter 却有意复用；但 UID、路径、ASR SHA、WAV SHA 和视频配置没有显示跨视频来源错误。不能仅凭目录名把文本判成污染，也不能忽略这项正式复用事实。

### 5.3 MP4、WAV、Whisper 记录

本轮使用 `ffprobe` 只读核对当前正式 MP4。表中“视频 SHA”是冻结 `V6/outputs/experiments/hourvideo_dev50_r1_offline_v1/cases/<uid>/source_artifact_audit.json` 所记录的完整 SHA；为避免对正在运行任务造成约 11 GiB 额外顺序 I/O，本轮没有重新哈希当前 MP4 bytes。7 个非空 ASR 视频的 WAV 当前 SHA 均已与 `audio_cost.audio_sha256` 核对。

| UID | MP4 / 音轨时长 (s) | 冻结视频 SHA-256 | WAV 时长 / SHA 关系 | ASR 记录 |
|---|---|---|---|---|
| `6fd90...0342` | 3598.067 / AAC 3598.005 | `1058a4bd4c43789f7a75996cae06b5b79ba96a936da7e372f22d911e3f694218` | 3597.739；SHA match | small, CPU, 359, API 0 |
| `4572b...2261` | 3003.306 / AAC 3003.263 | `08e05e60ae7c5710ad9bec43126e98659995c6beb95e46899909344af0f91b15` | 3005.867；SHA match；比 MP4 container 长约 2.56 s | small, CPU, 80, API 0 |
| `71fbc...e904` | 4044.867 / 无音轨 | `5a45153480ee8ce839f675ec9d98e630f91e28aa3edf4d20e396e7befe9ce4cc` | 无 WAV | no_audio_source, 0 |
| `11577...0ee7` | 4396.867 / AAC 4396.812 | `aa8616d60bf71c4b47e8712bd88cc0c2ce7e5d35f237829a60a0da0697c4845a` | 4396.715；SHA match | small, CPU, 34, API 0 |
| `7e512...fd06` | 3507.300 / 无音轨 | `b2a0318618d799700e660c9b3e1c2da7e71c8928ba63135c02fb0347bfd41636` | 无 WAV | no_audio_source, 0 |
| `d3a08...193a` | 3720.434 / 无音轨 | `2700e901bbda3aaffab5e0c88df212aadf8f21eeb513b76d267233e03bdec867` | 无 WAV | no_audio_source, 0 |
| `41a86...fbac` | 2783.467 / 无音轨 | `98be87fde0d9a7dad2dc683ff5dac03cfc2118b498d8a9297b4dad90cca46985` | 无 WAV | no_audio_source, 0 |
| `db3f7...ed45` | 1594.870 / AAC 1594.827 | `40674b78a39ebbd0c33c76d6549ba7b44c7bde2c145c8621db9461d0da0cfbf0` | 1594.752；SHA match | small, CPU, 39, API 0 |
| `7ddbf...831b` | 5645.900 / AAC 5645.857 | `9ca9a0b79aaeecd7d079be64d01fc14f86b7603e745b5b3f6481af840739fc75` | 5645.931；SHA match | small, CPU, 205, API 0 |
| `819c8...54b1` | 1792.446 / AAC 1792.403 | `e4ddc242dbe7192f7f6dfbefb671852e4ef138615c100c6888693a041d600ddb` | 1792.427；由 ARCH20 WAV 补链，SHA match | small, CPU, 129, API 0 |
| `70f2a...ed4` | 1675.193 / AAC 1675.150 | `a86976b2e747d4a862e0d2836e2c7b3a93637e0feddc63a6a72b688480208eb9` | 1675.243；由 ARCH20 WAV 补链，SHA match | small, CPU, 162, API 0 |
| `a6d45...16a` | 1800.000 / 无音轨 | `fed0f36767ed1a36fcadaf221faa0ab68fb189e551e01b6caf3b6ca7241982af` | 无 WAV | no_audio_source, 0 |

所有保留的 `ARCH20/case_configs/<uid>.json` 均满足：配置 `video_uid`、ASR `video_uid` 和 `video_path=HVVID/<uid>.mp4` 一致；audio config 为 `/usr/bin/ffmpeg`、Whisper `small`、language `en`。这与上述生成代码、WAV SHA 和时长共同构成 E3 来源证据。

## 6. 真实转录样例与异常模式

以下完全按 source segment 时间选取：开头=第一条，中间=segment midpoint 最接近 ASR 时间范围中点，结尾=最后一条；重复项为最常见 exact text。没有按问题、答案或正确性挑选。引号内是 artifact 原文。

| UID | 开头样例 | 中间样例 | 结尾样例 | 明显重复样例 |
|---|---|---|---|---|
| `6fd90...0342` | A0001 0.00–6.86: “ok” | A0190 1823.10–1826.04: “Ok.” | A0359 3583.70–3598.24: “Copyright included in the wär比較 작아.” | “Yes.” 6 IDs；“Add the tomato paste.” 连续 4 IDs |
| `4572b...2261` | A0001 0.00–2.00: “I” | A0042 1472.82–1482.84: “Get me a lollipop an air hole.” | A0080 2999.98–3003.98: “I'll take this off. I'll put it on my head and I'll stop it.” | “I”、“or”、“The” 各 2 IDs |
| `11577...0ee7` | A0001 0.00–8.54: “this is so good!” | A0023 2021.04–2026.16: “...” | A0034 4311.52–4319.90: “SOFT DIFFERENT” | “I” 2 IDs |
| `db3f7...ed45` | A0001 0.00–2.00: “I” | A0016 891.48–898.58: “the” | A0039 1563.36–1565.42: “you” | “you” 17 IDs；“the” 7 IDs |
| `7ddbf...831b` | A0001 0.00–2.00: “I” | A0111 2818.16–2820.22: “you” | A0205 5638.16–5640.22: “you” | “you” 107 IDs；“Okay.” 8 IDs；“Boom.” 4 连续 IDs |
| `819c8...54b1` | A0001 0.00–6.10: “.” | A0020 877.90–878.90: “Okay.” | A0129 1777.48–1779.48: “The water is getting bigger.” | “This is the place where the fire was put on.” 3 IDs；“16.” 3 连续 IDs |
| `70f2a...ed4` | A0001 30.00–32.00: “that she flipped this.” | A0076 842.92–847.98: “And I eat a lot of bread.” | A0162 1681.24–1685.00: “Golden” | “I'm going to put it in the oven.” 3 IDs |

机械扫描还得到：47 个长度不超过 2 字符的非空 segments，9 个不含任何字母/数字的 segments，46 个含非 ASCII 字符的 segments。后三类只是 review flags，不等于错误。

重点异常：

- `7dd...` 的 107 个独立 IDs 具有 exact text `you`，多数在后半段以约 30 秒间隔出现；`db3...` 也有 17 个 `you`。这不是同一 segment 跨 Coarse 重引，而是 source ASR 本身不同 IDs 的文本重复。
- `6fd...` 末条为多语种/乱码样式文本；`115...` 中有 punctuation-only `...`；`819...` 以 punctuation-only `.` 开头。
- `70f...` MP4/WAV 约 1675.2 s，而 source A0161 为 1674.82–1681.24、A0162 为 1681.24–1685.00。A0161 因 overlap 进入 map，A0162 未进入。这是已确认的时间戳越界异常。
- `457...` WAV 比 MP4 container 时长约长 2.56 s；ASR 末尾 3003.98 s 仍处于 WAV 内，但略过 MP4 container duration。它可能来自 container/extraction timing，现有证据不足以判断原因。

本轮只通过 ffprobe、哈希和结构化 artifact 核对，**没有人工播放/听辨这些片段**。所以可以说存在真实音轨和可追溯 WAV，不能说每条文字都得到真实语音逐句支持；重复、乱码或越界也只能标记为“转录异常/需补证”，不能直接宣称 hallucination。

## 7. 是否进入实际 Direct 请求

### 7.1 已直接证明的部分

- `DRUN/formal_manifest_final_candidate_no_api_v3.json` 的 600 routes 为每条 route 锁定 `method`、`video_id`、`map_path` 和 `map_sha256`。
- `DIRECT/src/direct_api_v1/formal_runtime.py:289-297` 按 route map path 和 expected SHA 调用 loader，然后用该 `DirectInput` 建 session/agent。
- `DIRECT/src/direct_api_v1/maps.py:38-57` 验证 map JSON 和 SHA；没有删除 `audio_channel`。
- `DIRECT/src/direct_api_v1/anthropic_provider.py:125-131` 每轮重新读取 `map_path` 原始 UTF-8 文本，并把 `VIDEO MAP ... + raw_map` 放入 system content。
- 同文件 `:229-267` 实际将 `system=self._system(state)` 传给 `client.messages.create`。没有 map-field truncation、ASR filtering 或摘要转换；ephemeral cache 只改变缓存计费/复用，不改变内容可见性。
- candidate 的 provider/loader fingerprints 与这些当前文件精确匹配。因此对正式冻结代码和 map bytes 的映射属于 E2 强证据。

### 7.2 未保存的部分

未找到正式 API request 的完整 `system/messages` 正文：

- `DRUN/journals/request_start.jsonl` 只保存时间、experiment/route/attempt/turn、provider/model；
- `attempt_end.jsonl` 保存响应与 token/cost/action telemetry；
- `controller_result.jsonl` 保存解析后的 action；
- 600 个 `route_artifacts/*.json` 保存 route/turn/provider telemetry；
- 均不含发送前的 raw map/system body。

因此“模型看到了完整 `audio_channel`”不是 request-body log 的 E1 逐请求复现，而是由**正式 route map SHA + 正式代码 SHA + 无过滤数据流**得出的 E2 结论。证据很强，但报告必须保持这个区分。

### 7.3 暴露路线的精确定义

正式 route ID 格式是 `R1:<question_id>` 或 `R3:<question_id>`。非空 ASR 暴露集可完整定义为：

```text
method == R1
AND video_id in {
  6fd90f8d-7a4d-425d-a812-3268db0b0342,
  4572b198-2c1c-4920-bcf0-95fcebe12261,
  115774b6-534d-444f-b7aa-d1b834eb0ee7,
  db3f7933-dfa0-4678-9d4f-393b628ded45,
  7ddbf8a2-5b3e-44cd-b9dc-db17ec06831b,
  819c8af7-851f-434f-ab32-318285bc54b1,
  70f2a750-f403-41b8-aabb-480eb3ab4ed4
}
```

每个 UID 正好 25 条，所以共 175 条。其余 5 个 UID 的 125 条 R1 routes 只有空 `audio_channel`；全部 300 条 R3 routes 只有空 `exact_source_asr`。这里的“暴露”只表示文本进入模型输入范围，**不表示模型使用了文本，更不表示答案或准确率受影响**。

## 8. 与冻结设计和当前 visual-only 标准的一致性

| 判断维度 | 证据 | 结论 |
|---|---|---|
| 当前论文主实验目标 | 本轮用户明确：全部比较组 visual-only，不使用音频或音频派生信息，包括 map ASR | 正式 Direct 全体不满足：R1 175 routes 有非空 ASR，R3 300 routes 无 ASR |
| 当时实验专属冻结说明 | `DIRECT/docs/experiments/direct_v1_2_3x16_r1_r3_eval300_v1_thesis_data_report.md:9-19` 定义同一 Direct agent 比较两种 native maps；`:23-30` 将 R1 明确列为 `Structural/ASR-oriented`，并说 native map 是 intended method difference | 当时设计允许且预期 R1 ASR；actual map 与该设计一致 |
| 正式 source adapter | `formal_eval300.py:74-82,177-188,201-209` 强制寻找 ASR source、传入 R1 builder、冻结 path/SHA | 不是 loader 偶然读到，也不是 query-time 注入；是 offline formal input adaptation 的显式步骤 |
| R3 Variant C | 正式 R3 maps 的每个 `exact_source_asr=[]` | R3 实际 audio-free |
| generic repo 文档 | `DIRECT/README.md` 和 visual pipeline 文档含 visual-only 叙述，但不是该 Direct R1/R3 experiment-specific contract | 不能覆盖正式专属 report 与实际 SHA-locked maps；属于文档范围差异 |

准确表述应是：

- **历史正式 Direct 实验**：R1 structural+ASR map versus R3 visual-caption semantic map，共享 Direct downstream；不是严格的 visual-only representation comparison。
- **当前论文期望的主实验条件**：R1/R3 均不得含 ASR。现有正式 Direct 结果不能在不加限定的情况下被描述成满足此条件。
- 这是已经确认的**输入条件差异**和**冻结设计与当前论文目标的冲突**。没有发现 ASR 来自另一 UID/另一视频的证据；有转录/时间戳异常，但真实性仍需听辨补证。

## 9. 已确认问题、未确认事项和最小后续检查

### 已确认

1. 7/12 R1 maps 有非空 ASR，7 个 map 合计 1,007 个 unique attached segments；R3 全空。
2. 175 条 R1 routes 处于 ASR 暴露范围，125 条 R1 与 300 条 R3 routes 无非空 ASR。
3. Direct 冻结实现把完整 map 作为 system content 发送，无 ASR 过滤/截断。
4. source adapter 有意附着 ASR；当时 experiment-specific 设计也明确写 ASR-oriented。
5. 7 个 ASR+ UID 都有对应视频音轨及 WAV SHA lineage；5 个 ASR0 UID 当前 MP4 无音轨，cost 标记 `no_audio_source`。
6. 存在大量独立 segment exact-text repetition、短/标点/混合文字以及 `70f...` 尾部越界。
7. 归档标签是 superseded/current-comparison policy，不是逐文件污染证明；正式 builder 后来显式 fallback 到该目录。

### 未确认

1. 未保存完整 request body，故模型输入范围依赖 E2 冻结实现追溯，而非 E1 request log。
2. 未人工听辨；不能确认每个 segment 是否对应真实人声、背景媒体语音或模型幻觉。
3. 当前 12 个 MP4 bytes 未重新计算 11 GiB SHA；使用冻结 source audit SHA，并以当前 ffprobe、UID/path/config 交叉核对。
4. `457...` WAV 较 container 长、`70f...` ASR 越界的具体成因未确认。
5. 本轮没有、也不能从现有事实推断 ASR 对任何答案、准确率或路线行为的影响。

### 最小后续检查建议（本轮不执行）

1. 在服务器空闲窗口对 12 个当前 MP4 重新做 SHA-256，并与 `source_artifact_audit.json` 比较，补齐 current-byte E1。
2. 由人工在不看问题/gold的条件下，对 7 个 ASR+ 视频固定抽取开头/中间/结尾、重复高发和 `70f...` 越界附近片段做盲听对齐；记录“speech / non-speech / media speech / mismatch / uncertain”。
3. 论文先冻结命名：把现有结果明确标为 `Direct R1 structural+ASR vs R3 visual-semantic`；不要标成 all-visual-only。
4. 若未来需要真正的 visual-only 主比较，应先冻结 input contract、允许字段白名单、R1 map 的 ASR-removal/materialization 规则、12-video map SHA、request-body logging policy 与 paired route manifest，再决定是否需要独立正式运行。本报告不建议直接修改现有 map。

## 10. 对 A–D 的明确回答

**A. 是否存在非空且模型可见的 ASR？**  
是。7 个 R1 maps 含非空 ASR，覆盖 175 条正式 R1 routes。完整 request body 未保存，但 SHA-locked map 与精确 provider 实现形成 E2 强证据：原始 map 在每轮进入 system input，无过滤或截断。R3 没有非空 ASR。

**B. 是否确认来自对应视频，是否有真实语音支持？**  
确认到“对应 UID 视频的真实音轨/WAV 来源”层面：case config、MP4 path/UID、音轨、WAV SHA、Whisper cost 和 ASR UID 一致，未发现错视频来源。不能确认到“每条文字由真实语音支持”层面，因为未做人工听辨；重复、乱码和越界条目仍需补证。

**C. 哪些正式路线暴露于这些文本？**  
上述 7 个 UID 下的全部 25 条 R1 routes，共 175 条。其余 125 条 R1 routes 和全部 300 条 R3 routes 无非空 ASR。暴露不等于答案受影响。

**D. 目前发现的性质是什么？**  
已确认的是：相对于当前 all-visual-only 目标的**输入条件差异/不对称**；相对于当时冻结设计则是预期的 R1 ASR attachment。未发现错 UID/错视频来源。已确认有重复、短文本及尾部越界等**转录异常**；是否为真实语音误转或 hallucination 仍需人工听辨补证。
