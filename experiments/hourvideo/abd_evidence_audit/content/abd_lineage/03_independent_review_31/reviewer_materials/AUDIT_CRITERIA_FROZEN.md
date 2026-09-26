# Codex辅助证据审计：冻结标准 v1

本审计由当前 Codex 分批执行，不是人工复核，也不调用独立评审模型。审阅阶段隐藏方法标签、gold、正确性和旧版本结果；Direct map 的结构可能暴露方法类型，因此不宣称完全盲法。

## A：最终选项的证据支持

- `supported`：实际输入中存在能区分候选项的明确证据。
- `partially_supported`：存在相关线索，但关键判断仍缺证据。
- `unsupported`：实际输入无法支持所选项。
- `contradicted`：实际输入明确反驳所选项。
- `unreviewable`：图片无法查看、证据包损坏或无法完成核验。
- `no_final_answer`：路线没有最终答案。

## B：最终理由的依据（可多选）

- `corresponds_to_input`
- `reasonable_unverified_inference`
- `common_sense_option_wording_or_guess`
- `unsupported_factual_assertion`
- `contradicts_input`
- `input_evidence_cannot_verify_claim`

必须核对实际发送图片；不能仅凭模型理由、文件名或统计报告判断。Direct 还须核对实际 map、截至最终回答已观察的图片与工具反馈，并把 evidence source 标为 `map`、`images`、`map_and_images` 或 `none`。map 支持但图片未确认时不能称为独立视觉支持。

持续时间、次数、先后、全程不存在等问题要求相应时间覆盖。更多选中帧不等于持续更久；选中帧未显示不等于视频中不存在。理由提到输入未显示内容时，只标为输入证据无法核验，不推断现实中不存在。

每条必须记录简短说明、可定位证据引用、缺少证据、置信度（high/medium/low）和是否需进一步复核。审计标签不改预测、不改正式评分。
