#!/usr/bin/env python3
"""Offline/no-gold diagnosis of the 31 ABD correction-control disagreements."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "outputs/abd_eval300_evidence_audit_correction_validation_v1"
OLD = ROOT / "outputs/abd_eval300_evidence_audit_v1"
CORRECTION = (
    ROOT
    / "outputs/abd_eval300_evidence_audit_correction_inputs_v1"
    / "ABD_EVIDENCE_AUDIT_CORRECTION_BLINDED_INPUTS"
)
OUT = ROOT / "outputs/abd_eval300_evidence_audit_disagreement_diagnostic_v1"


EXPLANATIONS = {
    "E0008": "旧审把托盘/披萨相关线索视为对复合选项的部分支持，新审认为面包铲、刷子和所涂物质三个关键条件均未出现，因而降为无支持。",
    "E0021": "旧审把黄瓜/萝卜准备与短暂搅拌作时长比较并接受选项，新审同时认为食材身份不符且地图还有多段搅拌，故判为明确冲突。",
    "E0023": "旧审只认为番茄酱无法确认，新审把相关时段地图中的“加水并搅拌”视为对番茄酱这一具体选择的正面反证。",
    "E0061": "旧审把冲洗后的后续活动概括为准备离开，新审认定其间明确插入了走动和看电视，因此“紧接着准备离开”被时序反驳。",
    "E0072": "旧审把推杆和笔的若干相关事件保留为部分支持，新审认为这些粗粒度事件不足以完成两者全程频次比较。",
    "E0077": "旧审把面包制作背景当作长清单的部分支持，新审要求逐项识别十五种食物，认为现有九帧不能支持该穷举选项。",
    "E0112": "旧审认为吸尘前的房间移动和取物大体吻合，新审指出实际物品是包和枕头而非桌子、鞋和狗链，故认为前置序列冲突。",
    "E0113": "旧审凭帧中可见的厨房到客厅通路接受选项，新审要求验证不在包内的选项C导航图本身，因无法把路径对应到该图而判无支持。",
    "E0114": "两轮都认为稀疏帧不足以完成时长比较，分歧仅在把相关活动可见性记作部分支持，还是因缺少完整边界直接记作无支持。",
    "E0122": "旧审把画面中的勺状器具视为执行敲落动作的工具，新审认为具体动作及工具均看不清，尤其不能确认是选项所称的夹子。",
    "E0269": "旧审用反复敲砖模式及稍后事件支持“下一步仍敲砖”，新审要求锚点后紧邻画面，认为重复模式只能让预测合理而不能证明实际下一事件。",
    "E0289": "两轮都承认缺少连续时长边界；旧审因此判无支持，新审把木工作业覆盖更广、清理采样更短视为方向性线索而给部分支持。",
    "E0311": "旧审认为刀和勺的频次比较未被完整计数，新审按地图中分散出现的操作区间计数，认为刀的出现区间足以区分。",
    "E0330": "旧审只称锅具放置证据不完整，新审读取C12为从水槽移到灶台，认为这明确反驳“放到台面”。",
    "E0339": "旧审把厨房/食品背景算作长食材清单的部分支持，新审要求穷举清单中每种食材均被清楚识别和处理，故判无支持。",
    "E0351": "旧审把后番茄阶段仅视为证据不足，新审按晚段地图认定随后是走动和长时间用手机，与所选六步烹饪/清洁序列冲突。",
    "E0394": "旧审未对关键先后作确定判断，新审认定椰子处理发生在加水之前，直接反驳选项所称“加水后、磨椰子前”。",
    "E0400": "旧审把多种蔬菜准备背景视为部分线索，新审认为无法得到“恰好四种加入炖菜”的完整可计数记录。",
    "E0440": "旧审认为地图没写手套或颜色所以无支持，新审把手上透明塑料袋解释为透明临时手套，因名称未明确而给部分支持。",
    "E0448": "旧审只按两个地图区间的明显长度差支持选项，新审认为选项还限定楼梯及女性共同参与，而这些条件未被明确建立。",
    "E0460": "旧审认为三步动作及顺序没有出现，新审进一步把可见的锅/蛋、盘子和橱柜序列当作与所选三步顺序相冲突。",
    "E0489": "两轮都认为只看到长技术物件清单的一部分；旧审计为部分支持，新审因穷举条件未满足而计为无支持。",
    "E0532": "旧审认为活动混杂所以无法比较时长，新审从地图覆盖量判断砖/水泥活动明显更多，认为所选方向呈相反趋势。",
    "E0557": "旧审认为反复搅拌多于刀具使用而反驳选项，新审认为刀具相关区间可能更多但类别和计数不清，只保留部分支持。",
    "E0657": "旧审把长时间处理电脑内部部件视为足以支持，新审认为题目要求与接线活动作比较，而两类事件边界不足以可靠比较。",
    "E0666": "旧审笼统认为透明手套颜色细节未完全建立，新审在1575–1594秒帧中直接识别到透明手部覆盖物，因而升为支持。",
    "E0705": "旧审遵循“未提及清洗不能证明没洗”而判无支持，新审认为清洗清单明确列出盘/锅却只写刀的放取，足以支持刀和木勺均未清洗。",
    "E0756": "旧审认为看见多次用刀但没完整统计磨椰器，不能完成频次比较；新审把反复用刀且相关证据未出现磨椰器视为足够支持方向。",
    "E0825": "两轮都只确认多个工业/施工场景而不能明确区分车库、工地、仓库；旧审给部分支持，新审因三地点穷举未成立而给无支持。",
    "E0856": "两轮都只看到二十五项工具清单中的一部分；旧审把已见子集记作部分支持，新审因无法验证完整清单而记作无支持。",
    "E0892": "两轮都认为稀疏证据不能证明全程没有剪纸和量米；旧审记无支持，新审把“现有证据未出现两事”保留为部分支持。",
}


BOUNDARY = {
    "E0008", "E0072", "E0077", "E0114", "E0269", "E0289", "E0448",
    "E0489", "E0657", "E0825", "E0856", "E0892",
}
OPERATIONAL = {"E0113"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"refuse to overwrite existing diagnostic directory: {OUT}")
    OUT.mkdir(parents=True)

    rows = list(csv.DictReader((VALIDATION / "control_disagreements.csv").open(encoding="utf-8")))
    if len(rows) != 31 or set(EXPLANATIONS) != {r["correction_audit_id"] for r in rows}:
        raise RuntimeError("31-row disagreement identity check failed")

    old_ledger = load_json(OLD / "valid_second_pass_batches.json")["batches"]
    comparison_rows = []
    full = [
        "# ABD evidence-audit disagreement diagnosis (no gold)", "",
        "This report preserves both rounds' labels. It does not adjudicate or create a new label.", "",
    ]
    for n, row in enumerate(rows, 1):
        aid = row["correction_audit_id"]
        old = load_json(OLD / "review_packages" / f"{aid}.json")
        new = load_json(CORRECTION / "items" / aid / "input.json")
        core_same = (
            old["question"] == new["question"]
            and old["options"] == new["options"]
            and old["prediction"] == new["predicted_option"]
            and old["reason"] == new["original_reason"]
        )

        old_map = Path(old["map_locator"]) if old.get("map_locator") else None
        new_map = CORRECTION / "items" / aid / "map.raw.json"
        map_same = (old_map is None and not new_map.exists()) or (
            old_map is not None and new_map.is_file() and old_map.read_bytes() == new_map.read_bytes()
        )

        old_images = old["images"]
        new_images = new["images"]
        image_identity_same = len(old_images) == len(new_images) and all(
            a["frame_sha256"] == b["sha256"]
            and float(a["resolved_timestamp_sec"]) == float(b["resolved_timestamp_sec"])
            for a, b in zip(old_images, new_images)
        )
        copied_bytes_valid = all(
            sha(CORRECTION / b["bundle_path"]) == b["sha256"] for b in new_images
        )
        timestamp_text_valid = all(
            b["model_visible_timestamp_text"]
            == f"Frame timestamp={float(b['resolved_timestamp_sec']):.3f}s from video start."
            for b in new_images
        )
        batch = row["original_batch_id"]
        old_attested = old_ledger[batch]["status"] == "valid_actual_evidence_review"
        cause = (
            "operational_definition_difference_missing_option_image"
            if aid in OPERATIONAL
            else "label_boundary_difference"
            if aid in BOUNDARY
            else "evidence_interpretation_difference"
        )
        display = (
            "未确定：旧轮有batch级实际查看证明，修正轮只有整体图片/地图查看统计；"
            "两轮均未保存逐条屏幕排版、打开顺序或逐图展示日志。"
        )
        comparison_rows.append({
            "audit_id": aid,
            "workflow": row["original_workflow"],
            "old_label": row["old_label"],
            "new_label": row["new_label"],
            "question_options_prediction_reason_same": core_same,
            "map_source_bytes_same": map_same,
            "image_count": len(old_images),
            "image_sha_order_resolved_timestamps_same": image_identity_same,
            "correction_copied_image_bytes_valid": copied_bytes_valid,
            "correction_timestamp_text_matches_resolved_time": timestamp_text_valid,
            "old_batch_actual_evidence_view_attested": old_attested,
            "actual_reviewer_presentation_identical": "undetermined",
            "diagnostic_cause": cause,
        })
        full += [
            f"## {n}. {aid}: `{row['old_label']}` → `{row['new_label']}`", "",
            f"- 旧理由：{row['old_reason']}",
            f"- 旧证据引用：`{row['old_evidence_references']}`",
            f"- 新理由：{row['new_reason']}",
            f"- 新证据引用：`{row['new_evidence_references']}`",
            f"- 具体分歧：{EXPLANATIONS[aid]}",
            f"- 源材料：问题/选项/预测/reason一致={core_same}；地图字节一致={map_same}；"
            f"图片SHA/顺序/resolved时间一致={image_identity_same}（{len(old_images)}张）；"
            f"修正包时间文字有效={timestamp_text_valid}。",
            f"- 实际展示：{display}",
            f"- 诊断归类：`{cause}`；该归类不裁决哪轮正确。", "",
        ]

    with (OUT / "material_comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0]))
        writer.writeheader(); writer.writerows(comparison_rows)
    write_text(OUT / "ALL_31_DISAGREEMENTS.md", "\n".join(full))

    cause_counts = Counter(r["diagnostic_cause"] for r in comparison_rows)
    summary = {
        "schema_version": "abd_audit_disagreement_diagnostic_v1",
        "records": 31,
        "all_core_text_same": all(r["question_options_prediction_reason_same"] for r in comparison_rows),
        "all_map_source_bytes_same": all(r["map_source_bytes_same"] for r in comparison_rows),
        "all_image_sha_order_resolved_timestamps_same": all(r["image_sha_order_resolved_timestamps_same"] for r in comparison_rows),
        "all_correction_copied_image_bytes_valid": all(r["correction_copied_image_bytes_valid"] for r in comparison_rows),
        "all_correction_timestamp_text_valid": all(r["correction_timestamp_text_matches_resolved_time"] for r in comparison_rows),
        "actual_reviewer_presentation_identical": "undetermined_for_all_31",
        "cause_counts": dict(cause_counts),
        "gold_read": False,
        "correctness_read": False,
        "labels_modified_or_created": False,
        "model_or_external_api_calls": 0,
    }
    (OUT / "diagnostic_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    protocol = """# GenS evidence-audit protocol recovery and comparison

## Recovered original sources

The original pre-review instruction is preserved in local Codex history at entry 1306/1307,
session `01a06c43-6803-7301-9a6f-2703b6e6c179`. Its review phase required:

- current Codex, not a paid API or independent reviewer model;
- neutral IDs, with method, gold, correctness and prior versions hidden;
- batches of at most ten routes, smaller when image-heavy;
- actual inspection of transmitted images, maps and necessary tool feedback;
- no inference from reason wording or filenames alone;
- the six option-support labels and the independent multi-label reason dimension;
- evidence locators, missing evidence, confidence and further-review flag;
- special caution for duration, counts, ordering and full-video absence claims;
- classification freeze before any later association.

The turn context immediately preceding the audit request and the subsequent audit turns records model
`gpt-5.6-sol` (the longer-lived session had used `gpt-5.6-terra` in earlier, unrelated turns). The frozen criteria file SHA-256
is `7e9e25909a97ade16b86c896e5b086f5148b7d3f1bda8a8118db8c07f318f4e3`.
No separate item-level reviewer system prompt, decoding settings, or pre-review labelled exemplars
were preserved. `fixed_examples.json` was generated only later and is outside this no-association
diagnostic; it was not opened here.

## Labels recovered verbatim

- `supported`: actual input contains clear evidence capable of distinguishing the selected option.
- `partially_supported`: relevant clues exist, but decisive evidence is missing.
- `unsupported`: actual input cannot support the selected option.
- `contradicted`: actual input explicitly refutes the selected option.
- `unreviewable`: images cannot be viewed, package is damaged, or verification cannot be completed.
- `no_final_answer`: no final answer exists.

The preserved operational examples are rules rather than labelled item examples: duration, count,
ordering and global non-occurrence require corresponding temporal coverage; more selected frames do
not prove longer duration, and absence from selected frames does not prove absence from the video.

## Comparison

### Original GenS/R1/R3 audit versus old ABD audit

The old ABD `AUDIT_CRITERIA_FROZEN.md` is byte-identical to the original file and has the same SHA.
The ABD task instruction also explicitly required reusing the five applicable labels, judging only
model-visible evidence, no re-answering, explicit contradiction for `contradicted`, and actual map/image
inspection. Its valid second-pass ledger attests actual evidence viewing. It changed workflow execution,
however: the old ABD second pass used clean-context subagents (local session metadata records the
parent as `gpt-5.6-sol` and the three contemporaneous worker sessions as `gpt-5.6-luna`), whereas the
original audit was documented as current-Codex batched review under `gpt-5.6-sol`. Exact
per-item prompts and decoding settings for those ABD workers were not persisted, so full execution
equivalence cannot be confirmed.

### Original GenS/R1/R3 audit versus correction review

The portable correction input archive contains no criteria/prompt file. The returned report states a
compatible high-level rule—record-local supplied evidence, the five labels, no common-sense completion,
and support assessment rather than re-answering—but adds a specific operational rule that absent option
images are `unsupported`, not technical `unreviewable`. The correction reviewer/model identity, exact
prompt, decoding settings, per-item presentation and navigation sequence were not included in the
returned freeze. Therefore the correction round cannot be verified as protocol-identical to GenS.
"""
    write_text(OUT / "GENS_AUDIT_PROTOCOL_RECOVERY.md", protocol)

    diagnosis = f"""# Diagnostic conclusion

## What differs

- Verified source evidence difference: none among the 31 controls. Core text, map bytes, image SHA/order,
  resolved timestamps and correction-copy bytes all match.
- Exact reviewer presentation equivalence: undetermined for all 31 because item-level display/open logs
  were not preserved. Package structures differ: the old package referenced local files and a separate
  map locator, while the correction package used portable copied images, an inline parsed map plus
  `map.raw.json`, and explicit `model_visible_timestamp_text` fields.
- Diagnostic reason categories: {cause_counts['evidence_interpretation_difference']} evidence-interpretation
  disagreements, {cause_counts['label_boundary_difference']} label-boundary disagreements, and
  {cause_counts['operational_definition_difference_missing_option_image']} explicit operationalization
  disagreement concerning a missing navigation-option image. These categories do not choose a winner.

## Minimum scope if GenS protocol is to be the common reference

The irreducible scope is all 31 disagreeing controls: each must be independently adjudicated using the
recovered frozen GenS criteria and a newly frozen presentation protocol. Resolving only one transition
type would leave observed disagreements in every workflow. Before adjudication, the exact reviewer
prompt/model and presentation contract must be frozen because correction provenance does not preserve
them. The 116 targets should not be merged merely because their files validated; whether they also need
fresh review depends on that 31-control adjudication and is not decided in this diagnostic.
"""
    write_text(OUT / "DIAGNOSTIC_CONCLUSION.md", diagnosis)

    files = {}
    for p in sorted(OUT.iterdir()):
        if p.is_file():
            files[p.name] = {"sha256": sha(p), "size_bytes": p.stat().st_size}
    manifest = {
        "schema_version": "abd_audit_disagreement_diagnostic_manifest_v1",
        "files": files,
        "source_control_disagreements_sha256": sha(VALIDATION / "control_disagreements.csv"),
        "source_old_criteria_sha256": sha(OLD / "AUDIT_CRITERIA_FROZEN.md"),
        "source_gens_audit_criteria_sha256": sha(ROOT / "audit/codex_evidence_audit_r1_r3_gens_v3_v1/AUDIT_CRITERIA_FROZEN.md"),
        "gold_read": False,
        "correctness_read": False,
        "labels_modified_or_created": False,
        "external_api_calls": 0,
    }
    (OUT / "diagnostic_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
