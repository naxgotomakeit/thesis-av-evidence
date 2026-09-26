#!/usr/bin/env python3
"""Post-freeze, no-gold comparison of old, correction, and independent Sol labels."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/abd_eval300_disagreement_independent_review_v1"
FREEZE = OUT / "BLINDED_REVIEW_FREEZE.json"
MAPPING = OUT / "identity_mapping_private.json"
REVIEWS = OUT / "blinded_reviewer_output/blinded_reviews.jsonl"
DISAGREE = ROOT / "outputs/abd_eval300_evidence_audit_correction_validation_v1/control_disagreements.csv"


WHY = {
"E0400":"本轮与原ABD一致：多种蔬菜是关键方向线索，但无法证明恰好四种，因此保留部分支持。",
"E0114":"本轮与correction一致：稀疏帧缺少两类活动的完整时间边界，不能支持时长比较。",
"E0061":"三轮不同：本轮既承认后续拿包与离开准备相容，也确认中间先发生走动/看电视，故取部分支持。",
"E0008":"本轮与correction一致：托盘和披萨背景存在，但面包铲、刷涂和所涂物均未建立。",
"E0269":"本轮与correction一致：反复敲砖只能支持可能延续，不能证明锚点后的紧接事件。",
"E0351":"本轮与correction一致：晚段明确转向走动和手机使用，反驳所选六步后续序列。",
"E0825":"本轮与correction一致：画面不能把工作区域区分为选项声称的三个独立地点。",
"E0557":"本轮与原ABD一致：地图中勺/搅拌区间多于刀具区间，方向与所选频次比较相反。",
"E0856":"本轮与correction一致：只识别出超长工具清单的一小部分，不能支持穷举选项。",
"E0289":"本轮与原ABD一致：木工和清理均可见，但没有连续边界可作时长比较。",
"E0460":"本轮与correction一致：可见的蛋和香肠处理顺序直接不符合所选三步顺序。",
"E0021":"三轮不同：本轮认为早期蔬菜准备提供方向线索，但食材身份与完整时长均不确定，故取部分支持。",
"E0113":"本轮与correction一致：通路本身可见，但缺失的导航选项图使证据无法对应到所选图。",
"E0532":"本轮与correction一致：砖/水泥作业覆盖明显更广，趋势反驳挖掘/对齐更久的说法。",
"E0489":"本轮与correction一致：只看见长技术物件清单的部分成员，穷举选项没有成立。",
"E0330":"三轮不同：本轮把水槽旁台式灶具同时视为台面位置，认为所选位置得到直接支持。",
"E0756":"三轮不同：本轮实际识别到磨椰动作，也看到刀具动作，但缺少完整次数，故判无支持。",
"E0705":"本轮与correction一致：清洗段明确列出盘锅清洗而只记录刀的放取，审阅者据此接受“两者均未清洗”。",
"E0666":"本轮与correction一致：1575–1594秒帧清楚显示透明手部覆盖物。",
"E0023":"本轮与correction一致：相关锅内液体呈浅色奶油状而非番茄酱，构成具体反证。",
"E0657":"三轮不同：本轮认为两类电脑操作都有出现，但无法分段汇总时长，因此判无支持。",
"E0122":"三轮不同：本轮认为器具外观与夹子相符，但静帧不能确认特定敲落动作，因此取部分支持。",
"E0311":"三轮不同：本轮按完整地图统计为勺类操作区间更多，认为明确反驳刀用得更多。",
"E0072":"三轮不同：本轮把长切割阶段反复使用推杆、早期有限用笔视为足够区分频次。",
"E0448":"本轮与原ABD一致：地图中喷涂/楼梯作业区间远长于与女性搭棚区间。",
"E0892":"三轮不同：本轮在相邻帧中直接看到干米倒入容器并测量，足以反驳“没有量米”的合取选项。",
"E0112":"三轮不同：本轮确认部分前置动作，但无法确认搬放的是桌子，因此取部分支持。",
"E0339":"本轮与correction一致：地图主题不对应且图片不足以逐项确认完整食品清单。",
"E0440":"本轮与原ABD一致：地图只写塑料袋套手，没有可见图片证明透明颜色。",
"E0394":"本轮与correction一致：椰子处理发生在加水前，后续没有另一段磨椰事件，时序与选项相反。",
"E0077":"本轮与correction一致：九张稀疏帧无法验证十五种食品的穷举清单。",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, text: str) -> None:
    if path.exists():
        raise RuntimeError(f"refuse overwrite: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    freeze = json.loads(FREEZE.read_text())
    if not freeze.get("frozen") or sha(REVIEWS) != freeze["blinded_reviews_sha256"]:
        raise RuntimeError("blinded review is not validly frozen")
    mapping = {x["review_id"]: x["source_audit_id"] for x in json.loads(MAPPING.read_text())["rows"]}
    old = {x["correction_audit_id"]: x for x in csv.DictReader(DISAGREE.open(encoding="utf-8"))}
    independent = [json.loads(x) for x in REVIEWS.read_text().splitlines() if x]
    if len(independent) != 31 or set(WHY) != set(old):
        raise RuntimeError("identity count mismatch")
    rows = []
    for n in independent:
        eid = mapping[n["review_id"]]
        o = old[eid]
        same_old = n["option_support"] == o["old_label"]
        same_corr = n["option_support"] == o["new_label"]
        alignment = "original_abd" if same_old else "correction" if same_corr else "neither_three_way_all_different"
        rows.append({
            "original_audit_id": eid,
            "independent_review_id": n["review_id"],
            "original_abd_label": o["old_label"],
            "correction_label": o["new_label"],
            "independent_sol_label": n["option_support"],
            "independent_alignment": alignment,
            "original_abd_reason": o["old_reason"],
            "correction_reason": o["new_reason"],
            "independent_reason": n["concise_rationale"],
            "independent_evidence_references": json.dumps(n["evidence_references"], ensure_ascii=False),
            "independent_missing_key_conditions": n["missing_key_conditions"],
            "independent_review_flag": n["review_flag"],
            "independent_review_flag_reason": n["review_flag_reason"],
            "comparison_explanation_zh": WHY[eid],
        })
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write(OUT / "THREE_WAY_COMPARISON.csv", buf.getvalue())

    align = Counter(x["independent_alignment"] for x in rows)
    labels = Counter(x["independent_sol_label"] for x in rows)
    flags = sum(str(x["independent_review_flag"]).lower() == "true" for x in rows)
    body = [
        "# 31-control post-freeze comparison", "",
        "The independent blinded result was frozen before this identity/label association.", "",
        f"- Agreement with original ABD: {align['original_abd']}/31.",
        f"- Agreement with correction: {align['correction']}/31.",
        f"- Different from both (all three labels differ): {align['neither_three_way_all_different']}/31.",
        f"- Independent review flags: {flags}/31.",
        f"- Independent labels: supported={labels['supported']}, partially_supported={labels['partially_supported']}, unsupported={labels['unsupported']}, contradicted={labels['contradicted']}, unreviewable={labels['unreviewable']}.",
        "", "These 31 were selected because the first two reviews disagreed; none of these rates estimates the 900-record population.", "",
    ]
    for x in rows:
        body += [
            f"## {x['original_audit_id']}", "",
            f"- Labels: `{x['original_abd_label']}` → `{x['correction_label']}` → `{x['independent_sol_label']}`",
            f"- Alignment: `{x['independent_alignment']}`",
            f"- Independent evidence: {x['independent_evidence_references']}",
            f"- Independent rationale: {x['independent_reason']}",
            f"- Explanation: {x['comparison_explanation_zh']}", "",
        ]
    write(OUT / "POST_FREEZE_COMPARISON_REPORT.md", "\n".join(body))

    recommendation = """# Recommendation for the 116 correction targets

Do not bulk-merge the 116 correction labels. On this deliberately selected 31-disagreement set, the
independent Sol review agrees with correction on 16, with original ABD on 5, and with neither on 10;
19/31 remain review-flagged. This shows that correction is closer to the new execution on this subset,
but does not validate its 116 target decisions or estimate performance over all 900 routes.

If every target is to receive a defensible replacement label, the concrete minimum is an independent
blinded review of all 116 under the now-frozen Sol instructions and presentation, followed by explicit
adjudication wherever that review and correction disagree. A smaller stratified sample can diagnose
likely drift, but cannot justify per-record replacement for the unreviewed remainder. Priority strata
should include correction review-flag records, duration/count/order/non-occurrence questions, exhaustive
list options, and opaque image-path options. This is a recommendation only; no expanded review begins here.
"""
    write(OUT / "TARGET_116_HANDLING_RECOMMENDATION.md", recommendation)

    generated = ["THREE_WAY_COMPARISON.csv", "POST_FREEZE_COMPARISON_REPORT.md", "TARGET_116_HANDLING_RECOMMENDATION.md"]
    manifest = {
        "schema_version": "abd_disagreement_independent_post_freeze_comparison_v1",
        "blind_freeze_sha256": sha(FREEZE),
        "blinded_reviews_sha256": sha(REVIEWS),
        "generated_files": {x: sha(OUT / x) for x in generated},
        "agreement_original_abd": align["original_abd"],
        "agreement_correction": align["correction"],
        "three_way_all_different": align["neither_three_way_all_different"],
        "independent_review_flags": flags,
        "gold_read": False,
        "correctness_read": False,
        "labels_overwritten_or_merged": False,
        "external_api_calls": 0,
    }
    write(OUT / "POST_FREEZE_COMPARISON_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
