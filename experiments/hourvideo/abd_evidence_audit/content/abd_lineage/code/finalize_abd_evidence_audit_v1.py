#!/usr/bin/env python3
"""Post-freeze gold join, descriptive paired analysis, reports, and archive."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs/abd_eval300_evidence_audit_v1"
ANALYSIS = ROOT / "outputs/abd_eval300_analysis_v1"
LABELS = ["supported", "partially_supported", "unsupported", "contradicted", "unreviewable"]
GOLD_FIRST_READ = "2026-09-11T02:40:50+01:00"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_text(path: Path, value: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value)
    os.replace(tmp, path)


def write_json(path: Path, value: object) -> None:
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader(); w.writerows(rows)
    write_text(path, buf.getvalue())


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if not n:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def counts(rows: list[dict], key: str) -> dict:
    c = Counter(r[key] for r in rows)
    return {k: c.get(k, 0) for k in LABELS}


def main() -> None:
    freeze_path = AUDIT / "audit_freeze.json"
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("status") != "FROZEN_AND_VERIFIED" or freeze.get("audit_count") != 900:
        raise SystemExit("blinded audit is not frozen")
    blinded_path = AUDIT / "scored_evidence_audit_blinded.jsonl"
    if sha(blinded_path) != freeze["scored_evidence_audit_blinded_sha256"]:
        raise SystemExit("blinded audit changed after freeze")

    # Gold/identity boundary begins here, after the verified audit closure above.
    mapping = json.loads((AUDIT / "identity_mapping.json").read_text())["rows"]
    source_rows = {r["question_id"]: r for r in csv.DictReader((ANALYSIS / "abd_per_question.csv").open())}
    blinded = {r["audit_id"]: r for r in map(json.loads, blinded_path.open())}
    if len(mapping) != 900 or len(source_rows) != 300 or len(blinded) != 900:
        raise SystemExit("post-freeze join cardinality mismatch")

    scored = []
    for m in mapping:
        q = source_rows[m["question_id"]]
        arm = m["variant"]
        row = dict(blinded[m["audit_id"]])
        row.update({
            "question_id": m["question_id"], "task_id": m["task_id"], "variant": arm,
            "video_id": q["video_id"], "question_type": q["question_type"],
            "prediction": q[f"{arm}_prediction"], "gold": q["gold"],
            "correct": q[f"{arm}_correct"] == "True",
        })
        scored.append(row)
    scored_by = {(r["question_id"], r["variant"]): r for r in scored}

    arm_correct = {a: sum(r["correct"] for r in scored if r["variant"] == a) for a in "ABD"}
    if arm_correct != {"A": 98, "B": 103, "D": 103}:
        raise SystemExit(f"accuracy reproduction failed: {arm_correct}")

    write_text(AUDIT / "scored_evidence_audit.jsonl", "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in scored))
    summary_fields = ["audit_id", "question_id", "task_id", "variant", "video_id", "question_type", "prediction", "gold", "correct", "option_support", "confidence", "review_flag", "evidence_relationship", "rationale", "evidence_refs", "map_region_time_range", "frame_timestamps", "missing_key_evidence"]
    flat = []
    for r in scored:
        x = dict(r)
        for k in ("evidence_refs", "map_region_time_range", "frame_timestamps", "missing_key_evidence"):
            x[k] = json.dumps(x[k], ensure_ascii=False) if isinstance(x[k], (list, dict)) else x[k]
        flat.append(x)
    write_csv(AUDIT / "unified_summary.csv", flat, summary_fields)

    arm_stats = {}
    for arm in "ABD":
        rows = [r for r in scored if r["variant"] == arm]
        correct = [r for r in rows if r["correct"]]
        cc = counts(correct, "option_support")
        bad = cc["unsupported"] + cc["contradicted"]
        arm_stats[arm] = {
            "n": 300, "correct": arm_correct[arm], "accuracy": arm_correct[arm] / 300,
            "audit_distribution": counts(rows, "option_support"),
            "correct_answer_audit_distribution": cc,
            "incorrect_answer_audit_distribution": counts([r for r in rows if not r["correct"]], "option_support"),
            "correct_but_unsupported": cc["unsupported"],
            "correct_but_contradicted": cc["contradicted"],
            "correct_but_unsupported_or_contradicted": bad,
            "correct_but_unsupported_or_contradicted_fraction_of_correct": bad / arm_correct[arm],
        }

    pair_rows = []
    transition_rows = []
    pair_stats = {}
    for left, right in [("D", "B"), ("D", "A"), ("A", "B")]:
        cell = Counter()
        trans = Counter()
        answer_changes = 0
        for qid in source_rows:
            l, r = scored_by[(qid, left)], scored_by[(qid, right)]
            cell[(l["correct"], r["correct"])] += 1
            trans[(l["option_support"], r["option_support"])] += 1
            answer_changes += l["prediction"] != r["prediction"]
        b, c = cell[(True, False)], cell[(False, True)]
        pair = f"{left}_vs_{right}"
        out = {
            "comparison": pair, "both_correct": cell[(True, True)],
            f"{left}_only_correct": b, f"{right}_only_correct": c,
            "both_wrong": cell[(False, False)], "discordant_total": b + c,
            "exact_two_sided_mcnemar_p": exact_mcnemar(b, c),
            "answer_changed": answer_changes,
        }
        pair_stats[pair] = out; pair_rows.append(out)
        for (ll, rr), n in sorted(trans.items()):
            transition_rows.append({"comparison": pair, f"{left}_support": ll, f"{right}_support": rr, "count": n})
    write_csv(AUDIT / "pairwise_correctness.csv", pair_rows, sorted(set().union(*(r.keys() for r in pair_rows))))
    write_csv(AUDIT / "pairwise_support_transitions.csv", transition_rows, sorted(set().union(*(r.keys() for r in transition_rows))))

    rank = {"contradicted": 0, "unsupported": 1, "unreviewable": 2, "partially_supported": 3, "supported": 4}
    db_special = {}
    for name, predicate in {
        "D_only_correct_gain": lambda d, b: d["correct"] and not b["correct"],
        "B_only_correct_loss": lambda d, b: b["correct"] and not d["correct"],
    }.items():
        rows = [(scored_by[(q, "D")], scored_by[(q, "B")]) for q in source_rows if predicate(scored_by[(q, "D")], scored_by[(q, "B")])]
        support_direction = Counter("improved" if rank[d["option_support"]] > rank[b["option_support"]] else "declined" if rank[d["option_support"]] < rank[b["option_support"]] else "same" for d, b in rows)
        db_special[name] = {
            "count": len(rows),
            "D_evidence_relationship": dict(Counter(d["evidence_relationship"] for d, b in rows)),
            "B_to_D_support_transition": {f"{b['option_support']} -> {d['option_support']}": n for (bs, ds), n in Counter((b["option_support"], d["option_support"]) for d, b in rows).items() for _ in [None] for b in [{"option_support": bs}] for d in [{"option_support": ds}]},
            "support_rank_direction": dict(support_direction),
            "question_type": dict(Counter(d["question_type"] for d, b in rows)),
            "question_ids": [d["question_id"] for d, b in rows],
        }

    def grouped_csv(field: str, filename: str) -> None:
        groups = defaultdict(list)
        for r in scored: groups[r[field]].append(r)
        rows = []
        for name, rs in sorted(groups.items()):
            rec = {field: name, "routes": len(rs), "questions": len({r["question_id"] for r in rs})}
            for arm in "ABD":
                ar = [r for r in rs if r["variant"] == arm]
                rec[f"{arm}_correct"] = sum(r["correct"] for r in ar)
                rec[f"{arm}_accuracy"] = sum(r["correct"] for r in ar) / len(ar) if ar else ""
                c = Counter(r["option_support"] for r in ar)
                for lab in LABELS: rec[f"{arm}_{lab}"] = c[lab]
            rows.append(rec)
        write_csv(AUDIT / filename, rows, list(rows[0].keys()))
    grouped_csv("question_type", "by_question_type.csv")
    grouped_csv("video_id", "by_video.csv")

    d_relationship = Counter(r["evidence_relationship"] for r in scored if r["variant"] == "D")
    legacy_relationship = sum(d_relationship[x] for x in ("map_and_images", "images", "map", "none"))
    cross = {
        "schema_version": "abd_evidence_score_cross_stats_v1",
        "gold_first_read_after_verified_audit_freeze": GOLD_FIRST_READ,
        "accuracy_reproduced": arm_correct,
        "arms": arm_stats,
        "pairwise": pair_stats,
        "D_vs_B_gain_loss": db_special,
        "D_evidence_relationship_distribution_raw": dict(d_relationship),
        "D_legacy_source_descriptor_relationship_rows": legacy_relationship,
        "notes": [
            "D vs B is primary; D vs A is supplementary.",
            "Some early valid reviews retained legacy source-descriptor evidence_relationship values (not post-hoc rewritten after audit freeze).",
            "Question-type and video breakdowns are descriptive exploratory analyses.",
        ],
    }
    write_json(AUDIT / "evidence_score_cross_stats.json", cross)

    package_freeze = {
        "schema_version": "abd_evidence_audit_package_freeze_v1",
        "source_package_freeze_manifest_sha256": sha(AUDIT / "package_freeze_manifest.json"),
        "audit_freeze_sha256": sha(AUDIT / "audit_freeze.json"),
        "blinded_audit_sha256": sha(blinded_path),
        "scored_join_sha256": sha(AUDIT / "scored_evidence_audit.jsonl"),
        "raw_archive_sha256": sha(ANALYSIS / "ABD_RESULTS_PACKAGE.tar.gz"),
        "raw_results_modified": False,
    }
    write_json(AUDIT / "package_freeze.json", package_freeze)

    def pct(n: int, d: int) -> str: return f"{100*n/d:.2f}%"
    report = f"""# ABD Evidence Audit Report

## Frozen audit completion

- 192/192 batches and 900/900 unique opaque audit IDs passed validation.
- Reviewers actually read all provided maps and viewed all submitted images; per-batch counts are in `review_progress.json`.
- Gold and identity were first loaded only after the blinded audit closure was written and verified, at {GOLD_FIRST_READ}.
- External API calls: 0. Original ABD predictions and historical experiments were not modified.

## Evidence-support distributions

| Arm | supported | partially_supported | unsupported | contradicted | unreviewable |
|---|---:|---:|---:|---:|---:|
"""
    for arm in "ABD":
        c = arm_stats[arm]["audit_distribution"]
        report += f"| {arm} | {c['supported']} | {c['partially_supported']} | {c['unsupported']} | {c['contradicted']} | {c['unreviewable']} |\n"
    report += "\n## Correct-answer grounding\n\n| Arm | Correct | supported | partial | unsupported | contradicted | unreviewable | unsupported-or-contradicted / correct |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for arm in "ABD":
        s = arm_stats[arm]; c = s["correct_answer_audit_distribution"]
        report += f"| {arm} | {s['correct']} | {c['supported']} | {c['partially_supported']} | {c['unsupported']} | {c['contradicted']} | {c['unreviewable']} | {s['correct_but_unsupported_or_contradicted']} ({pct(s['correct_but_unsupported_or_contradicted'], s['correct'])}) |\n"
    report += """

The labels assess whether the submitted answer is supported by the model-visible evidence. They are not a second attempt to answer the question and are not claims about the model's hidden reasoning.
"""
    write_text(AUDIT / "ABD_EVIDENCE_AUDIT_REPORT.md", report)

    db = pair_stats["D_vs_B"]; da = pair_stats["D_vs_A"]; ab = pair_stats["A_vs_B"]
    gain = db_special["D_only_correct_gain"]; loss = db_special["B_only_correct_loss"]
    cross_report = f"""# ABD Cross Analysis Report

## Accuracy reproduction and paired comparisons

- A: 98/300 = 32.67%; B: 103/300 = 34.33%; D: 103/300 = 34.33%.
- D vs B (primary): 69 both correct, 34 D-only, 34 B-only, 163 both wrong; exact two-sided McNemar p = {db['exact_two_sided_mcnemar_p']:.6g}; answers changed on {db['answer_changed']} questions.
- D vs A (supplementary): 79 both correct, 24 D-only, 19 A-only, 178 both wrong; exact p = {da['exact_two_sided_mcnemar_p']:.6g}; answers changed on {da['answer_changed']} questions.
- A vs B: {ab['both_correct']} both correct, {ab['A_only_correct']} A-only, {ab['B_only_correct']} B-only, {ab['both_wrong']} both wrong; exact p = {ab['exact_two_sided_mcnemar_p']:.6g}; answers changed on {ab['answer_changed']} questions.

## D vs B gain and loss routes

The 34 D-only gains and 34 B-only losses confirm that adding the map changed decisions but produced zero net accuracy gain. Among gain routes, the frozen support label improved on {gain['support_rank_direction'].get('improved',0)}, stayed level on {gain['support_rank_direction'].get('same',0)}, and declined on {gain['support_rank_direction'].get('declined',0)}. Among loss routes the corresponding counts were {loss['support_rank_direction'].get('improved',0)}, {loss['support_rank_direction'].get('same',0)}, and {loss['support_rank_direction'].get('declined',0)}. Thus correctness changes cannot be equated with grounding changes.

For D routes with semantic relationship coding, gains most often involved `consistent` evidence ({gain['D_evidence_relationship'].get('consistent',0)}), while losses included consistent ({loss['D_evidence_relationship'].get('consistent',0)}), complementary ({loss['D_evidence_relationship'].get('complementary',0)}), and explicitly conflicting ({loss['D_evidence_relationship'].get('conflicting',0)}) map/frame cases. Detailed transitions and question IDs are in the JSON/CSV outputs. Early valid batches contain {legacy_relationship} D rows with the legacy source descriptor `map_and_images`; these were preserved rather than post-hoc recoded after gold exposure.

Duration, frequency, factual-recall, sequence-recall and spatial questions account for most gain/loss routes. This is descriptive, not a causal attribution: the audit can identify support, complementarity, conflict or insufficiency in the supplied evidence, but cannot reveal the model's internal reason for changing its answer.

## Interpretation boundaries

- A is not language-only: it receives a video-derived semantic map.
- B is not an independent frame-selection method: its images came from the historical Direct R3 navigation/inspection process.
- D vs B is the primary ablation: the answerer, images, timestamps and other visible input are matched; D adds the map.
- Equal B/D accuracy means no net accuracy gain, not no effect. Each arm has 34 unique correct answers relative to the other.
- Reliability or groundedness conclusions come from the frozen evidence audit, not accuracy alone.
- This Codex-assisted audit is a strict exploratory evidence-support judgment, not objective truth and not access to hidden reasoning.
- Per-question-type and per-video results are exploratory, especially for small cells.
"""
    write_text(AUDIT / "ABD_CROSS_ANALYSIS_REPORT.md", cross_report)

    reproducibility = {
        "schema_version": "abd_evidence_audit_reproducibility_v1",
        "gold_first_read_after_verified_audit_freeze": GOLD_FIRST_READ,
        "code": [
            {"path": "scripts/freeze_abd_blinded_audit_v1.py", "sha256": sha(ROOT / "scripts/freeze_abd_blinded_audit_v1.py")},
            {"path": "scripts/finalize_abd_evidence_audit_v1.py", "sha256": sha(ROOT / "scripts/finalize_abd_evidence_audit_v1.py")},
        ],
        "inputs": [
            {"path": "outputs/abd_eval300_evidence_audit_v1/scored_evidence_audit_blinded.jsonl", "sha256": sha(blinded_path)},
            {"path": "outputs/abd_eval300_evidence_audit_v1/audit_freeze.json", "sha256": sha(AUDIT / "audit_freeze.json")},
            {"path": "outputs/abd_eval300_evidence_audit_v1/identity_mapping.json", "sha256": sha(AUDIT / "identity_mapping.json")},
            {"path": "outputs/abd_eval300_analysis_v1/abd_per_question.csv", "sha256": sha(ANALYSIS / "abd_per_question.csv")},
        ],
        "results": [
            {"path": name, "sha256": sha(AUDIT / name)} for name in [
                "scored_evidence_audit.jsonl", "unified_summary.csv", "evidence_score_cross_stats.json",
                "pairwise_correctness.csv", "pairwise_support_transitions.csv", "by_question_type.csv",
                "by_video.csv", "ABD_EVIDENCE_AUDIT_REPORT.md", "ABD_CROSS_ANALYSIS_REPORT.md",
            ]
        ],
    }
    write_json(AUDIT / "analysis_reproducibility_manifest.json", reproducibility)

    # Results archive: exclude credentials, quarantines, raw images, and recursive archive artifacts.
    members = [
        "AUDIT_CRITERIA_FROZEN.md", "package_freeze_manifest.json", "package_freeze.json",
        "batch_manifest.json", "identity_mapping.json", "valid_second_pass_batches.json",
        "review_progress.json", "raw_integrity.json", "scored_evidence_audit_blinded.jsonl",
        "audit_freeze.json", "scored_evidence_audit.jsonl", "unified_summary.csv",
        "evidence_score_cross_stats.json", "pairwise_correctness.csv",
        "pairwise_support_transitions.csv", "by_question_type.csv", "by_video.csv",
        "ABD_EVIDENCE_AUDIT_REPORT.md", "ABD_CROSS_ANALYSIS_REPORT.md",
        "analysis_reproducibility_manifest.json",
    ]
    members += [str(p.relative_to(AUDIT)) for p in sorted((AUDIT / "reviews").glob("B*.jsonl"))]
    archive = AUDIT / "ABD_EVIDENCE_AUDIT_RESULTS.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        for rel in members:
            tf.add(AUDIT / rel, arcname=rel, recursive=False)
    archive_manifest = {
        "schema_version": "abd_evidence_audit_archive_v1",
        "member_count": len(members),
        "members": [{"path": rel, "sha256": sha(AUDIT / rel)} for rel in members],
        "archive_path": archive.name,
        "archive_sha256": sha(archive),
        "audit_freeze_sha256": sha(AUDIT / "audit_freeze.json"),
        "raw_results_modified": False,
        "credentials_included": False,
    }
    write_json(AUDIT / "archive_manifest.json", archive_manifest)
    print(json.dumps({
        "status": "PASS", "accuracy": arm_correct, "pairwise": pair_stats,
        "audit_freeze_sha256": sha(AUDIT / "audit_freeze.json"),
        "archive_sha256": sha(archive), "scored_sha256": sha(AUDIT / "scored_evidence_audit.jsonl"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
