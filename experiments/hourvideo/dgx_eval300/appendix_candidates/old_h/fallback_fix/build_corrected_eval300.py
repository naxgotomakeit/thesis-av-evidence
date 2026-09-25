#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORMAL = Path("/home/naxucl/data/HourVideo/experiments/hourvideo_v7_4_variant_c_budgets_v1/outputs/formal_eval300_20260821T095418Z")
ORCH = Path("/home/naxucl/data/HourVideo/experiments/hourvideo_v7_4_variant_c_budgets_v1/formal_eval300_orchestrator")
OUT = HERE / "corrected_eval300"
sys.path.insert(0, str(ORCH))
from finalize_profile import collect, summarize  # noqa: E402


REPLACEMENTS = [
    ("h6", "h_6", "4572b198-2c1c-4920-bcf0-95fcebe12261_7_32", "first", "first_pass"),
    ("h6", "h_6", "115774b6-534d-444f-b7aa-d1b834eb0ee7_1_1", "first", "first_pass"),
    ("h6", "h_6", "115774b6-534d-444f-b7aa-d1b834eb0ee7_12_27", "first", "first_pass"),
    ("h6", "h_6", "115774b6-534d-444f-b7aa-d1b834eb0ee7_17_7", "first", "first_pass"),
    ("h6", "h_6", "7e512589-aa97-41e8-83d3-af2e83e4fd06_7_20", "first", "first_pass"),
    ("h6", "h_6", "d3a0899e-2093-454c-9f65-30087883193a_17_13", "first", "first_pass"),
    ("h6", "h_6", "7ddbf8a2-5b3e-44cd-b9dc-db17ec06831b_17_15", "first", "first_pass"),
    ("h6", "h_6", "115774b6-534d-444f-b7aa-d1b834eb0ee7_7_10", "retry", "retry_1"),
    ("h15", "h_15", "115774b6-534d-444f-b7aa-d1b834eb0ee7_17_7", "first", "acceptance_v2"),
    ("h15", "h_15", "d3a0899e-2093-454c-9f65-30087883193a_22_14", "first", "first_pass"),
    ("h15", "h_15", "db3f7933-dfa0-4678-9d4f-393b628ded45_21_27", "first", "first_pass"),
    ("h30", "h_30", "41a86310-2cc1-48f9-b5b5-6b495a95fbac_23_8", "first", "first_pass"),
    ("h30", "h_30", "819c8af7-851f-434f-ab32-318285bc54b1_5_25", "retry", "retry_1"),
]


def replacement_root(profile_dir: str, phase: str) -> Path:
    if phase == "acceptance_v2":
        return HERE / "reruns" / "h_15" / "acceptance_v2" / "runs"
    return HERE / "reruns" / profile_dir / phase


def accuracy(rows: list[dict]) -> dict:
    completed = [r for r in rows if r["complete"]]
    correct = sum(r["complete"] and r["pred"] == r["gt"] and bool(r["gt"]) for r in rows)
    timeout = sum(r.get("status") == "timeout" or r.get("error_type") == "timeout" for r in rows)
    invalid = sum(r.get("status") == "success" and r.get("pred") not in set("ABCDE") for r in rows)
    return {
        "denominator": len(rows),
        "strict_completed": len(completed),
        "correct": correct,
        "accuracy_over_all": correct / len(rows),
        "completed_only_accuracy": correct / len(completed) if completed else None,
        "timeout": timeout,
        "status_success_but_strict_invalid": invalid,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    targets = [
        OUT / "replacement_manifest.tsv", OUT / "final_selected_attempts.jsonl",
        OUT / "accuracy_summary.json", OUT / "efficiency_summary.json",
        OUT / "before_after_comparison.md",
    ]
    if any(path.exists() for path in targets):
        raise SystemExit("refusing to overwrite existing Corrected Eval300 outputs")

    original: dict[str, list[dict]] = {}
    corrected: dict[str, list[dict]] = {}
    replacement_rows = []
    replacement_records = []
    mapping = {(p, uid): (pdir, attempt, phase) for p, pdir, uid, attempt, phase in REPLACEMENTS}
    for profile in ("h6", "h15", "h30"):
        rows = json.loads((FORMAL / profile / "merged" / "per_question_manifest.json").read_text(encoding="utf-8"))
        if len(rows) != 300 or len({r["uid"] for r in rows}) != 300:
            raise SystemExit(f"invalid original manifest for {profile}")
        original[profile] = rows
        out_rows = []
        for old in rows:
            key = (profile, old["uid"])
            if key not in mapping:
                out_rows.append(dict(old))
                continue
            profile_dir, old_attempt, phase = mapping[key]
            old_path = str(old.get("trajectory_path") or old.get("prediction_path") or "")
            expected = "/retry_1/" if old_attempt == "retry" else "/first_pass/"
            if expected not in old_path:
                raise SystemExit(f"old selected attempt mismatch for {profile} {old['uid']}: {old_path}")
            root = replacement_root(profile_dir, phase)
            new = collect(root, old["uid"])
            out_rows.append(new)
            replacement_records.append(new)
            replacement_rows.append({
                "profile": profile,
                "uid": old["uid"],
                "old_attempt": old_attempt,
                "old_trajectory": old.get("trajectory_path") or "",
                "old_prediction": old.get("prediction_path") or "",
                "new_run_id": Path(str(new["trajectory_path"])).parent.name,
                "new_trajectory": new["trajectory_path"],
                "new_prediction": new["prediction_path"],
                "selection_action": "replace_corresponding_old_attempt",
            })
        corrected[profile] = out_rows

    if len(replacement_rows) != 13:
        raise SystemExit(f"expected 13 replacements, got {len(replacement_rows)}")

    with (OUT / "replacement_manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(replacement_rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(replacement_rows)

    with (OUT / "final_selected_attempts.jsonl").open("w", encoding="utf-8") as handle:
        for profile in ("h6", "h15", "h30"):
            replaced = {uid for p, uid in mapping if p == profile}
            for row in corrected[profile]:
                item = dict(row); item["profile"] = profile
                item["corrected_replacement"] = row["uid"] in replaced
                handle.write(json.dumps(item, sort_keys=True) + "\n")

    accuracy_payload = {
        "view": "Corrected Eval300; replacement, not additive",
        "profiles": {
            p: {"before": accuracy(original[p]), "after": accuracy(corrected[p])}
            for p in ("h6", "h15", "h30")
        },
    }
    (OUT / "accuracy_summary.json").write_text(json.dumps(accuracy_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    extra_elapsed = [float(r["elapsed_sec"]) for r in replacement_records if r.get("elapsed_sec") is not None]
    efficiency_payload = {
        "view": "final selected attempts only; old replaced attempt costs are not added",
        "profiles": {
            p: {"before": summarize(original[p]), "after": summarize(corrected[p])}
            for p in ("h6", "h15", "h30")
        },
        "bug_correction_extra_compute": {
            "attempts": len(replacement_records),
            "attempt_elapsed_sec_total": sum(extra_elapsed),
            "attempt_elapsed_sec_mean": statistics.mean(extra_elapsed) if extra_elapsed else None,
            "excluded_from_corrected_method_efficiency": True,
        },
    }
    (OUT / "efficiency_summary.json").write_text(json.dumps(efficiency_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Eval300 fallback correction: before/after", "",
        "The corrected view replaces 13 selected attempts one-for-one. Original attempts and their costs remain preserved but are not added to corrected method efficiency.", "",
        "| Profile | Completed before→after | Correct before→after | Timeout before→after | Strict invalid before→after |", "|---|---:|---:|---:|---:|",
    ]
    for p in ("h6", "h15", "h30"):
        a, b = accuracy(original[p]), accuracy(corrected[p])
        lines.append(f"| {p.upper()} | {a['strict_completed']}→{b['strict_completed']} | {a['correct']}→{b['correct']} | {a['timeout']}→{b['timeout']} | {a['status_success_but_strict_invalid']}→{b['status_success_but_strict_invalid']} |")
    lines.extend(["", "The invalid first H-15 acceptance is diagnostic-only and is not selected. The accepted H-15 run is `20260826T164103Z-ge2b`.", ""])
    (OUT / "before_after_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "PASS", "replacements": 13, "records": 900}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
