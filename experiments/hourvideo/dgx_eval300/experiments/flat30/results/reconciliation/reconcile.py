#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("/home/naxucl/data/HourVideo/videoseal_original")
RETRY_BASE = BASE / "eval300_timeout_retry_20260819T084545Z"
FORMAL = Path("/home/naxucl/data/HourVideo/experiments/hourvideo_v7_4_variant_c_budgets_v1/outputs/formal_eval300_20260821T095418Z")
MANIFEST = Path("/home/naxucl/data/HourVideo/benchmark/v1.0_release/hourvideo_eval300_v1_uids.txt")
OUT = Path(__file__).resolve().parent
STRICT = re.compile(r"^[A-E]$")
SNAPSHOT_UTC = "2026-08-22T14:42:51Z"

GROUPS = {
    "pretrained_flat": {
        "label": "训练前 Planner + VideoSEAL Flat",
        "first": BASE / "runs_dgx_eval300_v1",
        "retry": RETRY_BASE / "original_retry",
        "planned": RETRY_BASE / "manifests/original_timeout_uids.txt",
        "status": "FINAL",
    },
    "trained_flat": {
        "label": "训练后 Planner + VideoSEAL Flat",
        "first": BASE / "runs_dgx_eval300_videoseal8b_v1",
        "retry": RETRY_BASE / "trained_retry",
        "planned": RETRY_BASE / "manifests/trained_timeout_uids.txt",
        "status": "FINAL",
    },
    "pretrained_h6": {
        "label": "训练前 Planner + H-6 hierarchical indexer",
        "first": FORMAL / "h6/first_pass",
        "retry": FORMAL / "h6/retry_1",
        "planned": FORMAL / "h6/status/retry-pending-20260821T095441Z.txt",
        "status": "RETRY IN PROGRESS",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_uids(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def index_json(paths) -> tuple[dict[str, tuple[Path, dict]], list[str], list[str]]:
    out: dict[str, tuple[Path, dict]] = {}
    duplicates, parse_errors = [], []
    for path in sorted(paths):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            parse_errors.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        uid = str(item.get("uid") or "")
        if not uid:
            parse_errors.append(f"{path}: missing uid")
        elif uid in out:
            duplicates.append(uid)
        else:
            out[uid] = (path, item)
    return out, duplicates, parse_errors


def normalize(value) -> tuple[str, bool]:
    value = str(value if value is not None else "").strip().upper()
    return value, bool(STRICT.fullmatch(value))


def stage(root: Path) -> dict:
    metrics, metric_dups, metric_errors = index_json(root.glob("*/metrics/*.json"))
    preds, pred_dups, pred_errors = index_json(root.glob("*/preds/*.json"))
    trajectories, traj_dups, traj_errors = index_json(root.glob("*/*/trajectory.json"))
    # Multiple trajectory attempts can legitimately exist; index_json reports these
    # separately and scoring deliberately does not use trajectory answer text.
    return {
        "root": str(root), "metrics": metrics, "preds": preds,
        "trajectories": trajectories,
        "metric_duplicates": metric_dups, "prediction_duplicates": pred_dups,
        "trajectory_duplicate_uids": sorted(set(traj_dups)),
        "parse_errors": metric_errors + pred_errors + traj_errors,
    }


def raw_status(metric: dict | None) -> str:
    if metric is None:
        return "missing"
    status = str(metric.get("status") or "").lower()
    error = str(metric.get("error_type") or "").lower()
    if status == "success": return "success"
    if status == "timeout" or error == "timeout": return "timeout"
    return "error"


def lists_write(prefix: str, phase: str, categories: dict[str, list[str]]) -> None:
    for category, values in categories.items():
        (OUT / f"{prefix}_{phase}_{category}_uids.txt").write_text(
            "".join(f"{uid}\n" for uid in values), encoding="utf-8"
        )


def pct(n: int, d: int) -> str:
    return f"{n}/{d} = {100*n/d:.1f}%" if d else "N/A"


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if not n: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


canonical = read_uids(MANIFEST)
canonical_set = set(canonical)
if len(canonical) != 300 or len(canonical_set) != 300:
    raise SystemExit("canonical manifest is not 300 unique UIDs")

# Ground truth is taken from the explicit gt field in the complete original Flat
# prediction artifacts, then cross-checked against every other available pred.
gt_stage = stage(GROUPS["pretrained_flat"]["first"])
ground_truth = {}
for uid in canonical:
    item = gt_stage["preds"].get(uid)
    if not item:
        raise SystemExit(f"missing canonical ground truth artifact: {uid}")
    gt, valid = normalize(item[1].get("gt"))
    if not valid:
        raise SystemExit(f"invalid canonical gt {uid}: {gt!r}")
    ground_truth[uid] = gt

invalid_rows: list[dict] = []
provenance: list[dict] = []
summary = {
    "snapshot_utc": SNAPSHOT_UTC,
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "canonical_manifest": str(MANIFEST),
    "canonical_sha256": sha256(MANIFEST),
    "canonical_total": len(canonical),
    "canonical_duplicate_count": len(canonical) - len(canonical_set),
    "groups": {}, "paired": {},
}

for key, cfg in GROUPS.items():
    first, retry = stage(cfg["first"]), stage(cfg["retry"])
    planned = read_uids(cfg["planned"])
    if len(set(planned)) != len(planned):
        raise SystemExit(f"duplicate retry planned UID in {key}")
    if not set(planned) <= canonical_set:
        raise SystemExit(f"retry plan outside canonical in {key}")

    gt_mismatches = []
    for phase_name, st in (("first_pass", first), ("retry", retry)):
        for uid, (_, pred) in st["preds"].items():
            if uid in ground_truth and pred.get("gt") is not None:
                seen, seen_valid = normalize(pred.get("gt"))
                if not seen_valid or seen != ground_truth[uid]:
                    gt_mismatches.append({"uid": uid, "phase": phase_name, "gt": seen})

    def classify_stage(st: dict, universe: list[str], phase_name: str):
        cats = {x: [] for x in ("attempted", "status_success", "valid", "invalid", "timeout", "error", "missing")}
        correct, incorrect = [], []
        records = {}
        for uid in universe:
            metric_pair = st["metrics"].get(uid)
            metric = metric_pair[1] if metric_pair else None
            status = raw_status(metric)
            pred_pair = st["preds"].get(uid)
            raw_pred = pred_pair[1].get("pred") if pred_pair else None
            norm, valid = normalize(raw_pred)
            if metric is not None: cats["attempted"].append(uid)
            if status == "success":
                cats["status_success"].append(uid)
                if valid:
                    cats["valid"].append(uid)
                    (correct if norm == ground_truth[uid] else incorrect).append(uid)
                else:
                    cats["invalid"].append(uid)
                    invalid_rows.append({"uid": uid, "group": key, "phase": phase_name,
                                         "raw_prediction": raw_pred, "normalized_prediction": norm,
                                         "reason": "status-success prediction does not full-match ^[A-E]$"})
            elif status == "timeout": cats["timeout"].append(uid)
            elif status == "error": cats["error"].append(uid)
            else: cats["missing"].append(uid)
            # Scan every explicit illegal prediction, including timeout placeholders.
            if pred_pair and not valid and status != "success":
                invalid_rows.append({"uid": uid, "group": key, "phase": phase_name,
                                     "raw_prediction": raw_pred, "normalized_prediction": norm,
                                     "reason": f"illegal prediction present with raw status={status}; status bucket retained"})
            records[uid] = {"status": status, "raw_prediction": raw_pred, "normalized_prediction": norm,
                            "strict_valid": status == "success" and valid,
                            "correct": status == "success" and valid and norm == ground_truth[uid]}
        for values in cats.values(): values.sort()
        return cats, sorted(correct), sorted(incorrect), records

    first_cats, first_correct, first_incorrect, first_records = classify_stage(first, canonical, "first_pass")
    retry_cats, retry_correct, retry_incorrect, retry_records = classify_stage(retry, planned, "retry")
    lists_write(key, "first_pass", first_cats)
    lists_write(key, "retry", {**retry_cats, "pending": sorted(set(planned)-set(retry_cats["attempted"]))})

    final_cats = {x: [] for x in ("valid", "correct", "incorrect", "invalid", "timeout", "error", "missing", "unresolved")}
    selected = {}
    recovered = recovered_correct = 0
    for uid in canonical:
        f, r = first_records[uid], retry_records.get(uid)
        if f["strict_valid"]:
            source, selected_record = "first_pass", f
        elif r and r["strict_valid"]:
            source, selected_record = "retry", r
            recovered += 1
            recovered_correct += int(r["correct"])
        else:
            source, selected_record = "none", None
        if selected_record:
            final_cats["valid"].append(uid)
            bucket = "correct" if selected_record["correct"] else "incorrect"
            final_cats[bucket].append(uid)
            exclusion = ""
        else:
            # Final unresolved category prioritizes an attempted retry, then first-pass.
            terminal = r["status"] if r else f["status"]
            if terminal == "success": terminal = "invalid"
            final_cats[terminal].append(uid)
            final_cats["unresolved"].append(uid)
            exclusion = f"no strict-valid prediction; final raw status={terminal}"
        chosen = selected_record or (r if r else f)
        provenance.append({
            "uid": uid, "group": key, "selected_source": source,
            "raw_status": chosen["status"], "raw_prediction": chosen["raw_prediction"],
            "normalized_prediction": chosen["normalized_prediction"],
            "strict_valid": bool(selected_record), "ground_truth": ground_truth[uid],
            "correct": bool(selected_record and selected_record["correct"]),
            "exclusion_reason": exclusion,
        })
        selected[uid] = selected_record
    for values in final_cats.values(): values.sort()
    lists_write(key, "final", final_cats)

    first_stats = {
        "manifest_total": 300, "attempted": len(first_cats["attempted"]),
        "status_success": len(first_cats["status_success"]), "timeout": len(first_cats["timeout"]),
        "non_timeout_error": len(first_cats["error"]), "missing": len(first_cats["missing"]),
        "strict_valid": len(first_cats["valid"]), "invalid": len(first_cats["invalid"]),
        "correct": len(first_correct), "incorrect": len(first_incorrect),
        "completed_only_accuracy": len(first_correct)/len(first_cats["valid"]) if first_cats["valid"] else None,
        "full_300_accuracy": len(first_correct)/300,
    }
    pending = sorted(set(planned)-set(retry_cats["attempted"]))
    retry_stats = {
        "planned": len(planned), "attempted": len(retry_cats["attempted"]),
        "raw_status_success": len(retry_cats["status_success"]), "timeout": len(retry_cats["timeout"]),
        "non_timeout_error": len(retry_cats["error"]), "missing_not_yet_run": len(pending),
        "strict_valid": len(retry_cats["valid"]), "invalid": len(retry_cats["invalid"]),
        "correct": len(retry_correct), "incorrect": len(retry_incorrect),
    }
    final_stats = {
        "strict_valid": len(final_cats["valid"]), "invalid": len(final_cats["invalid"]),
        "timeout": len(final_cats["timeout"]), "error": len(final_cats["error"]),
        "missing_not_yet_run": len(final_cats["missing"]), "correct": len(final_cats["correct"]),
        "incorrect": len(final_cats["incorrect"]), "retry_valid_recovered": recovered,
        "retry_correct_recovered": recovered_correct,
        "completed_only_accuracy": len(final_cats["correct"])/len(final_cats["valid"]) if final_cats["valid"] else None,
        "full_300_accuracy": len(final_cats["correct"])/300,
    }
    summary["groups"][key] = {
        "label": cfg["label"], "status": cfg["status"], "first_pass_path": str(cfg["first"]),
        "retry_path": str(cfg["retry"]), "retry_plan_path": str(cfg["planned"]),
        "first_pass": first_stats, "retry": retry_stats, "final": final_stats,
        "integrity": {
            "retry_plan_unique": len(set(planned)) == len(planned),
            "retry_plan_outside_eval300": sorted(set(planned)-canonical_set),
            "retry_attempted_outside_plan": sorted(set(retry["metrics"])-set(planned)),
            "first_metric_outside_eval300": sorted(set(first["metrics"])-canonical_set),
            "first_metric_duplicates": first["metric_duplicates"],
            "first_prediction_duplicates": first["prediction_duplicates"],
            "retry_metric_duplicates": retry["metric_duplicates"],
            "retry_prediction_duplicates": retry["prediction_duplicates"],
            "gt_mismatches": gt_mismatches,
            "parse_errors": first["parse_errors"] + retry["parse_errors"],
            "separate_first_retry_roots": Path(first["root"]).resolve() != Path(retry["root"]).resolve(),
        },
        "_selected": {uid: rec for uid, rec in selected.items()},
    }

# Paired comparison is intentionally withheld while H-6 retry is active.
summary["paired"]["pretrained_flat_vs_pretrained_h6"] = {
    "status": "WITHHELD: H-6 retry is in progress at snapshot time",
    "flat_full_300": summary["groups"]["pretrained_flat"]["final"]["full_300_accuracy"],
    "h6_snapshot_full_300": summary["groups"]["pretrained_h6"]["final"]["full_300_accuracy"],
}

# Remove internal selected maps before serialization.
for value in summary["groups"].values(): value.pop("_selected", None)

with (OUT / "invalid_predictions.tsv").open("w", encoding="utf-8") as f:
    f.write("uid\tgroup\tphase\traw_prediction\tnormalized_prediction\treason\n")
    for row in sorted(invalid_rows, key=lambda x: (x["group"], x["phase"], x["uid"])):
        vals = [row[k] for k in ("uid", "group", "phase", "raw_prediction", "normalized_prediction", "reason")]
        f.write("\t".join(str(v if v is not None else "<MISSING>").replace("\t", "\\t").replace("\n", "\\n") for v in vals) + "\n")

with (OUT / "merge_provenance.tsv").open("w", encoding="utf-8") as f:
    keys = ("uid", "group", "selected_source", "raw_status", "raw_prediction", "normalized_prediction", "strict_valid", "ground_truth", "correct", "exclusion_reason")
    f.write("\t".join(keys) + "\n")
    for row in sorted(provenance, key=lambda x: (x["group"], canonical.index(x["uid"]))):
        f.write("\t".join(str(row[k] if row[k] is not None else "<MISSING>").replace("\t", "\\t").replace("\n", "\\n") for k in keys) + "\n")

(OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

lines = [
    "# HourVideo Eval300 reconciliation audit", "", f"Snapshot UTC: `{SNAPSHOT_UTC}`", "",
    "## Inputs", "", f"Canonical manifest: `{MANIFEST}`", f"SHA-256: `{sha256(MANIFEST)}`", "",
]
for key, cfg in GROUPS.items():
    lines += [f"### {cfg['label']}", "", f"- First pass: `{cfg['first']}`", f"- Retry: `{cfg['retry']}`", f"- Retry plan: `{cfg['planned']}`", ""]
lines += ["## Unified results", "",
          "| Group | First valid | First correct | First acc. | First timeout | Retry attempted | Retry valid recovered | Retry correct recovered | Retry timeout | Final valid | Final correct | Completed-only acc. | Full-300 acc. | Status |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
for key, g in summary["groups"].items():
    a, r, z = g["first_pass"], g["retry"], g["final"]
    lines.append(f"| {g['label']} | {a['strict_valid']} | {a['correct']} | {pct(a['correct'],a['strict_valid'])} | {a['timeout']} | {r['attempted']} | {z['retry_valid_recovered']} | {z['retry_correct_recovered']} | {r['timeout']} | {z['strict_valid']} | {z['correct']} | {pct(z['correct'],z['strict_valid'])} | {pct(z['correct'],300)} | {g['status']} |")
lines += ["", "## Detailed counts", ""]
for key, g in summary["groups"].items():
    a, r, z = g["first_pass"], g["retry"], g["final"]
    lines += [f"### {g['label']}", "",
              f"First pass: attempted={a['attempted']}, status-success={a['status_success']}, strict-valid={a['strict_valid']}, invalid={a['invalid']}, timeout={a['timeout']}, error={a['non_timeout_error']}, missing={a['missing']}, correct={a['correct']}, incorrect={a['incorrect']}; completed-only {pct(a['correct'],a['strict_valid'])}; full-set {pct(a['correct'],300)}.", "",
              f"Retry: planned={r['planned']}, attempted={r['attempted']}, raw-success={r['raw_status_success']}, strict-valid={r['strict_valid']}, invalid={r['invalid']}, timeout={r['timeout']}, error={r['non_timeout_error']}, pending={r['missing_not_yet_run']}, correct={r['correct']}, incorrect={r['incorrect']}.", "",
              f"Merged: valid={z['strict_valid']}, invalid={z['invalid']}, timeout={z['timeout']}, error={z['error']}, missing/pending={z['missing_not_yet_run']}, correct={z['correct']}, incorrect={z['incorrect']}; recovered-valid={z['retry_valid_recovered']}, recovered-correct={z['retry_correct_recovered']}; completed-only {pct(z['correct'],z['strict_valid'])}; full-set {pct(z['correct'],300)}. Status: **{g['status']}**.", ""]
lines += ["## Paired comparison", "", "Not frozen: H-6 retry was still running at the snapshot time. No paired UID list or McNemar result was generated. The full-300 snapshot values remain visible above.", "",
          "## Scoring rule", "", "Only the explicit prediction field, normalized with `str(value).strip().upper()`, is accepted when it fully matches `^[A-E]$`. Trajectory answer text is never used to replace it.", ""]
(OUT / "reconciliation_report.md").write_text("\n".join(lines), encoding="utf-8")

manifest_lines = []
for path in sorted(OUT.iterdir()):
    if path.name in {"MANIFEST.sha256", "reconcile.py"}: continue
    manifest_lines.append(f"{sha256(path)}  {path.name}")
(OUT / "MANIFEST.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
print(json.dumps({k: {"status":v["status"], "first":v["first_pass"], "retry":v["retry"], "final":v["final"]} for k,v in summary["groups"].items()}, indent=2))
