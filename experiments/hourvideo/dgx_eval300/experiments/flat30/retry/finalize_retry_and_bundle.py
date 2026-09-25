#!/usr/bin/env python3
"""Merge only valid retry completions, summarize recovery, and package the frozen runtime snapshot."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/naxucl/data/HourVideo/videoseal_original/eval300_timeout_retry_20260819T084545Z")
BASE = Path("/home/naxucl/data/HourVideo/videoseal_original")
FROZEN_UIDS = Path("/home/naxucl/data/HourVideo/benchmark/v1.0_release/hourvideo_eval300_v1_uids.txt")
SNAPSHOT = ROOT / ".runtime_snapshot"
GROUPS = {
    "original": {"source": BASE / "runs_dgx_eval300_v1", "retry": ROOT / "original_retry"},
    "trained": {"source": BASE / "runs_dgx_eval300_videoseal8b_v1", "retry": ROOT / "trained_retry"},
}
LEGAL = re.compile(r"^[A-E](?:[\s,]*[A-E])*$")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_json(root: Path, pattern: str) -> dict[str, tuple[Path, dict]]:
    result: dict[str, tuple[Path, dict]] = {}
    for path in root.glob(pattern):
        try:
            item = read_json(path)
        except Exception:
            continue
        uid = str(item.get("uid") or "")
        if uid:
            result[uid] = (path, item)
    return result


def index_trajectories(root: Path) -> dict[str, list[tuple[Path, dict]]]:
    result: dict[str, list[tuple[Path, dict]]] = {}
    for path in root.glob("*/*/trajectory.json"):
        try:
            item = read_json(path)
        except Exception:
            continue
        uid = str(item.get("uid") or "")
        if uid:
            result.setdefault(uid, []).append((path, item))
    for values in result.values():
        values.sort(key=lambda pair: str(pair[1].get("finished_at") or ""))
    return result


def valid_trajectory(values: list[tuple[Path, dict]]) -> tuple[Path, dict] | None:
    valid = []
    for path, item in values:
        answer = str(item.get("answer") or "").strip().upper()
        if item.get("finished_at") and isinstance(item.get("steps"), list) and item["steps"] and LEGAL.fullmatch(answer):
            valid.append((path, item))
    return valid[-1] if valid else None


def classify_termination(traj: dict) -> str:
    note = str(traj.get("note") or "").lower()
    steps = traj.get("steps") if isinstance(traj.get("steps"), list) else []
    if "forced full-video visual_inspect fallback" in note or any(
        isinstance(step.get("observation"), dict)
        and step["observation"].get("forced") is True
        and step["observation"].get("mode") == "full_video"
        for step in steps
        if isinstance(step, dict)
    ):
        return "full_video_64_frame_fallback"
    if len(steps) >= 15:
        step = steps[14]
        action = step.get("action") if isinstance(step, dict) else None
        if isinstance(action, dict) and action.get("name") == "visual_inspect":
            return "step15_forced_inspector_inferred"
    return "normal_end"


def merge_group(name: str, source: Path, retry: Path, frozen: list[str]) -> dict:
    source_metrics = index_json(source, "*/metrics/*.json")
    source_preds = index_json(source, "*/preds/*.json")
    source_traj = index_trajectories(source)
    retry_metrics = index_json(retry, "*/metrics/*.json")
    retry_preds = index_json(retry, "*/preds/*.json")
    retry_traj = index_trajectories(retry)

    out = ROOT / f"{name}_merged"
    if out.exists():
        raise RuntimeError(f"Refusing existing merged directory: {out}")
    out.mkdir()
    rows = []
    unresolved = []
    recovered = 0
    terminations: Counter[str] = Counter()
    retry_failures: Counter[str] = Counter()
    retry_elapsed = 0.0
    for _, metric in retry_metrics.values():
        elapsed = metric.get("elapsed_sec")
        if isinstance(elapsed, (int, float)):
            retry_elapsed += float(elapsed)
        if metric.get("status") != "success":
            retry_failures[str(metric.get("error_type") or metric.get("status") or "other")] += 1

    for uid in frozen:
        chosen = None
        origin = None
        source_status = source_metrics.get(uid, ({}, {}))[1].get("status")
        if source_status == "success":
            trajectory = valid_trajectory(source_traj.get(uid, []))
            if trajectory and uid in source_preds:
                chosen = (source_preds[uid], trajectory)
                origin = "original_success"
        else:
            retry_status = retry_metrics.get(uid, ({}, {}))[1].get("status")
            trajectory = valid_trajectory(retry_traj.get(uid, []))
            if retry_status == "success" and trajectory and uid in retry_preds:
                chosen = (retry_preds[uid], trajectory)
                origin = "retry_recovered"
                recovered += 1
        if chosen is None:
            unresolved.append(uid)
            continue
        (pred_path, pred), (traj_path, traj) = chosen
        prediction = str(pred.get("pred") or traj.get("answer") or "").strip().upper()
        ground_truth = str(pred.get("gt") or traj.get("groundtruth") or "").strip().upper()
        termination = classify_termination(traj)
        terminations[termination] += 1
        rows.append(
            {
                "uid": uid,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "correct": prediction == ground_truth,
                "source": origin,
                "termination": termination,
                "prediction_file": str(pred_path),
                "trajectory_file": str(traj_path),
            }
        )

    with (out / "merged_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "remaining_timeout_uids.txt").write_text("".join(f"{uid}\n" for uid in unresolved), encoding="utf-8")
    completed_before = sum(
        source_metrics.get(uid, ({}, {}))[1].get("status") == "success" for uid in frozen
    )
    correct = sum(row["correct"] for row in rows)
    summary = {
        "group": name,
        "eval300_uid_sha256": sha256(FROZEN_UIDS),
        "original_completed": completed_before,
        "original_timeout": len(frozen) - completed_before,
        "recovered": recovered,
        "remaining_timeout": len(unresolved),
        "merged_completed": len(rows),
        "merged_correct": correct,
        "merged_accuracy_completed_denominator": correct / len(rows) if rows else None,
        "merged_accuracy_eval300_denominator": correct / len(frozen),
        "termination_counts": dict(terminations),
        "retry_elapsed_sec_sum": retry_elapsed,
        "retry_failure_reasons": dict(retry_failures),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def build_bundle(summaries: dict[str, dict]) -> tuple[Path, str]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle_name = f"VideoSEAL_eval300_runtime_reference_{stamp}"
    staging = ROOT / bundle_name
    if staging.exists():
        raise RuntimeError(staging)
    shutil.copytree(SNAPSHOT / "source", staging / "source")
    shutil.copytree(SNAPSHOT / "metadata", staging / "metadata")
    (staging / "recovery_summaries.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    (staging / "environment.template").write_text(
        "# Supply values locally; never commit credentials.\n"
        "EMBEDDING_API_BASE=https://api.openai.com/v1\n"
        "EMBEDDING_MODEL=text-embedding-3-large\n"
        "EMBEDDING_API_KEY=\n",
        encoding="utf-8",
    )
    snapshot = read_json(SNAPSHOT / "metadata/snapshot.json")
    readme = f"""# VideoSEAL Eval300 runtime reference

This bundle contains only the frozen runtime-code import closure and DGX launch configuration used for Eval300 timeout recovery.

- Source repository: `{snapshot['source_repository']}`
- Branch: `{snapshot['branch']}`
- HEAD: `{snapshot['head']}`
- Frozen Eval300 UID SHA-256: `{sha256(FROZEN_UIDS)}`
- Runtime: concurrency=1, max_steps=16, task timeout=1000s, HTTP timeout=300s
- Planner: BF16, max-model-len=36864, max-num-seqs=1
- Visual: BF16, max-model-len=65536, max-num-seqs=1, image limit=64, enforce-eager

The target machine must provide datasets, videos, semantic indexes, model weights, and its own isolated Python/vLLM environments. Copy `source/` into a clean experimental workspace, inspect `metadata/runtime_dirty.patch`, and reproduce dependencies with `uv sync --extra dgx-runner` where compatible.

Excluded: secrets and `.env` files, datasets, videos, indexes, models, outputs, trajectories, caches, large logs, Conda environments.

The DGX does not have `/cs/student/project_msc` mounted, so filesystem sharing with the experiment host cannot be established from this machine. Use rsync unless the experiment host independently confirms a shared mount.
"""
    (staging / "README.md").write_text(readme, encoding="utf-8")
    transfer = f"""# Replace EXPERIMENT_HOST with the approved SSH host alias.
rsync -a --checksum --partial --info=progress2 \\
  {ROOT}/{bundle_name}.tar.gz \\
  EXPERIMENT_HOST:/cs/student/project_msc/ucemxna/

# If the target independently confirms the same shared filesystem:
cp -a {ROOT}/{bundle_name}.tar.gz \\
  /cs/student/project_msc/ucemxna/
"""
    (staging / "TRANSFER_COMMANDS.txt").write_text(transfer, encoding="utf-8")

    manifest_lines = []
    for path in sorted(p for p in staging.rglob("*") if p.is_file()):
        manifest_lines.append(f"{sha256(path)}  {path.relative_to(staging)}")
    (staging / "MANIFEST.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    forbidden = re.compile(rb"sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}|BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY")
    for path in (p for p in staging.rglob("*") if p.is_file()):
        if forbidden.search(path.read_bytes()):
            raise RuntimeError(f"Potential secret in bundle: {path}")

    archive = ROOT / f"{bundle_name}.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(staging, arcname=bundle_name)
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getnames()
        if any(re.search(r"(?:^|/)(?:\.env(?:\.|$)|auth\.json$)|\.(?:mp4|safetensors)$|trajectory", item, re.I) for item in members):
            raise RuntimeError("Forbidden archive member detected")
    return archive, sha256(archive)


def main() -> None:
    continuation_path = ROOT / "trained_continuation_report.json"
    if not continuation_path.is_file():
        raise SystemExit("Trained continuation report missing; refusing merge/bundle.")
    continuation = read_json(continuation_path)
    if continuation.get("expected") != 63 or continuation.get("resolved", 0) + continuation.get("unresolved", 0) != 63:
        raise SystemExit("Trained continuation did not account for exactly 63 UIDs; refusing merge/bundle.")
    if not SNAPSHOT.is_dir():
        raise SystemExit("Runtime snapshot missing")
    frozen = [line.strip() for line in FROZEN_UIDS.read_text().splitlines() if line.strip()]
    if len(frozen) != 300 or len(set(frozen)) != 300:
        raise SystemExit("Frozen UID list is not exactly 300 unique UIDs")
    summaries = {name: merge_group(name, **paths, frozen=frozen) for name, paths in GROUPS.items()}

    for name, summary in summaries.items():
        (ROOT / f"{name}_merged_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        merged_rows = [json.loads(line) for line in (ROOT / f"{name}_merged/merged_results.jsonl").read_text().splitlines() if line]
        (ROOT / f"{name}_successfully_merged_uids.txt").write_text(
            "".join(f"{row['uid']}\n" for row in merged_rows), encoding="utf-8"
        )
        shutil.copy2(ROOT / f"{name}_merged/remaining_timeout_uids.txt", ROOT / f"{name}_remaining_timeout_uids.txt")

    retry_items = []
    for name, paths in GROUPS.items():
        metrics = index_json(paths["retry"], "*/metrics/*.json")
        for uid in sorted(metrics):
            _, item = metrics[uid]
            retry_items.append({"group": name, "uid": uid, "status": item.get("status"), "error_type": item.get("error_type"), "elapsed_sec": item.get("elapsed_sec")})
    with (ROOT / "retry_item_status.jsonl").open("w", encoding="utf-8") as handle:
        for item in retry_items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    retry_summary = {"description": "Retry-augmented results; not the original single-pass Eval300 result.", "original": summaries["original"], "trained": summaries["trained"], "trained_continuation": {key: continuation.get(key) for key in ("expected", "resolved", "unresolved")}}
    (ROOT / "retry_summary.json").write_text(json.dumps(retry_summary, indent=2) + "\n", encoding="utf-8")
    report_md = f"""# Eval300 timeout recovery report

These are retry-augmented results and must not be described as the original single-pass Eval300 results.

## Original Planner
- First pass: 228 completed + 72 timeout
- Retry recovered: {summaries['original']['recovered']}
- Retry still timeout/incomplete: {summaries['original']['remaining_timeout']}
- Retry-augmented: {summaries['original']['merged_completed']} completed + {summaries['original']['remaining_timeout']} timeout/incomplete
- Retry-augmented accuracy (300 denominator): {summaries['original']['merged_accuracy_eval300_denominator']:.6f}

## Trained Planner
- First pass: 235 completed + 65 timeout
- Retry recovered: {summaries['trained']['recovered']}
- Retry still timeout/incomplete: {summaries['trained']['remaining_timeout']}
- Retry-augmented: {summaries['trained']['merged_completed']} completed + {summaries['trained']['remaining_timeout']} timeout/incomplete
- Retry-augmented accuracy (300 denominator): {summaries['trained']['merged_accuracy_eval300_denominator']:.6f}

Only retries with status=success, a finished trajectory, non-empty steps, and a legal A-E prediction were merged.
"""
    (ROOT / "RETRY_REPORT.md").write_text(report_md, encoding="utf-8")
    archive, digest = build_bundle(summaries)
    final = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "summaries": summaries, "bundle": str(archive), "bundle_sha256": digest}
    (ROOT / "FINAL_REPORT.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
