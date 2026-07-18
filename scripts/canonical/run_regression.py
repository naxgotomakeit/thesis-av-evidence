from __future__ import annotations

import hashlib
import html
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.canonical_pipeline.regression import compare_state  # noqa: E402
from src.canonical_pipeline.runner import CanonicalOnlineRunner  # noqa: E402
from src.canonical_pipeline.state import ExecutionMode  # noqa: E402
from src.canonical_pipeline.versions import load_canonical_config  # noqa: E402


CASES = ("00006_3", "00061_5")
LOGGER = logging.getLogger("canonical_pipeline.regression")


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    return {row["case_id"]: row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tracked_hashes(config) -> dict[str, str]:
    paths = [config.path(name) for name in ("planner_plans", "task5b_frozen", "task5c_frozen", "task6_frozen", "task7a_frozen")]
    task7b = config.path("task7b_v3_frozen")
    paths.extend(path for path in task7b.rglob("*") if path.is_file())
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def architecture_audit(config) -> dict[str, Any]:
    return {
        "canonical_version_map": config.versions,
        "historical_structure": ["Task5A v2 serialized output", "Task5B base then v1.1 packaging correction", "Task5C v1 then v1.1 then v1.2 correction files", "Task6 v1 then v1.1 then v1.2 correction files", "Task7A payload file", "Task7B historical disk label v0_3"],
        "canonical_runtime": ["planner v2 once", "Task5B v1.1 retrieval once", "Task5C v1.2 final sufficiency plus at-most-one fallback", "Task6 v1.2 final packet once", "Task7A v1 payload/preflight once", "Task7B logical v3 once in live mode", "local validation once"],
        "historical_versions_as_runtime_stages": [],
        "per_stage_sources": config.raw["source_lineage"],
        "serialized_intermediate_files_required_between_stages": False,
        "regression_injections": ["Task5A v2 frozen planner record", "Task5C v1.2 frozen fallback evidence when fallback is required"],
        "fallback_execution_location": "src/canonical_pipeline/sufficiency.py::run_evidence_sufficiency",
        "final_evidence_location": "src/canonical_pipeline/reranking.py::build_evidence_packet",
        "final_payload_location": "src/canonical_pipeline/payload.py::build_final_payload",
        "task7b_logical_to_disk_mapping": config.raw["historical_disk_mapping"],
    }


def make_html(version_map: dict[str, Any], audit: dict[str, Any], results: list[dict[str, Any]]) -> str:
    pre = lambda value: f"<pre>{html.escape(json.dumps(value, ensure_ascii=False, indent=2))}</pre>"
    cards = "".join(f"<article><h2>{item['case_id']}</h2><p>Classification: <b>{item['overall_classification']}</b></p>{pre(item)}</article>" for item in results)
    return f"""<!doctype html><meta charset=utf-8><title>Canonical baseline regression</title><style>body{{font:15px system-ui;margin:2rem;max-width:1450px}}pre{{white-space:pre-wrap;background:#f4f7f9;padding:1rem}}article{{border-top:4px solid #54708b;margin-top:2rem}}</style><h1>Baseline v1 canonical executable system</h1><h2>Canonical version map</h2>{pre(version_map)}<h2>Historical execution vs canonical runtime</h2>{pre(audit)}<h2>Regression comparisons</h2>{cards}<h2>Live execution safety</h2><p>{'Safe to proceed to a separately authorized live verification.' if all(item['live_safe_to_proceed'] for item in results) else 'Blocked by research-behavior mismatch.'}</p>"""


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_canonical_config(ROOT)
    out = config.path("canonical_output")
    before = tracked_hashes(config)
    frozen_maps = {name: read_jsonl(config.path(path_name)) for name, path_name in (("planner", "planner_plans"), ("retrieval", "task5b_frozen"), ("sufficiency", "task5c_frozen"), ("packet", "task6_frozen"), ("payload", "task7a_frozen"))}
    runner = CanonicalOnlineRunner(config)
    results = []
    for case_id in CASES:
        LOGGER.info("Running zero-call canonical regression for %s", case_id)
        state = runner.run_case(case_id, mode=ExecutionMode.REGRESSION_REPLAY)
        comparison = compare_state(state, {name: rows[case_id] for name, rows in frozen_maps.items()})
        results.append(comparison)
        write_json(out / f"regression_{case_id}.json", comparison)
    after = tracked_hashes(config)
    if before != after:
        raise RuntimeError("Frozen canonical upstream outputs changed during regression")
    audit = architecture_audit(config)
    write_json(out / "canonical_version_map.json", config.raw)
    write_json(out / "architecture_audit.json", audit)
    summary = {"baseline": "Baseline v1 canonical executable system", "mode": "regression_replay", "cases": list(CASES), "external_calls": 0, "case_results": results, "research_behavior_mismatch_count": sum(bool(item["differences"]) for item in results), "fallback_at_most_once": all(item["fallback_execution_count"] <= 1 for item in results), "historical_patch_scripts_executed_as_stages": False, "live_safe_to_proceed": all(item["live_safe_to_proceed"] for item in results), "frozen_upstream_integrity": {"before": before, "after": after, "unchanged": True}}
    write_json(out / "regression_summary.json", summary)
    lines = ["# Canonical pipeline regression", "", f"- Cases: {', '.join(CASES)}", "- External calls: 0", f"- Research-behavior mismatches: {summary['research_behavior_mismatch_count']}", f"- Fallback at most once: {summary['fallback_at_most_once']}", f"- Historical patch scripts executed as stages: {summary['historical_patch_scripts_executed_as_stages']}", f"- Live verification safe to proceed: {summary['live_safe_to_proceed']}", f"- Frozen upstream unchanged: {summary['frozen_upstream_integrity']['unchanged']}"]
    for item in results:
        lines.append(f"- {item['case_id']}: {item['overall_classification']} ({item['exact_check_count']}/{item['check_count']} checks)")
    (out / "regression_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "canonical_pipeline_review.html").write_text(make_html(config.raw, audit, results), encoding="utf-8")
    print(json.dumps({"cases": list(CASES), "external_calls": 0, "mismatches": summary["research_behavior_mismatch_count"], "live_safe_to_proceed": summary["live_safe_to_proceed"], "output": str(out)}, ensure_ascii=False, indent=2))
    return 0 if summary["live_safe_to_proceed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

