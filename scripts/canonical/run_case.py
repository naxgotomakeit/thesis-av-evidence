from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.canonical_pipeline.fingerprint import build_run_fingerprint  # noqa: E402
from src.canonical_pipeline.runner import CanonicalOnlineRunner  # noqa: E402
from src.canonical_pipeline.state import ExecutionMode  # noqa: E402
from src.canonical_pipeline.versions import load_canonical_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one canonical baseline case")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--manifest", type=Path, help="Manifest containing the requested case; defaults to the frozen six-case manifest")
    parser.add_argument("--data-root", type=Path, help="Local EgoSound root; may also be supplied through EGOSOUND_DATA_ROOT")
    parser.add_argument("--execute-live", action="store_true", help="Enable explicitly injected external-call boundaries")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.execute_live:
        raise SystemExit("Live execution is gated: use the canonical Python API with explicit planner, fallback and Gemini clients after regression authorization")
    config = load_canonical_config(ROOT)
    manifest_path = args.manifest or config.path("case_manifest")
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    runner = CanonicalOnlineRunner(config, manifest_path=manifest_path, data_root=args.data_root)
    safe = runner.validate_case(args.case_id)
    fingerprint = build_run_fingerprint(
        ROOT,
        config,
        manifest_path,
        [safe["video_id"]],
        planner_model=os.environ.get("ANTHROPIC_MODEL"),
    )
    state = runner.run_case(args.case_id, mode=ExecutionMode.REGRESSION_REPLAY)
    print(json.dumps({"case_id": state.case_id, "mode": state.mode.value, "fallback_execution_count": state.fallback_execution_count, "retained_evidence_ids": [item["candidate_id"] for item in state.evidence_packet["retained_candidates"]], "preflight_status": state.preflight_result["status"], "external_calls": state.external_calls, "run_fingerprint_sha256": fingerprint["fingerprint_sha256"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
