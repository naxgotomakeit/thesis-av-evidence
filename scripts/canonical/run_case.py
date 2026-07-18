from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.canonical_pipeline.runner import CanonicalOnlineRunner  # noqa: E402
from src.canonical_pipeline.state import ExecutionMode  # noqa: E402
from src.canonical_pipeline.versions import load_canonical_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one canonical baseline case")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--execute-live", action="store_true", help="Enable explicitly injected external-call boundaries")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.execute_live:
        raise SystemExit("Live execution is gated: use the canonical Python API with explicit planner, fallback and Gemini clients after regression authorization")
    state = CanonicalOnlineRunner(load_canonical_config(ROOT)).run_case(args.case_id, mode=ExecutionMode.REGRESSION_REPLAY)
    print(json.dumps({"case_id": state.case_id, "mode": state.mode.value, "fallback_execution_count": state.fallback_execution_count, "retained_evidence_ids": [item["candidate_id"] for item in state.evidence_packet["retained_candidates"]], "preflight_status": state.preflight_result["status"], "external_calls": state.external_calls}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
