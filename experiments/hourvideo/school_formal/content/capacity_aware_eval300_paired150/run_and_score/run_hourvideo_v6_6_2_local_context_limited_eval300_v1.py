from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.hourvideo_v6_6_2_local_context_limited_eval300_v1.formal_runtime import (  # noqa: E402
    build_static_closure,
    generate_planners,
    initialize_status_population,
    run_formal,
    run_gate,
    verify_frozen_inputs,
    write_metric_parity_matrix,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/hourvideo_v6_6_2_local_context_limited_eval300_v1.json")
    parser.add_argument("--stage", required=True, choices=("preflight", "planner", "gate", "formal"))
    parser.add_argument("--question-id")
    args = parser.parse_args()
    config = ROOT / args.config
    if args.stage == "preflight":
        result = {
            "frozen_inputs": verify_frozen_inputs(ROOT, config),
            "status_population": initialize_status_population(ROOT, config),
            "metric_parity": write_metric_parity_matrix(ROOT, config),
            "static_closure": build_static_closure(ROOT, config),
        }
    elif args.stage == "planner":
        result = generate_planners(ROOT, config, args.question_id)
    elif args.stage == "gate":
        result = run_gate(ROOT, config)
    else:
        result = run_formal(ROOT, config)
    print(json.dumps({"stage": args.stage, "result": result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
