from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.fine_reranking.core import run_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline Fine reranking over frozen 226 Medium results")
    parser.add_argument(
        "--index",
        type=Path,
        default=ROOT / "outputs/experiments/structured_organizer_v1/226/hierarchical_index_v1.json",
    )
    parser.add_argument(
        "--planner-dir",
        type=Path,
        default=ROOT / "outputs/experiments/planner_medium_retrieval_v1/226",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/experiments/fine_reranking_v1/reranking_config.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/experiments/fine_reranking_v1",
    )
    args = parser.parse_args()
    result = run_experiment(
        root=ROOT,
        index_path=args.index.resolve(),
        planner_dir=args.planner_dir.resolve(),
        config_path=args.config.resolve(),
        output_root=args.output.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
