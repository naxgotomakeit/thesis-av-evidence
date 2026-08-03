from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.planner_medium_retrieval.core import DEFAULT_CONFIG, run_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Planner + Medium retrieval v1 on frozen 226")
    parser.add_argument(
        "--index",
        type=Path,
        default=ROOT / "outputs/experiments/structured_organizer_v1/226/hierarchical_index_v1.json",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "configs/experiments/planner_medium_retrieval_v1/questions_226.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/experiments/planner_medium_retrieval_v1/retrieval_config.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/experiments/planner_medium_retrieval_v1",
    )
    parser.add_argument("--model-cache", type=Path, default=None)
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not configured; no API call was made.")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["config_version"] != DEFAULT_CONFIG["config_version"]:
        raise SystemExit(f"Unexpected config version: {config['config_version']}")
    if args.model_cache is not None:
        config["siglip"]["cache_dir"] = str(args.model_cache)
    result = run_experiment(
        root=ROOT,
        index_path=args.index.resolve(),
        questions_path=args.questions.resolve(),
        output_root=args.output.resolve(),
        api_key=api_key,
        config=config,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
