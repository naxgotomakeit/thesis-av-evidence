#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diagnostics.cradio_v4.option_semantic import run


def main() -> None:
    thesis_root = ROOT.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "config/data/egopolice_ablation_questions_v1.json")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "outputs/diagnostics/cradio_v4/YKI08_1fps_siglip2g_embeddings.npz")
    parser.add_argument("--prior-result", type=Path, default=ROOT / "outputs/diagnostics/cradio_v4/YKI08_cradio_v4_so400m_1fps_results.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/diagnostics/cradio_v4/option_semantic")
    parser.add_argument("--cache-root", type=Path, default=thesis_root / "models/diagnostic_caches/cradio_v4")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run(
        manifest_path=args.manifest,
        embedding_path=args.embeddings,
        prior_generic_result_path=args.prior_result,
        output_dir=args.output,
        cache_root=args.cache_root,
        device_name=args.device,
    )
    aggregate = result["aggregate"]
    print("option-semantic diagnostic complete")
    for key in (
        "generic_question_concrete_gt_cohort",
        "correct_option_semantic_concrete_gt_only",
        "all_options_balanced_concrete_gt_cohort",
    ):
        values = aggregate[key]
        hits = ", ".join(f"H@{k}={values['hit_at_k'][str(k)]['hits']}/{values['n']}" for k in (1, 5, 8, 10, 20))
        print(f"{key}: {hits}; median-rank={values['median_first_inside_rank']}")


if __name__ == "__main__":
    main()
