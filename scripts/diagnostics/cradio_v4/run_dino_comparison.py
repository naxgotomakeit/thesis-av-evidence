#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diagnostics.cradio_v4.dino_comparison import run


def main() -> None:
    thesis_root = ROOT.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/visual_pipeline_v1.json")
    parser.add_argument("--video", type=Path, default=thesis_root / "data/EgoPolice_1.0.0/videos/pasadena/YKI08.mp4")
    parser.add_argument("--cradio-embeddings", type=Path, default=ROOT / "outputs/diagnostics/cradio_v4/YKI08_1fps_siglip2g_embeddings.npz")
    parser.add_argument("--cradio-result", type=Path, default=ROOT / "outputs/diagnostics/cradio_v4/YKI08_cradio_v4_so400m_1fps_results.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/diagnostics/cradio_v4/dino_comparison")
    parser.add_argument("--cache-root", type=Path, default=thesis_root / "models/diagnostic_caches/dinov2")
    args = parser.parse_args()
    result = run(
        config_path=args.config,
        video_path=args.video,
        cradio_embedding_path=args.cradio_embeddings,
        cradio_result_path=args.cradio_result,
        output_dir=args.output,
        cache_root=args.cache_root,
    )
    dino = result["representations"]["dinov2"]
    cradio = result["representations"]["cradio"]
    agreement = result["boundary_agreement"]["exact_frozen_threshold_outputs"]["2.0"]
    print(
        f"DINO Fine/Medium={dino['fine_metrics']['number_of_segments']}/"
        f"{dino['medium_duration_summary']['count']}"
    )
    print(
        f"C-RADIO Fine/Medium={cradio['fine_metrics']['number_of_segments']}/"
        f"{cradio['medium_duration_summary']['count']}"
    )
    print(f"Boundary agreement @2s: {agreement['matched_count']} matches, F1={agreement['f1']:.3f}")


if __name__ == "__main__":
    main()

