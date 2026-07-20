"""Deterministically freeze ten EgoSchema videos using segmentation statistics only."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/manifests/egoschema_comparison_pilot.json"
TARGET = ROOT / "data/manifests/coarse_segmentation_3way_10.json"


def build_manifest(project_root: Path = ROOT) -> dict[str, Any]:
    source = json.loads(
        (project_root / "data/manifests/egoschema_comparison_pilot.json").read_text(
            encoding="utf-8"
        )
    )
    rows = []
    for case in source["cases"]:
        video_id = str(case["video_id"])
        index_path = project_root / "outputs/visual_index" / video_id / "visual_state_regions.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        duration = float(case["duration_sec"])
        regions = index["visual_state_regions"]
        maximum = max(float(row["end_time"]) - float(row["start_time"]) for row in regions)
        rows.append(
            {
                "video_id": video_id,
                "case_id": str(case["case_id"]),
                "q_uid": str(case["case_id"]),
                "video_path": str(case["video_path"]),
                "duration": duration,
                "existing_current_region_count": len(regions),
                "current_max_region_duration": maximum,
                "current_largest_region_ratio": maximum / duration,
            }
        )
    median = statistics.median(row["current_largest_region_ratio"] for row in rows)
    problematic = sorted(
        rows, key=lambda row: (-row["current_largest_region_ratio"], row["video_id"])
    )[:5]
    problematic_ids = {row["video_id"] for row in problematic}
    controls = sorted(
        (row for row in rows if row["video_id"] not in problematic_ids),
        key=lambda row: (
            abs(row["current_largest_region_ratio"] - median),
            row["current_largest_region_ratio"],
            row["video_id"],
        ),
    )[:5]
    selected = []
    for rank, row in enumerate(problematic, start=1):
        selected.append(
            {
                **row,
                "group": "problematic",
                "selection_rank": rank,
                "selection_reason": "ranked among five highest current largest_region_ratio values",
            }
        )
    for rank, row in enumerate(controls, start=1):
        selected.append(
            {
                **row,
                "group": "control",
                "selection_rank": rank,
                "selection_reason": (
                    "ranked among five absolute-closest to the 25-video median "
                    "largest_region_ratio after excluding problematic videos"
                ),
                "distance_from_pilot_median_ratio": abs(
                    row["current_largest_region_ratio"] - median
                ),
            }
        )
    return {
        "manifest_version": "coarse-segmentation-3way-10-v1",
        "source_manifest": "data/manifests/egoschema_comparison_pilot.json",
        "selection_uses_only_segmentation_statistics": True,
        "qa_gold_used": False,
        "qa_correctness_used": False,
        "source_video_count": len(rows),
        "selected_video_count": len(selected),
        "source_median_largest_region_ratio": median,
        "selection_policy": {
            "problematic": "top five largest current largest_region_ratio; ties by video_id",
            "control": "five closest to source median; ties by ratio then video_id",
        },
        "videos": selected,
    }


def main() -> int:
    manifest = build_manifest()
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": TARGET.as_posix(), "videos": len(manifest["videos"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
