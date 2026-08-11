"""Compress navigation maps by keeping only essential fields for planner."""
from __future__ import annotations

import json
from pathlib import Path


def compress_navigation_map(map_doc: dict) -> dict:
    """Drop only fields verified as unread by v6.3's actual code path
    (_planner_map in hourvideo_r1_av_r3_2_ten_video_pilot_v1/live.py:77 and
    downstream core.py). Keeps exact_source_captions/exact_source_asr/
    audio_channel intact -- _planner_map serializes these as-is into the
    real planner prompt for R3.2/R1_AV respectively, so they are load-bearing,
    not dead weight.

    Dropped (confirmed unread anywhere in v6.3):
      top-level: storyline_events, has_storyline, boundary_policy
      per-region: uncertainty_notes, visual_structured_fallback, semantic_summary
    """
    compressed = {
        "map_type": map_doc.get("map_type"),
        "semantic_fields_available": map_doc.get("semantic_fields_available"),
        "hard_filtering_allowed": map_doc.get("hard_filtering_allowed"),
        "coarse_regions": [],
    }

    for coarse in map_doc.get("coarse_regions", []):
        compressed_coarse = {
            "coarse_id": coarse["coarse_id"],
            "start_sec": coarse["start_sec"],
            "end_sec": coarse["end_sec"],
            "source_medium_ids": coarse.get("source_medium_ids", []),
            "navigation_summary": coarse.get("navigation_summary", ""),
        }
        if "exact_source_captions" in coarse:
            compressed_coarse["exact_source_captions"] = coarse["exact_source_captions"]
        if "exact_source_asr" in coarse:
            compressed_coarse["exact_source_asr"] = coarse["exact_source_asr"]
        if "audio_channel" in coarse:
            compressed_coarse["audio_channel"] = coarse["audio_channel"]

        compressed["coarse_regions"].append(compressed_coarse)

    return compressed


def main() -> None:
    root = Path(__file__).resolve().parents[2]

    # Video UIDs from V6.3
    video_uids = [
        "06638e64-21ea-4065-8d8e-919b0aaf4538",
        "141ce529-f793-48be-acde-24747af68b07",
        "6baa673a-0cc6-4da2-a481-877de0d2faa5",
        "70f2a750-f403-41b8-aabb-480eb3ab4ed4",
        "819c8af7-851f-434f-ab32-318285bc54b1",
        "824e7896-904a-4ff5-b7ee-e21df21918c2",
        "9a7a189e-3494-40c0-84ee-5be47af574f7",
        "a6d45e95-8dc0-4932-83bf-ec53e265a16a",
        "ab93e55b-11cb-4332-b247-b3fb2fc67f53",
        "b9eed644-56a9-4a0b-b1a3-5cc0d6688297",
    ]

    source_dir = Path("C:/Users/72977/msc_thesis/thesis/thesis-av-evidence-hourvideo-pilot/outputs/experiments/hourvideo_r1_av_r3_2_ten_video_pilot_v1/cases")
    output_base = Path("C:/Users/72977/msc_thesis/thesis/thesis-av-evidence-hourvideo-pilot/outputs/experiments/hourvideo_v6_3_compressed_maps")
    output_base.mkdir(parents=True, exist_ok=True)

    total_original_size = 0
    total_compressed_size = 0

    for uid in video_uids:
        case_dir = source_dir / uid
        output_dir = output_base / uid
        output_dir.mkdir(parents=True, exist_ok=True)

        for map_name in ["r1_av_navigation_map.json", "r3_2_navigation_map.json"]:
            src_path = case_dir / map_name
            if not src_path.exists():
                continue

            # Read original
            original_data = json.loads(src_path.read_text(encoding="utf-8"))
            original_size = len(src_path.read_bytes())

            # Compress
            compressed_data = compress_navigation_map(original_data)
            compressed_json = json.dumps(compressed_data, ensure_ascii=False, indent=2)
            compressed_size = len(compressed_json.encode("utf-8"))

            # Write compressed
            output_path = output_dir / map_name
            output_path.write_text(compressed_json, encoding="utf-8")

            # Stats
            total_original_size += original_size
            total_compressed_size += compressed_size
            coarse_count = len(original_data.get("coarse_regions", []))
            ratio = (1 - compressed_size / original_size) * 100

            print(f"{uid[:8]}... {map_name}: {coarse_count} regions, {original_size:,}→{compressed_size:,} bytes ({ratio:.1f}% reduction)")

    print(f"\n总计: {total_original_size:,} → {total_compressed_size:,} bytes ({(1 - total_compressed_size/total_original_size)*100:.1f}% reduction)")
    print(f"Token 估算: {total_original_size//4:,} → {total_compressed_size//4:,} tokens")
    print(f"输出目录: {output_base}")


if __name__ == "__main__":
    main()
