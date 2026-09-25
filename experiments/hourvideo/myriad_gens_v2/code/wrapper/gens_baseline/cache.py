from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .access import AccessPolicy
from .config import resolve_profile_path


CANONICAL_RUNTIME_CACHE = Path(
    "/myriadfs/home/ucemxna/Scratch/workspace/HourVideo/runtime"
).resolve(strict=False)
FORBIDDEN_FIRST_CACHE = Path(
    "/myriadfs/home/ucemxna/Scratch/workspace/HourVideo/"
    "videoseal_original/cache/frames_16s_fps1"
).resolve(strict=False)


def _is_under(path: Path, root: Path) -> bool:
    if path == root:
        return True
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_profile(profile: dict[str, Any]) -> tuple[dict[str, Any], Path, Path]:
    if profile.get("profile") not in {
        "gens_hybrid_cap16_cache_reuse_v1",
        "gens_hybrid_symmetric_mcq_cap16_v1",
    }:
        raise RuntimeError("runtime cache loader is restricted to cache-reuse-v1")
    cache = profile["candidate_cache"]
    root = resolve_profile_path(profile, cache["root"]).resolve(strict=True)
    forbidden = Path(cache["forbidden_cache_root"]).resolve(strict=False)
    if root != CANONICAL_RUNTIME_CACHE:
        raise RuntimeError(f"noncanonical candidate cache rejected: {root}")
    if forbidden != FORBIDDEN_FIRST_CACHE:
        raise RuntimeError("frozen forbidden-cache identity drift")
    if _is_under(root, forbidden) or _is_under(forbidden, root):
        raise RuntimeError("canonical and forbidden cache roots must be isolated")
    if any(
        cache[key]
        for key in ("open_source_video", "decode_source_video", "nearest_pts", "extract_frames")
    ):
        raise RuntimeError("cache-reuse-v1 forbids video decoding and frame extraction")
    if cache.get("fallback") is not None:
        raise RuntimeError("cache-reuse-v1 forbids fallback candidate sources")
    return cache, root, forbidden


def load_frozen_cache_candidates(
    *,
    profile: dict[str, Any],
    video_id: str,
    duration_sec: float,
    policy: AccessPolicy,
) -> list[dict[str, Any]]:
    """Load one video's candidates from the sole frozen JPEG manifest.

    This module deliberately has no video decoder and never searches for another cache.
    """
    cache, cache_root, forbidden = _validate_profile(profile)
    manifest_path = resolve_profile_path(profile, cache["manifest"]).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["cache_root"] != str(cache_root):
        raise RuntimeError("candidate manifest cache root drift")
    if Path(manifest["forbidden_cache_root"]).resolve(strict=False) != forbidden:
        raise RuntimeError("candidate manifest forbidden root drift")
    if manifest["total_tree_sha256"] != cache["total_tree_sha256"]:
        raise RuntimeError("candidate manifest tree hash drift")
    if not manifest.get("structural_quality_pass"):
        raise RuntimeError("candidate manifest did not pass structural quality audit")

    verification_path = resolve_profile_path(
        profile, cache["source_verification"]
    ).resolve(strict=True)
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if not verification.get("source_verification_pass"):
        raise RuntimeError("cache anomaly source verification did not pass")

    matches = [item for item in manifest["videos"] if item["video_id"] == video_id]
    if len(matches) != 1:
        raise RuntimeError(f"video_id must appear once in frozen cache manifest: {video_id}")
    video = matches[0]
    filename_re = re.compile(cache["filename_regex"])
    expected_dir = (cache_root / video_id / "frames_1fps").resolve(strict=True)
    if not _is_under(expected_dir, cache_root):
        raise RuntimeError("per-video cache path escaped canonical root")

    candidates: list[dict[str, Any]] = []
    for expected_index, frame in enumerate(video["frames"]):
        filename = frame["filename"]
        match = filename_re.fullmatch(filename)
        if match is None or int(match.group(1)) != expected_index:
            raise RuntimeError("frozen cache filename/order drift")
        if int(frame["frame_number"]) != expected_index:
            raise RuntimeError("frozen cache frame number/order drift")
        timestamp = float(expected_index)
        if float(frame["timestamp_sec"]) != timestamp or not 0 <= timestamp < duration_sec:
            raise RuntimeError("frozen cache timestamp is invalid or out of range")
        image_path = (cache_root / frame["relative_path"]).resolve(strict=True)
        if not _is_under(image_path, expected_dir) or _is_under(image_path, forbidden):
            raise RuntimeError("cache image escaped the unique canonical per-video directory")
        policy.assert_read_allowed(image_path)
        stat = image_path.stat()
        if stat.st_size != int(frame["size_bytes"]):
            raise RuntimeError(f"cache image size drift: {image_path}")
        candidates.append(
            {
                "video_id": video_id,
                "target_timestamp_sec": expected_index,
                "timestamp_sec": timestamp,
                "frame_index": expected_index,
                "source_pts": None,
                "source_time_base_numerator": None,
                "source_time_base_denominator": None,
                "source_video_path": None,
                "extracted_frame_path": str(image_path),
                "candidate_source": "runtime_frames_1fps_cache_reuse_v1",
                "cache_relative_path": frame["relative_path"],
                "cache_filename": filename,
                "cache_sha256": frame["sha256"],
                "cache_tree_sha256": video["tree_sha256"],
                "cache_ordering_index": expected_index,
            }
        )

    if len(candidates) != int(video["frame_count"]):
        raise RuntimeError("candidate count does not match frozen cache manifest")
    if len({item["frame_index"] for item in candidates}) != len(candidates):
        raise RuntimeError("frozen cache produced duplicate candidates")
    return candidates
