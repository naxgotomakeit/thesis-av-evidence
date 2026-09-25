#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--uid", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--output", required=True)
    ns = ap.parse_args()

    roots = sorted(Path(ns.root).glob(f"*/20*/trajectory.json"))
    matches = []
    for path in roots:
        data = json.loads(path.read_text(encoding="utf-8"))
        if str(data.get("uid")) == ns.uid:
            matches.append((path, data))
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one trajectory for {ns.uid}, found {len(matches)}")
    path, traj = matches[0]
    retrieval = []
    fallback_steps = []
    for step in traj.get("steps") or []:
        action = step.get("action") if isinstance(step.get("action"), dict) else {}
        obs = step.get("observation") if isinstance(step.get("observation"), dict) else {}
        meta = obs.get("metadata") if isinstance(obs.get("metadata"), dict) else {}
        if action.get("name") == "visual_retrieve":
            retrieval.append((obs, meta))
        if action.get("name") == "visual_inspect" and obs.get("forced") is True and obs.get("mode") == "full_video":
            fallback_steps.append((step, obs, meta))

    errors = [str(obs.get("error") or "") for obs, _ in retrieval if not obs.get("ok")]
    hierarchy_complete = bool(retrieval) and all(meta.get("hierarchy_complete") is True for _, meta in retrieval)
    hierarchy_used = bool(retrieval) and all(meta.get("hierarchy_used") is True for _, meta in retrieval)
    fallback_valid = True
    fallback_details = []
    for step, obs, meta in fallback_steps:
        timestamps = [float(x) for x in meta.get("image_timestamps") or []]
        valid = (
            obs.get("ok") is True
            and int(meta.get("decoded_frame_count") or 0) == 64
            and int(meta.get("sent_image_count") or 0) == 64
            and len(timestamps) == 64
            and len(set(timestamps)) == 64
            and all(a < b for a, b in zip(timestamps, timestamps[1:]))
            and timestamps[0] == 0.0
        )
        fallback_valid = fallback_valid and valid
        fallback_details.append({
            "valid": valid,
            "decoded_frames": meta.get("decoded_frame_count"),
            "sent_images": meta.get("sent_image_count"),
            "timestamp_count": len(timestamps),
            "unique_timestamp_count": len(set(timestamps)),
            "first_timestamp_sec": timestamps[0] if timestamps else None,
            "last_timestamp_sec": timestamps[-1] if timestamps else None,
            "span": ((step.get("action") or {}).get("arguments") or {}).get("spans"),
        })

    passed = not errors and hierarchy_complete and hierarchy_used and fallback_valid
    report = {
        "status": "PASS" if passed else "FAIL",
        "profile": ns.profile,
        "uid": ns.uid,
        "run_id": traj.get("run_id"),
        "trajectory_path": str(path.resolve()),
        "finished_at": traj.get("finished_at"),
        "note": traj.get("note"),
        "retrieval_calls": len(retrieval),
        "retrieval_errors": errors,
        "all_hierarchy_complete": hierarchy_complete,
        "all_hierarchy_used": hierarchy_used,
        "forced_fallback_count": len(fallback_steps),
        "forced_fallback_valid": fallback_valid,
        "fallback_details": fallback_details,
    }
    target = Path(ns.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
