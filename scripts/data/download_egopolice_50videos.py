from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def probe(path: Path, ffprobe: str) -> tuple[bool, str]:
    completed = run([
        ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=codec_type", "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ])
    return completed.returncode == 0 and "video" in completed.stdout, completed.stderr.strip()


def destination(data_root: Path, row: dict[str, Any]) -> Path:
    parts = str(row["relative_video_path"]).replace("\\", "/").split("/")
    return data_root.joinpath(*parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download only the frozen EgoPolice 50-video subset; dry-run unless --execute is set"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATA_ROOT", "")))
    parser.add_argument("--yt-dlp-path", default="yt-dlp")
    parser.add_argument("--ffmpeg-path", default=os.environ.get("FFMPEG_PATH", "ffmpeg"))
    parser.add_argument("--ffprobe-path", default=os.environ.get("FFPROBE_PATH", "ffprobe"))
    parser.add_argument("--cookies-from-browser")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data_root:
        raise SystemExit("DATA_ROOT is required")
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))["videos"][: args.limit]
    results = []
    for row in rows:
        target = destination(args.data_root, row)
        if target.is_file():
            valid, error = probe(target, args.ffprobe_path)
            results.append({"video_id": row["video_id"], "status": "existing_verified" if valid else "existing_invalid", "path": str(target), "error": error or None})
            continue
        command = [
            args.yt_dlp_path, "--no-playlist", "--no-part", "--newline",
        ]
        ffmpeg_path = Path(args.ffmpeg_path)
        if ffmpeg_path.parent != Path("."):
            command.extend(["--ffmpeg-location", str(ffmpeg_path.parent)])
        command.extend(["--merge-output-format", "mp4", "-f", "bv*+ba/b", "-o", str(target)])
        if args.cookies_from_browser:
            command.extend(["--cookies-from-browser", args.cookies_from_browser])
        command.append(str(row["source_url"]))
        if not args.execute:
            results.append({"video_id": row["video_id"], "status": "dry_run", "path": str(target), "command": command[:-1] + ["<source_url>"]})
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        completed = run(command)
        valid, probe_error = probe(target, args.ffprobe_path) if target.is_file() else (False, "file not created")
        results.append(
            {
                "video_id": row["video_id"],
                "status": "downloaded_verified" if completed.returncode == 0 and valid else "download_failed",
                "path": str(target),
                "returncode": completed.returncode,
                "stderr_tail": completed.stderr[-1000:],
                "probe_error": probe_error or None,
            }
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if all(row["status"] in {"dry_run", "existing_verified", "downloaded_verified"} for row in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
