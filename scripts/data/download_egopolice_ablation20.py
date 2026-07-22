from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _probe(path: Path, ffprobe_path: str) -> tuple[bool, str | None]:
    completed = _run([
        ffprobe_path, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_type",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ])
    valid = completed.returncode == 0 and "video" in completed.stdout
    return valid, completed.stderr.strip() or None


def _blocker(stderr: str) -> str | None:
    lowered = stderr.lower()
    if "http error 429" in lowered or "too many requests" in lowered:
        return "vimeo_rate_limited_http_429"
    if (
        "http error 401" in lowered or "oauth" in lowered
        or "log in" in lowered or "cookies" in lowered
    ):
        return "vimeo_authentication_required_http_401_or_oauth"
    return None


def _resolved_rows(target_manifest: Path, parent_manifest: Path) -> list[dict[str, Any]]:
    targets = json.loads(target_manifest.read_text(encoding="utf-8"))["videos"]
    parents = {
        row["video_id"]: row
        for row in json.loads(parent_manifest.read_text(encoding="utf-8"))["videos"]
    }
    rows = []
    for target in targets:
        video_id = str(target["video_id"])
        if video_id not in parents:
            raise ValueError(f"Formal target is absent from frozen parent: {video_id}")
        rows.append({**parents[video_id], **target})
    if len(rows) != 20 or len({row["video_id"] for row in rows}) != 20:
        raise ValueError("Formal download target must contain exactly 20 unique videos")
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download only frozen EgoPolice ablation20 targets")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--target-manifest", type=Path,
        default=ROOT / "config/data/egopolice_ablation20_v1.json",
    )
    parser.add_argument(
        "--parent-manifest", type=Path,
        default=ROOT / "config/data/egopolice_50videos.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs/data_audit/egopolice_ablation20_download_attempt.json",
    )
    parser.add_argument("--yt-dlp-path", default="yt-dlp")
    parser.add_argument("--ffprobe-path", default="ffprobe")
    parser.add_argument("--ffmpeg-path", default="ffmpeg")
    parser.add_argument("--cookies-from-browser")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = _resolved_rows(args.target_manifest, args.parent_manifest)
    results: list[dict[str, Any]] = []
    blocked_provider: dict[str, str] = {}
    for row in rows:
        relative = PurePosixPath(str(row["relative_video_path"]))
        target = args.data_root.joinpath(*relative.parts)
        provider = str(row.get("provider") or "unknown")
        if target.is_file():
            valid, error = _probe(target, args.ffprobe_path)
            results.append({
                "video_id": row["video_id"], "provider": provider,
                "path": str(target),
                "status": "existing_verified" if valid else "existing_invalid",
                "error": error,
            })
            continue
        if provider in blocked_provider:
            results.append({
                "video_id": row["video_id"], "provider": provider,
                "path": str(target), "status": "blocked_not_attempted",
                "error": blocked_provider[provider],
            })
            continue
        command = [
            args.yt_dlp_path, "--no-playlist", "--continue", "--part", "--newline",
            "--retries", "2", "--fragment-retries", "2", "--sleep-requests", "2",
            "--merge-output-format", "mp4", "-f", "bv*+ba/b", "-o", str(target),
        ]
        ffmpeg_path = Path(args.ffmpeg_path)
        if ffmpeg_path.parent != Path("."):
            command.extend(["--ffmpeg-location", str(ffmpeg_path.parent)])
        if args.cookies_from_browser:
            command.extend(["--cookies-from-browser", args.cookies_from_browser])
        command.append(str(row["source_url"]))
        if not args.execute:
            results.append({
                "video_id": row["video_id"], "provider": provider,
                "path": str(target), "status": "dry_run",
                "command": command[:-1] + ["<official_source_url>"],
            })
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        completed = _run(command)
        valid, probe_error = _probe(target, args.ffprobe_path) if target.is_file() else (False, None)
        stderr_tail = completed.stderr[-2000:]
        blocker = _blocker(completed.stderr) if provider == "Vimeo" else None
        if blocker:
            blocked_provider[provider] = blocker
        results.append({
            "video_id": row["video_id"], "provider": provider,
            "path": str(target),
            "status": "downloaded_verified" if completed.returncode == 0 and valid else "download_failed",
            "returncode": completed.returncode,
            "blocker": blocker,
            "stderr_tail": stderr_tail,
            "probe_error": probe_error,
        })
    payload = {
        "target_manifest": str(args.target_manifest),
        "target_count": 20,
        "execute": args.execute,
        "safe_provider_blocks": blocked_provider,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    ready = {"existing_verified", "downloaded_verified"}
    return 0 if all(row["status"] in ready for row in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
