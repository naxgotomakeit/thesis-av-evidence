from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    "torch", "torchvision", "transformers", "accelerate", "bitsandbytes",
    "numpy", "Pillow", "scipy", "scikit-learn",
    "sentence-transformers", "timm", "huggingface-hub", "pytest",
)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def executable_version(command: str) -> dict[str, Any]:
    resolved = shutil.which(command) if not Path(command).is_file() else str(Path(command))
    if not resolved:
        return {"configured": command, "available": False, "path": None, "version_line": None}
    try:
        completed = subprocess.run(
            [resolved, "-version"], capture_output=True, text=True, check=False, timeout=10
        )
        text = completed.stdout or completed.stderr
        version_line = text.splitlines()[0] if text else None
        return {
            "configured": command,
            "available": completed.returncode == 0,
            "path": resolved,
            "version_line": version_line,
        }
    except Exception as exc:
        return {
            "configured": command, "available": False, "path": resolved,
            "version_line": None, "error": repr(exc),
        }


def torch_environment() -> dict[str, Any]:
    if package_version("torch") is None:
        return {
            "installed": False, "version": None, "cuda_runtime": None,
            "cuda_available": False, "gpu_count": 0, "gpus": [],
        }
    try:
        import torch

        available = torch.cuda.is_available()
        gpus = []
        if available:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                free_bytes = None
                total_bytes = int(properties.total_memory)
                try:
                    free_bytes, _ = torch.cuda.mem_get_info(index)
                    free_bytes = int(free_bytes)
                except Exception:
                    pass
                gpus.append(
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "total_vram_bytes": total_bytes,
                        "free_vram_bytes": free_bytes,
                    }
                )
        return {
            "installed": True,
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": available,
            "gpu_count": len(gpus),
            "gpus": gpus,
        }
    except Exception as exc:
        return {
            "installed": True, "version": package_version("torch"),
            "cuda_runtime": None, "cuda_available": False, "gpu_count": 0,
            "gpus": [], "import_error": repr(exc),
        }


def selected_media_status(manifest_path: Path, data_root: Path | None) -> dict[str, Any]:
    if not manifest_path.is_file():
        return {"manifest_exists": False, "selected_count": 0, "available_count": 0}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("videos", [])
    available = 0
    if data_root:
        for row in rows:
            relative = Path(*str(row["relative_video_path"]).replace("\\", "/").split("/"))
            available += (data_root / relative).is_file()
    return {
        "manifest_exists": True,
        "manifest_path": str(manifest_path),
        "selected_count": len(rows),
        "available_count": available,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report the local thesis runtime without loading model weights")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--manifest", type=Path, default=ROOT / "config/data/egopolice_50videos.json")
    parser.add_argument("--ffmpeg-path")
    parser.add_argument("--ffprobe-path")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root or (Path(os.path.expanduser(os.environ["DATA_ROOT"])) if os.environ.get("DATA_ROOT") else None)
    model_path = args.model_path or (Path(os.path.expanduser(os.environ["MODEL_PATH"])) if os.environ.get("MODEL_PATH") else None)
    model_root = args.model_root or (Path(os.path.expanduser(os.environ["MODEL_ROOT"])) if os.environ.get("MODEL_ROOT") else None)
    if model_path is None and model_root is not None:
        model_path = model_root / "Qwen2.5-VL-7B-Instruct"
    ffmpeg = args.ffmpeg_path or os.environ.get("FFMPEG_PATH", "ffmpeg")
    ffprobe = args.ffprobe_path or os.environ.get("FFPROBE_PATH", "ffprobe")
    metadata = {}
    if data_root:
        metadata = {
            f"mcq_{duration_type}.json": (data_root / f"mcq_{duration_type}.json").is_file()
            for duration_type in ("1s", "10s", "60s")
        }
    report = {
        "os": platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "packages": {name: package_version(name) for name in PACKAGES},
        "torch": torch_environment(),
        "ffmpeg": executable_version(ffmpeg),
        "ffprobe": executable_version(ffprobe),
        "paths": {
            "data_root": str(data_root) if data_root else None,
            "data_root_exists": bool(data_root and data_root.is_dir()),
            "model_path": str(model_path) if model_path else None,
            "model_path_exists": bool(model_path and model_path.is_dir()),
        },
        "egopolice_metadata": metadata,
        "egopolice_frozen_subset": selected_media_status(args.manifest, data_root),
        "full_qwen_model_loaded": False,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
