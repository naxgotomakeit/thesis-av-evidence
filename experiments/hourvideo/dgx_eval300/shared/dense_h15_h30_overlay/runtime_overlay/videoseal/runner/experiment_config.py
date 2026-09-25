from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.run(args, cwd=cwd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.strip()
    except Exception:
        return ""


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _server_version(python: str, package: str, cwd: Path) -> str | None:
    if not python or not Path(python).is_file():
        return None
    value = _run([python, "-c", f"import importlib.metadata; print(importlib.metadata.version({package!r}))"], cwd)
    return value or None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def update_experiment_config(
    *, output_path: Path, repo_root: Path, parquet: Path, uids_file: Path | None,
    save_runs: Path, max_steps: int, task_timeout_sec: float, concurrency: int,
    boundary: str,
) -> None:
    """Best-effort secret-free experiment provenance record."""
    try:
        data: dict[str, Any] = {}
        if output_path.is_file():
            try:
                data = json.loads(output_path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        if not data:
            diff = subprocess.run(
                ["git", "diff", "--binary", "HEAD"], cwd=repo_root, check=False, stdout=subprocess.PIPE
            ).stdout
            status = subprocess.run(
                ["git", "status", "--porcelain=v1", "-z"], cwd=repo_root, check=False, stdout=subprocess.PIPE
            ).stdout
            gpu_line = _run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], repo_root
            )
            try:
                import torch
                torch_version = str(torch.__version__)
                torch_cuda = str(torch.version.cuda)
            except Exception:
                torch_version = _version("torch")
                torch_cuda = None
            data = {
                "experiment_name": os.getenv("EXPERIMENT_NAME", ""),
                "uid_file": {"path": str(uids_file) if uids_file else None, "sha256": _sha256(uids_file) if uids_file else None},
                "parquet": {"path": str(parquet), "sha256": _sha256(parquet)},
                "output_directory": str(save_runs),
                "git": {
                    "head": _run(["git", "rev-parse", "HEAD"], repo_root),
                    "dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
                    "status_sha256": hashlib.sha256(status).hexdigest(),
                    "dirty": bool(status),
                    "dirty_diff_hash_scope": "git diff --binary HEAD; status hash separately includes tracked/untracked path state",
                },
                "uv_lock_sha256": _sha256(repo_root / "uv.lock"),
                "software": {
                    "python": platform.python_version(),
                    "python_executable": sys.executable,
                    "runner_torch": torch_version,
                    "runner_torch_cuda": torch_cuda,
                    "server_python": os.getenv("VLLM_PYTHON", ""),
                    "torch": _server_version(os.getenv("VLLM_PYTHON", ""), "torch", repo_root),
                    "cuda_runtime": (_run([os.getenv("VLLM_PYTHON", ""), "-c", "import torch; print(torch.version.cuda)"], repo_root) or None),
                    "vllm": _server_version(os.getenv("VLLM_PYTHON", ""), "vllm", repo_root),
                    "transformers": _server_version(os.getenv("VLLM_PYTHON", ""), "transformers", repo_root),
                },
                "hardware": {"gpu_driver": gpu_line, "architecture": platform.machine()},
                "models": {
                    "planner": os.getenv("PLANNER_MODEL", ""),
                    "visual": os.getenv("VISUAL_MODEL", ""),
                },
                "parameters": {
                    "concurrency": concurrency,
                    "max_steps": max_steps,
                    "http_timeout_sec": float(os.getenv("AGENT_LLM_TIMEOUT", "0") or 0),
                    "task_timeout_sec": task_timeout_sec,
                    "planner": {
                        "dtype": "bfloat16", "gpu_memory_utilization": os.getenv("PLANNER_GPU_MEMORY_UTIL", ""),
                        "max_model_len": os.getenv("PLANNER_MAX_MODEL_LEN", "36864"), "max_num_seqs": 1,
                        "port": os.getenv("PLANNER_PORT", "18082"), "enforce_eager": True,
                        "generation_config": "vllm", "request_logging": False,
                    },
                    "visual": {
                        "dtype": "bfloat16", "gpu_memory_utilization": os.getenv("VISUAL_GPU_MEMORY_UTIL", ""),
                        "max_model_len": os.getenv("VISUAL_MAX_MODEL_LEN", "65536"), "max_num_seqs": 1,
                        "limit_mm_per_prompt": {"image": 64}, "enforce_eager": True,
                        "port": os.getenv("VISUAL_PORT", "18083"), "generation_config": "vllm",
                        "request_logging": False,
                    },
                    "runner_environment": {name: os.getenv(name, "") for name in (
                        "AGENT_LLM_BACKEND", "AGENT_API_USE_MESSAGES", "AGENT_LLM_MAX_TOKENS",
                        "AGENT_LLM_TEMPERATURE", "AGENT_LLM_TIMEOUT", "VISUAL_RETRIEVE_SUMMARY_ENABLED",
                        "RETRIEVE_SUMMARY_MAX_SPANS", "SEMANTIC_RETRIEVE_MIX", "SEMANTIC_RETRIEVE_TOPK",
                        "VISUAL_RETRIEVE_TOPK", "RETRIEVE_EMBED_WEIGHT", "RETRIEVE_BM25_WEIGHT",
                        "INSPECT_MAX_LONG_EDGE", "INSPECT_VLM_MAX_TOKENS", "INSPECT_VLM_TEMPERATURE",
                        "INSPECT_MAX_TOTAL_IMAGES", "INSPECT_FPS", "VISUAL_INSPECT_DYNAMIC_MAX_LONG_EDGE",
                        "VISUAL_INSPECT_TOTAL_PIXELS", "VISUAL_INSPECT_MIN_PIXELS", "VISUAL_INSPECT_EDGE_MULTIPLE",
                        "MLLM_TIMEOUT", "MLLM_RETRY_TIMES", "MLLM_RETRY_DELAY", "EMBED_RETRY_TIMES",
                        "EMBED_RETRY_DELAY", "EMBEDDING_API_BASE", "EMBEDDING_MODEL",
                        "AGENT_FORCE_LAST_STEP_VISUAL_INSPECT", "AGENT_ENABLE_LAST_STEP_VISUAL_INSPECT_FALLBACK",
                        "AGENT_ENABLE_MAX_STEP_VISUAL_INSPECT_FALLBACK", "AGENT_LAST_STEP_VISUAL_INSPECT_PROMPT_MODE",
                    )},
                    "runner_argv": sys.argv,
                },
                "runner_timing": {},
            }
        timing = data.setdefault("runner_timing", {})
        attempts = timing.setdefault("attempts", [])
        if boundary == "started_at_utc":
            attempts.append({"started_at_utc": _utc_now(), "finished_at_utc": None})
        elif attempts:
            attempts[-1]["finished_at_utc"] = _utc_now()
        else:
            attempts.append({"started_at_utc": None, "finished_at_utc": _utc_now()})
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, output_path)
    except Exception as exc:
        print(json.dumps({"experiment_config_warning": f"{type(exc).__name__}: {exc}"}), flush=True)
