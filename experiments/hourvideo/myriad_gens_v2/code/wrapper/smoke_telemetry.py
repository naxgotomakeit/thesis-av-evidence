from __future__ import annotations

import csv
import datetime as dt
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, TypeVar


T = TypeVar("T")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def timed_call(
    timings: dict[str, float],
    name: str,
    function: Callable[..., T],
    *args: Any,
    synchronize: Callable[[], None] | None = None,
    **kwargs: Any,
) -> T:
    if synchronize is not None:
        synchronize()
    started = time.perf_counter()
    result = function(*args, **kwargs)
    if synchronize is not None:
        synchronize()
    timings[name] = (time.perf_counter() - started) * 1000.0
    return result


class Tee:
    def __init__(self, primary: Any, path: Path):
        self.primary = primary
        self.file = path.open("a", encoding="utf-8", buffering=1)

    def write(self, value: str) -> int:
        self.primary.write(value)
        self.file.write(value)
        return len(value)

    def flush(self) -> None:
        self.primary.flush()
        self.file.flush()

    def close(self) -> None:
        self.file.close()


class SafetyAudit:
    """Process-local guard; it never probes the forbidden directory."""

    def __init__(self, forbidden_prefix: str):
        self.forbidden_prefix = os.path.abspath(os.path.normpath(forbidden_prefix))
        self.private_access_attempts = 0
        self.network_attempts = 0

    def install(self) -> None:
        def hook(event: str, args: tuple[Any, ...]) -> None:
            if event == "open" and args:
                path = args[0]
                if isinstance(path, (str, bytes, os.PathLike)):
                    normalized = os.path.abspath(os.path.normpath(os.fsdecode(path)))
                    if normalized == self.forbidden_prefix or normalized.startswith(
                        self.forbidden_prefix + os.sep
                    ):
                        self.private_access_attempts += 1
                        raise PermissionError("private evaluation path blocked by smoke audit")
            if event in {"socket.connect", "socket.connect_ex"}:
                self.network_attempts += 1
                raise PermissionError("network/API access blocked by smoke audit")

        sys.addaudithook(hook)


class ResourceMonitor:
    def __init__(self, csv_path: Path, interval_sec: float = 0.5, target_gpu_index: int = 0):
        self.csv_path = csv_path
        self.interval_sec = interval_sec
        self.target_gpu_index = target_gpu_index
        self.started_perf = time.perf_counter()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="smoke-resource-monitor", daemon=True)
        self.errors: list[str] = []
        self.cpu_rss_peak_bytes = 0
        self.device_memory_peak_mib = 0.0
        self.samples = 0

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)

    def _sample_cpu(self) -> None:
        try:
            import psutil

            rss = psutil.Process(os.getpid()).memory_info().rss
            self.cpu_rss_peak_bytes = max(self.cpu_rss_peak_bytes, int(rss))
        except Exception as exc:  # telemetry must not change selection behavior
            message = f"cpu_sampler:{type(exc).__name__}:{exc}"
            if message not in self.errors:
                self.errors.append(message)

    def _sample_gpu(self) -> list[list[str]]:
        command = [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,utilization.memory",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(
                command, check=True, capture_output=True, text=True, timeout=5
            )
            rows = list(csv.reader(completed.stdout.splitlines(), skipinitialspace=True))
            for row in rows:
                if len(row) >= 5 and int(row[0].strip()) == self.target_gpu_index:
                    self.device_memory_peak_mib = max(
                        self.device_memory_peak_mib, float(row[3].strip())
                    )
            return rows
        except Exception as exc:  # telemetry must not change selection behavior
            message = f"gpu_sampler:{type(exc).__name__}:{exc}"
            if message not in self.errors:
                self.errors.append(message)
            return []

    def _run(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "timestamp_utc",
                    "elapsed_sec",
                    "gpu_index",
                    "gpu_uuid",
                    "gpu_name",
                    "memory_used_mib",
                    "memory_total_mib",
                    "gpu_utilization_percent",
                    "memory_utilization_percent",
                ]
            )
            while not self.stop_event.is_set():
                self._sample_cpu()
                now = dt.datetime.now(dt.timezone.utc).isoformat()
                elapsed = time.perf_counter() - self.started_perf
                rows = self._sample_gpu()
                for row in rows:
                    writer.writerow([now, f"{elapsed:.6f}", *[cell.strip() for cell in row]])
                    self.samples += 1
                handle.flush()
                self.stop_event.wait(self.interval_sec)


def safe_json_write(path: Path, value: Any, telemetry_errors: list[str]) -> None:
    try:
        path.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        telemetry_errors.append(f"write:{path.name}:{type(exc).__name__}:{exc}")


def safe_text_write(path: Path, value: str, telemetry_errors: list[str]) -> None:
    try:
        path.write_text(value, encoding="utf-8")
    except Exception as exc:
        telemetry_errors.append(f"write:{path.name}:{type(exc).__name__}:{exc}")
