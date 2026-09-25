#!/usr/bin/env python3
"""Retry-v2 execution wrapper adding observation-only embedding usage logs."""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent / "retry_infrastructure_audit_20260913T115931Z" / "fixed_orchestration"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BASE_DIR))

import embedding_usage_proxy
import gold_free_retry_runner_v2 as base


def child_run(task, root, max_steps, result_queue) -> None:
    embedding_usage_proxy.install()
    result_queue.put(base.run_one(task, Path(root), max_steps))


if __name__ == "__main__":
    base.child_run = child_run
    raise SystemExit(base.main())

