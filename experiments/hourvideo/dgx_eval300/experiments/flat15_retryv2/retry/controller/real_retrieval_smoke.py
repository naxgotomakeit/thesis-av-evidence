#!/usr/bin/env python3
"""One-call real Flat-15 retrieval smoke; no Planner, Inspector, or gold."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import embedding_usage_proxy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    if os.getenv("VISUAL_RETRIEVE_TOPK") != "15":
        raise SystemExit("smoke Top-K must equal 15")
    if os.getenv("MLLM_RETRY_TIMES") != "1":
        raise SystemExit("smoke transport cap requires MLLM_RETRY_TIMES=1")
    embedding_usage_proxy.install()

    from videoseal.tools.visual_tools import VisualRetrieveAliasTool
    import videoseal.tools.visual_tools as visual_tools

    code_root = Path(os.environ["CODE_ROOT"]).resolve()
    imported = Path(visual_tools.__file__).resolve()
    if code_root not in imported.parents:
        raise SystemExit(f"runtime isolation failed: {imported}")

    tool = VisualRetrieveAliasTool()
    result = tool.forward(
        query=args.query,
        video_id=args.video_id,
        index_path=args.index,
        original_question=args.query,
    )
    metadata = dict(result.metadata or {})
    spans = list(metadata.get("spans") or [])
    record = {
        "status": "PASS" if not result.error else "FAIL",
        "video_id": args.video_id,
        "query_sha256_only": True,
        "runtime_module": str(imported),
        "backend": "flat_embed",
        "visual_retrieve_topk": 15,
        "retrieval_error": result.error,
        "selected_span_count": len(spans),
        "selected_span_count_le_15": len(spans) <= 15,
        "summarizer_logical_request_count": int(metadata.get("visual_request_count") or 0),
        "summarizer_usage": {
            "prompt_tokens": metadata.get("prompt_tokens"),
            "completion_tokens": metadata.get("completion_tokens"),
            "total_tokens": metadata.get("total_tokens"),
        },
        "planner_calls": 0,
        "inspector_calls": 0,
        "gold_loaded": False,
    }
    if not spans or len(spans) > 15:
        record["status"] = "FAIL"
    if record["summarizer_logical_request_count"] != 1:
        record["status"] = "FAIL"
    Path(args.result).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return 0 if record["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

