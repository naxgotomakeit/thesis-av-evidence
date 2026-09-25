from __future__ import annotations

import gc
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from .access import AccessPolicy
from .clip_stage import ClipRetriever
from .config import resolve_profile_path
from .contract import blank_latencies, build_package, cap_gens_frames
from .gens_stage import ContextOverflow, GenSSelector
from .parser import ParserFailure, map_parsed_frames, parse_gens_response
from .query import build_retrieval_query, load_selector, load_template, query_sha256
from .cache import load_frozen_cache_candidates


def atomic_write_json(path: str | Path, value: Any, policy: AccessPolicy) -> None:
    destination = Path(policy.assert_write_allowed(path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, destination)


def atomic_write_text(path: str | Path, value: str, policy: AccessPolicy) -> None:
    destination = Path(policy.assert_write_allowed(path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        handle.write(value)
        temporary = handle.name
    os.replace(temporary, destination)


def build_access_policy(profile: dict[str, Any]) -> tuple[AccessPolicy, Path, Path]:
    allowlist_path = resolve_profile_path(profile, profile["inputs"]["allowlist"])
    output_root = resolve_profile_path(profile, profile["outputs"]["root"])
    model_roots = [
        str(resolve_profile_path(profile, profile["clip"]["local_path"])),
        str(resolve_profile_path(profile, profile["gens"]["local_path"])),
    ]
    cache_roots = []
    if "candidate_cache" in profile:
        cache_roots.append(str(resolve_profile_path(profile, profile["candidate_cache"]["root"])))
    policy = AccessPolicy.from_allowlist(
        allowlist_path,
        model_roots=model_roots,
        output_roots=[str(output_root)],
        cache_roots=cache_roots,
    )
    return policy, allowlist_path, output_root


def _base_provenance(profile: dict[str, Any], query: str) -> dict[str, Any]:
    return {
        "profile": profile["profile"],
        "profile_sha256": profile["_profile_sha256"],
        "query_template_sha256": profile["query"]["template_sha256"],
        "query_sha256": query_sha256(query),
        "query_text": query,
        "clip": {
            "model_id": profile["clip"]["model_id"],
            "revision": profile["clip"]["revision"],
            "weight_sha256": profile["clip"]["weight_sha256"],
            "resolution": profile["clip"]["resolution"],
            "top_k": profile["clip"]["top_k"],
        },
        "gens": {
            "model_id": profile["gens"]["model_id"],
            "revision": profile["gens"]["revision"],
            "resolution": profile["gens"]["resolution"],
            "dtype": profile["gens"]["dtype"],
            "attention": profile["gens"]["attention"],
            "instruction_template_sha256": profile["gens"]["instruction_template_sha256"],
        },
        "candidate_pool": [],
        "clip_candidates": [],
        "clip_text": {},
        "gens_context_tokens": None,
        "gens_raw_response": None,
        "gens_parser": {"status": "not_run"},
        "gens_relevance_order": [],
    }


def _load_allowlist_video_metadata(allowlist_path: Path, video_id: str) -> dict[str, Any]:
    data = json.loads(allowlist_path.read_text(encoding="utf-8"))
    matches = [item for item in data["videos"] if item["video_id"] == video_id]
    if len(matches) != 1:
        raise RuntimeError(f"video_id {video_id} is not uniquely allowlisted")
    return matches[0]


def _load_frozen_candidates(
    *,
    profile: dict[str, Any],
    record: dict[str, Any],
    policy: AccessPolicy,
) -> list[dict[str, Any]]:
    video_id = record["video_id"]
    return load_frozen_cache_candidates(
        profile=profile,
        video_id=video_id,
        duration_sec=float(record["video_duration_sec"]),
        policy=policy,
    )


def run_one_question(profile: dict[str, Any], qa_uid: str, device: str = "cuda:0") -> dict[str, Any]:
    started_total = time.perf_counter()
    policy, allowlist_path, output_root = build_access_policy(profile)
    records = load_selector(profile, policy)
    matches = [record for record in records if record["qa_uid"] == qa_uid]
    if len(matches) != 1:
        raise ValueError(f"qa_uid must identify exactly one selector record: {qa_uid}")
    record = matches[0]
    template = load_template(profile)
    query = build_retrieval_query(record, template)
    provenance = _base_provenance(profile, query)
    latencies = blank_latencies()
    question_dir = output_root / "questions" / qa_uid
    atomic_write_text(question_dir / "retrieval_query.txt", query, policy)

    def finish(status: str, error: dict[str, Any] | None, selected: list[dict]) -> dict:
        latencies["total"] = (time.perf_counter() - started_total) * 1000.0
        package = build_package(
            profile=profile,
            qa_uid=qa_uid,
            video_id=record["video_id"],
            status=status,
            error=error,
            selected_frames=selected,
            latencies=latencies,
            provenance=provenance,
        )
        atomic_write_json(question_dir / "selector_output.json", package, policy)
        return package

    started = time.perf_counter()
    try:
        candidates = _load_frozen_candidates(
            profile=profile,
            record=record,
            policy=policy,
        )
    except Exception as exc:
        return finish("candidate_failure", {"type": type(exc).__name__, "message": str(exc)}, [])
    latencies["candidate_extraction"] = (time.perf_counter() - started) * 1000.0
    provenance["candidate_pool"] = candidates

    clip_retriever = None
    try:
        clip_retriever = ClipRetriever(profile, policy, device=device)
        clip_candidates, text_metadata, clip_latencies = clip_retriever.retrieve(query, candidates)
        latencies.update(clip_latencies)
        provenance["clip_candidates"] = clip_candidates
        provenance["clip_text"] = text_metadata
        atomic_write_json(question_dir / "clip_candidates.json", clip_candidates, policy)
    except Exception as exc:
        return finish("clip_failure", {"type": type(exc).__name__, "message": str(exc)}, [])
    finally:
        if clip_retriever is not None:
            clip_retriever.close()
            del clip_retriever
            gc.collect()

    gens_selector = None
    try:
        gens_selector = GenSSelector(profile, policy, device=device)
        raw_response, chronological, gens_latencies, context_tokens = gens_selector.select(
            query, clip_candidates
        )
        latencies.update(gens_latencies)
        provenance["gens_context_tokens"] = context_tokens
        provenance["gens_raw_response"] = raw_response
        atomic_write_text(question_dir / "gens_raw_response.txt", raw_response, policy)
        atomic_write_json(question_dir / "gens_input_candidates.json", chronological, policy)
    except ContextOverflow as exc:
        return finish("context_overflow", {"type": type(exc).__name__, "message": str(exc)}, [])
    except Exception as exc:
        return finish("gens_failure", {"type": type(exc).__name__, "message": str(exc)}, [])
    finally:
        if gens_selector is not None:
            gens_selector.close()
            del gens_selector
            gc.collect()

    started = time.perf_counter()
    try:
        parsed = parse_gens_response(raw_response, len(chronological))
        mapped = map_parsed_frames(parsed, chronological)
    except ParserFailure as exc:
        latencies["gens_parse"] = (time.perf_counter() - started) * 1000.0
        provenance["gens_parser"] = {"status": "failure", **exc.to_dict()}
        return finish("parser_failure", exc.to_dict(), [])
    latencies["gens_parse"] = (time.perf_counter() - started) * 1000.0
    provenance["gens_parser"] = {
        "status": "ok",
        "parsed_unique_frame_count": len(mapped),
        "strict_json": True,
    }
    atomic_write_json(
        question_dir / "gens_parser_result.json",
        [item.to_dict() for item in parsed],
        policy,
    )
    if not mapped:
        return finish(
            "gens_no_valid_frames",
            {"type": "EmptyGenSSelection", "message": "GenS returned zero legal frames"},
            [],
        )

    selected, relevance_order = cap_gens_frames(mapped, cap=16)
    provenance["gens_relevance_order"] = relevance_order
    return finish("ok", None, selected)
