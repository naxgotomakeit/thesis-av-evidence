from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .schemas import PLANNER_SYSTEM_PROMPT, REPAIR_INSTRUCTION, planner_json_schema
from .validation import validate_planner_output, validate_ranking

DEFAULT_CONFIG: dict[str, Any] = {
    "config_version": "planner_medium_retrieval_v1",
    "planner": {
        "model": "claude-haiku-4-5-20251001",
        "temperature": 0.0,
        "max_tokens": 1800,
        "max_repair_calls": 1,
    },
    "siglip": {
        "model": "google/siglip-base-patch16-224",
        "cache_dir": r"D:\EgoPolice\model_cache\huggingface",
        "local_files_only": True,
        "embedding_dimension": 768,
        "max_text_tokens": 64,
    },
    "weights": {"visual_score": 0.60, "lexical_score": 0.30, "coarse_prior": 0.10},
    "selection": {
        "targeted_max_mediums": 8,
        "targeted_min_coarse": 2,
        "targeted_max_coarse": 4,
        "global_per_coarse": 1,
        "multi_target_per_unit": 4,
        "multi_target_max_merged": 10,
    },
}

STOPWORDS = {
    "a", "an", "and", "any", "are", "at", "be", "before", "did", "during", "for",
    "from", "happened", "if", "in", "is", "it", "of", "or", "other", "so", "the",
    "then", "throughout", "to", "use", "was", "were", "when", "with",
}


class PlannerError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_tokens(text: str) -> list[str]:
    raw = re.findall(r"[a-z0-9]+", text.lower())
    tokens: list[str] = []
    for token in raw:
        if token in STOPWORDS:
            continue
        if len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 4 and token.endswith("ied"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("ed"):
            token = token[:-2]
        elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return tokens


def lexical_similarity(query_text: str, caption: str) -> tuple[float, list[str]]:
    query_tokens = normalize_tokens(query_text)
    caption_tokens = set(normalize_tokens(caption))
    unique_query = list(dict.fromkeys(query_tokens))
    matched = [token for token in unique_query if token in caption_tokens]
    score = len(matched) / max(1, len(unique_query))
    return float(score), matched


def query_text(question: str, unit: dict[str, Any]) -> str:
    variants = " ; ".join(unit["query_variants"])
    return f"{question} Retrieval target: {unit['description']}. Variants: {variants}"


def _minmax(values: np.ndarray) -> np.ndarray:
    low, high = float(values.min()), float(values.max())
    if math.isclose(low, high):
        return np.ones_like(values, dtype=np.float64)
    return (values - low) / (high - low)


class SiglipTextEncoder:
    def __init__(self, config: dict[str, Any], *, device: str | None = None) -> None:
        self.config = config
        self.device_name = device
        self._model = None
        self._tokenizer = None

    @staticmethod
    def _protobuf_compatibility_bootstrap() -> str | None:
        try:
            import google.protobuf  # noqa: F401
            return None
        except ImportError:
            base_site = Path(sys.prefix).parents[1] / "Lib" / "site-packages"
            if not base_site.exists():
                return None
            sys.path.append(str(base_site))
            try:
                import google.protobuf  # noqa: F401
                return str(base_site)
            finally:
                sys.path.remove(str(base_site))

    def _load(self) -> None:
        if self._model is not None:
            return
        self._protobuf_compatibility_bootstrap()
        import torch
        from transformers import SiglipModel, SiglipTokenizer

        device = self.device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = SiglipTokenizer.from_pretrained(
            self.config["model"],
            cache_dir=self.config["cache_dir"],
            local_files_only=self.config["local_files_only"],
        )
        self._model = SiglipModel.from_pretrained(
            self.config["model"],
            cache_dir=self.config["cache_dir"],
            local_files_only=self.config["local_files_only"],
        ).eval().to(device)
        self.device_name = device

    def encode(self, texts: list[str]) -> np.ndarray:
        self._load()
        import torch

        inputs = self._tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.config["max_text_tokens"],
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device_name) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self._model.get_text_features(**inputs)
            features = output.pooler_output if hasattr(output, "pooler_output") else output
            features = torch.nn.functional.normalize(features.float(), dim=-1)
        array = features.cpu().numpy().astype(np.float32)
        if array.shape[1] != self.config["embedding_dimension"]:
            raise PlannerError(f"SigLIP text embedding dimension mismatch: {array.shape}")
        return array


def resolve_embedding_path(index_path: Path, ref: dict[str, Any]) -> Path:
    path = Path(ref["path"])
    return path if path.is_absolute() else index_path.parent / path


def load_medium_embeddings(index: dict[str, Any], index_path: Path) -> np.ndarray:
    media = index["medium_nodes"]
    paths = {resolve_embedding_path(index_path, node["pooled_visual_embedding_ref"]) for node in media}
    if len(paths) != 1:
        raise PlannerError(f"Expected one Medium embedding matrix, got: {sorted(map(str, paths))}")
    path = next(iter(paths))
    if not path.exists():
        raise PlannerError(f"Missing embedding reference: {path}")
    matrix = np.load(path, mmap_mode="r")
    rows = [node["pooled_visual_embedding_ref"]["row"] for node in media]
    dimensions = {node["pooled_visual_embedding_ref"]["dimension"] for node in media}
    if dimensions != {768} or matrix.ndim != 2 or matrix.shape[1] != 768:
        raise PlannerError(f"Invalid Medium embedding dimensions: refs={dimensions}, matrix={matrix.shape}")
    if max(rows) >= matrix.shape[0]:
        raise PlannerError("Medium embedding row outside matrix")
    selected = np.asarray(matrix[rows], dtype=np.float32)
    if not np.isfinite(selected).all():
        raise PlannerError("Medium embeddings contain NaN/Infinity")
    return selected


def _coarse_text(node: dict[str, Any]) -> str:
    fields = [
        node.get("coarse_summary", ""),
        " ".join(node.get("actions", [])),
        " ".join(node.get("entities", [])),
        " ".join(node.get("locations", [])),
    ]
    return " ".join(fields)


def route_coarse_nodes(
    plan: dict[str, Any],
    index: dict[str, Any],
    question: dict[str, Any],
    config: dict[str, Any] = DEFAULT_CONFIG,
) -> dict[str, Any]:
    strategy = plan["retrieval_strategy"]
    coarse = index["coarse_nodes"]
    story = index["storyline_events"]
    all_coarse_ids = [node["coarse_id"] for node in coarse]
    all_story_ids = [node["storyline_event_id"] for node in story]
    if not index["capabilities"].get("has_storyline", False):
        return {
            "strategy": strategy,
            "selected_storyline_ids": [],
            "selected_coarse_ids": all_coarse_ids,
            "per_search_unit": {
                unit["unit_id"]: {
                    "selected_coarse_ids": all_coarse_ids,
                    "score_breakdown": [],
                }
                for unit in plan["search_units"]
            },
            "score_breakdown": [],
            "fallback_used": True,
            "fallback_reason": "has_storyline=false; direct all-Medium retrieval",
        }
    if strategy == "global_coverage":
        return {
            "strategy": strategy,
            "selected_storyline_ids": all_story_ids,
            "selected_coarse_ids": all_coarse_ids,
            "per_search_unit": {},
            "score_breakdown": [],
            "fallback_used": False,
        }

    direct_story = [value for value in plan["candidate_storyline_ids"] if value in all_story_ids]
    coarse_from_story: list[str] = []
    story_by_id = {node["storyline_event_id"]: node for node in story}
    for story_id in direct_story:
        coarse_from_story.extend(story_by_id[story_id]["source_coarse_ids"])
    direct_coarse = [
        value for value in plan["candidate_coarse_ids"] if value in all_coarse_ids
    ] + coarse_from_story
    direct_coarse = list(dict.fromkeys(direct_coarse))

    def rank_for_unit(unit: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]], bool]:
        text = query_text(question["question"], unit)
        rows = []
        for node in coarse:
            lexical, terms = lexical_similarity(text, _coarse_text(node))
            direct_bonus = 1.0 if node["coarse_id"] in direct_coarse else 0.0
            score = lexical + 0.25 * direct_bonus
            rows.append(
                {
                    "coarse_id": node["coarse_id"],
                    "lexical_score": lexical,
                    "planner_candidate_bonus": direct_bonus,
                    "routing_score": score,
                    "matched_terms": terms,
                }
            )
        rows.sort(key=lambda row: (-row["routing_score"], row["coarse_id"]))
        limit = config["selection"]["targeted_max_coarse"]
        minimum = config["selection"]["targeted_min_coarse"]
        selection = list(direct_coarse[:limit])
        for row in rows:
            if row["coarse_id"] not in selection and len(selection) < max(minimum, len(selection)):
                selection.append(row["coarse_id"])
        if not selection:
            selection = [row["coarse_id"] for row in rows[:minimum]]
        return selection[:limit], rows, not bool(direct_coarse)

    per_unit: dict[str, Any] = {}
    merged: list[str] = []
    all_scores: list[dict[str, Any]] = []
    fallback_used = False
    for unit in plan["search_units"]:
        selected, scores, fallback = rank_for_unit(unit)
        per_unit[unit["unit_id"]] = {"selected_coarse_ids": selected, "score_breakdown": scores}
        all_scores.extend({"search_unit_id": unit["unit_id"], **row} for row in scores)
        fallback_used = fallback_used or fallback
        merged.extend(selected)
    selected_ids = list(dict.fromkeys(merged))
    if strategy != "multi_target_compare":
        selected_ids = selected_ids[: config["selection"]["targeted_max_coarse"]]
    return {
        "strategy": strategy,
        "selected_storyline_ids": direct_story,
        "selected_coarse_ids": selected_ids,
        "per_search_unit": per_unit,
        "score_breakdown": all_scores,
        "fallback_used": fallback_used,
    }


def compute_medium_ranking(
    *,
    index: dict[str, Any],
    index_path: Path,
    question: dict[str, Any],
    plan: dict[str, Any],
    routing: dict[str, Any],
    query_embeddings: np.ndarray,
    medium_embeddings: np.ndarray,
    config: dict[str, Any] = DEFAULT_CONFIG,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    media = index["medium_nodes"]
    coarse_universe = set(routing["selected_coarse_ids"])
    strategy = plan["retrieval_strategy"]
    weights = config["weights"]
    per_unit_rows: dict[str, list[dict[str, Any]]] = {}
    aggregated: dict[str, dict[str, Any]] = {}
    for unit_index, unit in enumerate(plan["search_units"]):
        if strategy == "multi_target_compare":
            allowed = set(routing["per_search_unit"][unit["unit_id"]]["selected_coarse_ids"])
        elif strategy == "global_coverage":
            allowed = {node["coarse_id"] for node in index["coarse_nodes"]}
        else:
            allowed = coarse_universe
        indices = [i for i, node in enumerate(media) if node["parent_coarse_id"] in allowed]
        if not indices and not index["capabilities"].get("has_storyline", False):
            indices = list(range(len(media)))
        if not indices:
            raise PlannerError(f"No Medium candidates for search unit {unit['unit_id']}")
        raw = medium_embeddings[indices] @ query_embeddings[unit_index]
        normalized = _minmax(raw.astype(np.float64))
        rows = []
        qtext = query_text(question["question"], unit)
        for local_index, medium_index in enumerate(indices):
            node = media[medium_index]
            lexical, matched = lexical_similarity(qtext, node["qwen_caption"])
            prior = 1.0 if node["parent_coarse_id"] in allowed else 0.0
            combined = (
                weights["visual_score"] * float(normalized[local_index])
                + weights["lexical_score"] * lexical
                + weights["coarse_prior"] * prior
            )
            row = {
                "medium_id": node["medium_id"],
                "start_sec": node["start_sec"],
                "end_sec": node["end_sec"],
                "parent_coarse_id": node["parent_coarse_id"],
                "caption": node["qwen_caption"],
                "visual_score_raw": float(raw[local_index]),
                "visual_score_normalized": float(normalized[local_index]),
                "lexical_score": lexical,
                "matched_terms": matched,
                "coarse_prior": prior,
                "coarse_prior_source": "routed_coarse_membership",
                "combined_score": combined,
                "matched_search_unit_ids": [unit["unit_id"]],
                "child_fine_count": len(node["child_fine_ids"]),
                "child_fine_ids": list(node["child_fine_ids"]),
            }
            rows.append(row)
        rows.sort(key=lambda row: (-row["combined_score"], row["start_sec"], row["medium_id"]))
        for rank, row in enumerate(rows, 1):
            row["search_unit_rank"] = rank
        per_unit_rows[unit["unit_id"]] = rows
        for row in rows:
            existing = aggregated.get(row["medium_id"])
            if existing is None or row["combined_score"] > existing["combined_score"]:
                merged = dict(row)
                if existing:
                    merged["matched_search_unit_ids"] = list(
                        dict.fromkeys(existing["matched_search_unit_ids"] + row["matched_search_unit_ids"])
                    )
                aggregated[row["medium_id"]] = merged
            elif unit["unit_id"] not in existing["matched_search_unit_ids"]:
                existing["matched_search_unit_ids"].append(unit["unit_id"])
    ranking = sorted(
        aggregated.values(),
        key=lambda row: (-row["combined_score"], row["start_sec"], row["medium_id"]),
    )
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank

    selected: list[dict[str, Any]] = []
    if strategy == "global_coverage":
        for coarse_node in index["coarse_nodes"]:
            candidates = [row for row in ranking if row["parent_coarse_id"] == coarse_node["coarse_id"]]
            selected.extend(candidates[: config["selection"]["global_per_coarse"]])
    elif strategy == "multi_target_compare":
        ids: list[str] = []
        for unit in plan["search_units"]:
            ids.extend(
                row["medium_id"]
                for row in per_unit_rows[unit["unit_id"]][
                    : config["selection"]["multi_target_per_unit"]
                ]
            )
        ids = list(dict.fromkeys(ids))[: config["selection"]["multi_target_max_merged"]]
        by_id = {row["medium_id"]: row for row in ranking}
        selected = [by_id[value] for value in ids]
    else:
        selected = ranking[: config["selection"]["targeted_max_mediums"]]
    return ranking, selected


def build_planner_input(
    question: dict[str, Any], index: dict[str, Any]
) -> dict[str, Any]:
    return {
        "question_id": question["question_id"],
        "question": question["question"],
        "answer_options": question.get("answer_options", []),
        "capabilities": index["capabilities"],
        "storyline_events": [
            {
                "storyline_event_id": node["storyline_event_id"],
                "start_sec": node["start_sec"],
                "end_sec": node["end_sec"],
                "summary": node["summary"],
                "source_coarse_ids": node["source_coarse_ids"],
            }
            for node in index["storyline_events"]
        ],
        "coarse_nodes": [
            {
                "coarse_id": node["coarse_id"],
                "start_sec": node["start_sec"],
                "end_sec": node["end_sec"],
                "coarse_summary": node["coarse_summary"],
                "child_medium_ids": node["child_medium_ids"],
            }
            for node in index["coarse_nodes"]
        ],
        "available_node_ids": {
            "storyline": [node["storyline_event_id"] for node in index["storyline_events"]],
            "coarse": [node["coarse_id"] for node in index["coarse_nodes"]],
        },
        "available_retrieval_channels": [
            channel
            for channel, capability in {
                "visual": "has_visual_embeddings",
                "audio": "has_asr",
                "detector": "has_detector_tags",
                "tracking": "has_tracking",
            }.items()
            if index["capabilities"].get(capability, False)
        ],
    }


def call_anthropic(
    *,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, dict[str, Any]]:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    started = time.perf_counter()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=PLANNER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        output_config={"format": {"type": "json_schema", "schema": planner_json_schema()}},
    )
    latency = time.perf_counter() - started
    raw = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )
    usage = {
        "provider": "anthropic",
        "model": model,
        "input_tokens": int(response.usage.input_tokens),
        "output_tokens": int(response.usage.output_tokens),
        "cache_creation_input_tokens": int(
            getattr(response.usage, "cache_creation_input_tokens", 0) or 0
        ),
        "cache_read_input_tokens": int(
            getattr(response.usage, "cache_read_input_tokens", 0) or 0
        ),
        "latency_sec": latency,
        "stop_reason": getattr(response, "stop_reason", None),
        "response_id": str(getattr(response, "id", "") or ""),
        "request_id": str(getattr(response, "_request_id", "") or ""),
        "estimated_cost_usd": None,
        "cost_note": "No reliable pricing table configured; cost not fabricated.",
    }
    if usage["stop_reason"] == "max_tokens":
        raise PlannerError("Planner response reached max_tokens")
    return raw, usage


def plan_question(
    *,
    question: dict[str, Any],
    index: dict[str, Any],
    output_dir: Path,
    api_key: str,
    config: dict[str, Any],
    caller: Callable[..., tuple[str, dict[str, Any]]] = call_anthropic,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    planner_input = build_planner_input(question, index)
    user_prompt = "PLANNER INPUT:\n" + json.dumps(
        planner_input, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "planner_prompt.txt").write_text(
        PLANNER_SYSTEM_PROMPT + "\n\n" + user_prompt + "\n", encoding="utf-8"
    )
    attempts: list[dict[str, Any]] = []
    raw, usage = caller(
        api_key=api_key,
        model=config["planner"]["model"],
        prompt=user_prompt,
        max_tokens=config["planner"]["max_tokens"],
        temperature=config["planner"]["temperature"],
    )
    (output_dir / "planner_raw_response.txt").write_text(raw, encoding="utf-8")
    attempts.append({"attempt": 1, "repair": False, "usage": usage})
    try:
        parsed = json.loads(raw)
        validation = validate_planner_output(parsed, question=question, index=index)
    except Exception as error:
        parsed = None
        validation = {"valid": False, "errors": [f"json_parse_error:{type(error).__name__}:{error}"]}
    if not validation["valid"]:
        repair_prompt = (
            user_prompt
            + "\n\nINVALID OUTPUT:\n"
            + raw
            + "\n\nVALIDATION ERRORS:\n"
            + json.dumps(validation["errors"], ensure_ascii=False)
            + "\n\n"
            + REPAIR_INSTRUCTION
        )
        repair_raw, repair_usage = caller(
            api_key=api_key,
            model=config["planner"]["model"],
            prompt=repair_prompt,
            max_tokens=config["planner"]["max_tokens"],
            temperature=config["planner"]["temperature"],
        )
        (output_dir / "planner_repair_response.txt").write_text(repair_raw, encoding="utf-8")
        attempts.append({"attempt": 2, "repair": True, "usage": repair_usage})
        try:
            parsed = json.loads(repair_raw)
            validation = validate_planner_output(parsed, question=question, index=index)
        except Exception as error:
            validation = {
                "valid": False,
                "errors": [f"repair_json_parse_error:{type(error).__name__}:{error}"],
            }
    if not validation["valid"] or parsed is None:
        write_json(output_dir / "api_usage.json", {"attempts": attempts})
        raise PlannerError(
            f"Planner failed after {len(attempts)} attempt(s): {validation['errors']}"
        )
    write_json(output_dir / "planner_parsed.json", parsed)
    totals = {
        "calls": len(attempts),
        "repair_calls": sum(bool(row["repair"]) for row in attempts),
        "input_tokens": sum(row["usage"]["input_tokens"] for row in attempts),
        "output_tokens": sum(row["usage"]["output_tokens"] for row in attempts),
        "latency_sec": sum(row["usage"]["latency_sec"] for row in attempts),
        "estimated_cost_usd": None,
    }
    write_json(output_dir / "api_usage.json", {"attempts": attempts, "totals": totals})
    return parsed, attempts, validation


def _manifest(index_path: Path, index: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    refs = sorted(
        {
            node["pooled_visual_embedding_ref"]["path"]
            for node in index["medium_nodes"]
        }
    )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(index_path.resolve()),
        "sha256": sha256_file(index_path),
        "file_size_bytes": index_path.stat().st_size,
        "schema_version": index["schema_version"],
        "video_id": index["video"]["video_id"],
        "counts": {
            "fine": len(index["fine_nodes"]),
            "medium": len(index["medium_nodes"]),
            "coarse": len(index["coarse_nodes"]),
            "storyline": len(index["storyline_events"]),
        },
        "capabilities": index["capabilities"],
        "medium_embedding_references": refs,
        "medium_embedding_dimension": 768,
        "planner_api": {
            "provider": "anthropic",
            **config["planner"],
            "authentication": "ANTHROPIC_API_KEY environment variable; key is never logged",
            "structured_output": "JSON schema",
        },
    }


def _render_review(
    *,
    index: dict[str, Any],
    questions: list[dict[str, Any]],
    results: dict[str, Any],
    path: Path,
) -> None:
    coarse_by_id = {node["coarse_id"]: node for node in index["coarse_nodes"]}
    sections = []
    for question in questions:
        result = results[question["question_id"]]
        plan = result["plan"]
        routing = result["routing"]
        selected_ids = {row["medium_id"] for row in result["selected"]}
        ranking_rows = result["ranking"]
        coarse_html = "".join(
            f"<li><b>{html.escape(cid)}</b>: {html.escape(coarse_by_id[cid]['coarse_summary'])}</li>"
            for cid in routing["selected_coarse_ids"]
        )
        unit_html = "".join(
            "<li><b>{}</b> — {}<br><small>{}</small></li>".format(
                html.escape(unit["unit_id"]),
                html.escape(unit["description"]),
                html.escape(" | ".join(unit["query_variants"])),
            )
            for unit in plan["search_units"]
        )
        rows = "".join(
            "<tr><td>{rank}</td><td>{selected}</td><td>{mid}</td><td>{time}</td><td>{coarse}</td>"
            "<td>{caption}</td><td>{visual:.4f}</td><td>{lexical:.4f}</td>"
            "<td>{prior:.2f}</td><td>{combined:.4f}</td><td>{terms}</td><td>{fine}</td></tr>".format(
                rank=row["rank"],
                selected="✓" if row["medium_id"] in selected_ids else "",
                mid=html.escape(row["medium_id"]),
                time=f"{row['start_sec']:.1f}–{row['end_sec']:.1f}",
                coarse=html.escape(row["parent_coarse_id"]),
                caption=html.escape(row["caption"]),
                visual=row["visual_score_normalized"],
                lexical=row["lexical_score"],
                prior=row["coarse_prior"],
                combined=row["combined_score"],
                terms=html.escape(", ".join(row["matched_terms"]) or "—"),
                fine=row["child_fine_count"],
            )
            for row in ranking_rows
        )
        sections.append(
            f"""<section><h2>{html.escape(question['question_id'])}</h2>
<p class="question">{html.escape(question['question'])}</p>
<div class="grid"><div><h3>Planner</h3><p>{plan['scope']} · {plan['operation']} ·
{plan['retrieval_strategy']}</p><h4>Search units</h4><ul>{unit_html}</ul>
<h4>Required evidence</h4><pre>{html.escape(json.dumps(plan['required_evidence'], indent=2))}</pre>
</div><div><h3>Routing</h3><p>Storyline: {html.escape(', '.join(routing['selected_storyline_ids']) or 'none')}</p>
<ul>{coarse_html}</ul></div></div>
<h3>Complete ranked Medium list (✓ = selected; no final evidence/answer)</h3>
<div class="table"><table><thead><tr><th>Rank</th><th>Selected</th><th>Medium</th><th>Time</th><th>Coarse</th>
<th>Caption</th><th>Visual</th><th>Lexical</th><th>Prior</th><th>Combined</th>
<th>Terms</th><th>Fine count</th></tr></thead><tbody>{rows}</tbody></table></div></section>"""
        )
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>Planner Medium Retrieval v1</title>
<style>body{{font:14px system-ui;margin:24px;color:#20242a}}section{{border-top:3px solid #334e68;padding:18px 0}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:22px}}.question{{font-size:17px}}pre{{white-space:pre-wrap}}
.table{{overflow:auto}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd;padding:6px;vertical-align:top}}
th{{background:#eef3f8;position:sticky;top:0}}small{{color:#52606d}}</style></head>
<body><h1>Planner + Medium Retrieval v1 — 226</h1>
<p>Read-only diagnostic over the frozen hierarchical index. No Fine reranking and no final answer generation.</p>
{''.join(sections)}</body></html>"""
    path.write_text(document, encoding="utf-8")


def validate_saved_outputs(
    *,
    index_path: Path,
    questions_path: Path,
    video_dir: Path,
    config: dict[str, Any] = DEFAULT_CONFIG,
) -> dict[str, Any]:
    index = load_json(index_path)
    questions = load_json(questions_path)["questions"]
    checks: dict[str, bool] = {}
    errors: list[str] = []
    warnings: list[str] = []
    expected_hash = load_json(video_dir / "frozen_input_manifest.json")["sha256"]
    checks["frozen_input_sha256_unchanged"] = sha256_file(index_path) == expected_hash
    checks["embedding_dimension_768"] = load_medium_embeddings(index, index_path).shape == (30, 768)
    checks["all_outputs_json_reload"] = True
    results: dict[str, Any] = {}
    all_medium_ids = {node["medium_id"] for node in index["medium_nodes"]}
    source_text = " ".join(
        [question["question"] for question in questions]
        + [node["summary"] for node in index["storyline_events"]]
        + [node["coarse_summary"] for node in index["coarse_nodes"]]
    ).lower()
    for question in questions:
        qid = question["question_id"]
        qdir = video_dir / qid
        try:
            plan = load_json(qdir / "planner_parsed.json")
            routing = load_json(qdir / "routing_result.json")
            ranking = load_json(qdir / "medium_ranking.json")
            selected = load_json(qdir / "selected_mediums.json")
            load_json(qdir / "api_usage.json")
            load_json(qdir / "provenance.json")
        except Exception as error:
            checks["all_outputs_json_reload"] = False
            errors.append(f"{qid}:json_reload:{type(error).__name__}:{error}")
            continue
        plan_validation = validate_planner_output(plan, question=question, index=index)
        checks[f"{qid}:planner_schema_and_contract"] = plan_validation["valid"]
        errors.extend(f"{qid}:{error}" for error in plan_validation["errors"])
        allowed = set(routing["selected_coarse_ids"])
        ranking_errors = validate_ranking(
            ranking,
            medium_ids=all_medium_ids,
            allowed_coarse_ids=allowed,
            weights=config["weights"],
        )
        checks[f"{qid}:ranking_ids_universe_formula_finite"] = not ranking_errors
        errors.extend(f"{qid}:{error}" for error in ranking_errors)
        selected_ids = [row["medium_id"] for row in selected]
        checks[f"{qid}:selected_medium_ids_exist"] = set(selected_ids) <= all_medium_ids
        checks[f"{qid}:selected_mediums_within_ranking"] = set(selected_ids) <= {
            row["medium_id"] for row in ranking
        }
        if qid == "q_global_summary":
            checks["global_summary_all_storyline"] = routing["selected_storyline_ids"] == [
                node["storyline_event_id"] for node in index["storyline_events"]
            ]
            checks["global_summary_all_coarse"] = routing["selected_coarse_ids"] == [
                node["coarse_id"] for node in index["coarse_nodes"]
            ]
            checks["global_summary_one_medium_per_coarse"] = {
                row["parent_coarse_id"] for row in selected
            } == {node["coarse_id"] for node in index["coarse_nodes"]}
        if qid == "q_handcuff_before_medical":
            checks["multi_target_two_independent_units"] = (
                len(plan["search_units"]) == 2
                and len(routing["per_search_unit"]) == 2
            )
        rendered_plan = json.dumps(plan, ensure_ascii=False).lower()
        if "firearm discharge" in rendered_plan and "firearm discharge" not in source_text:
            warnings.append(
                f"{qid}:search-unit phrase 'firearm discharge' is more specific than the supplied index text"
            )
        if re.search(r"\bvictim\b", rendered_plan) and not re.search(r"\bvictim\b", source_text):
            warnings.append(
                f"{qid}:search-unit role 'victim' is absent from the supplied question/index overview"
            )
        results[qid] = {"plan": plan, "routing": routing, "ranking": ranking, "selected": selected}
    checks["no_answer_fields"] = all(
        not any(
            key in json.dumps(result["plan"]).lower()
            for key in ('"answer"', '"final_answer"', '"selected_option"', '"conclusion"')
        )
        for result in results.values()
    )
    checks["question_count_6"] = len(results) == 6
    checks["all_required_checks_pass"] = all(checks.values())
    if not checks["all_required_checks_pass"]:
        errors.extend(name for name, passed in checks.items() if not passed)
    report = {
        "valid": not errors,
        "checks": checks,
        "errors": errors,
        "warnings": warnings
        + [
            "This diagnostic stops at Medium retrieval; selected Mediums are not final evidence.",
            "API cost is null because no reliable pricing table is configured.",
        ],
        "frozen_input_sha256_before": expected_hash,
        "frozen_input_sha256_after": sha256_file(index_path),
        "frozen_input_unchanged": checks["frozen_input_sha256_unchanged"],
        "counts": {"questions": len(results), "mediums": len(index["medium_nodes"])},
    }
    write_json(video_dir / "validation_report.json", report)
    _render_review(index=index, questions=questions, results=results, path=video_dir / "review.html")
    return report


def run_experiment(
    *,
    root: Path,
    index_path: Path,
    questions_path: Path,
    output_root: Path,
    api_key: str,
    config: dict[str, Any] = DEFAULT_CONFIG,
    encoder: SiglipTextEncoder | None = None,
) -> dict[str, Any]:
    before_hash = sha256_file(index_path)
    index = load_json(index_path)
    questions_payload = load_json(questions_path)
    questions = questions_payload["questions"]
    video_dir = output_root / "226"
    video_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(index_path, index, config)
    write_json(video_dir / "frozen_input_manifest.json", manifest)
    write_json(video_dir / "questions_226.json", questions_payload)
    medium_embeddings = load_medium_embeddings(index, index_path)
    encoder = encoder or SiglipTextEncoder(config["siglip"])
    results: dict[str, Any] = {}
    planner_rows: list[dict[str, Any]] = []
    validation_errors: list[str] = []
    for question in questions:
        qdir = video_dir / question["question_id"]
        plan, attempts, plan_validation = plan_question(
            question=question,
            index=index,
            output_dir=qdir,
            api_key=api_key,
            config=config,
        )
        routing = route_coarse_nodes(plan, index, question, config)
        texts = [query_text(question["question"], unit) for unit in plan["search_units"]]
        qemb = encoder.encode(texts)
        ranking, selected = compute_medium_ranking(
            index=index,
            index_path=index_path,
            question=question,
            plan=plan,
            routing=routing,
            query_embeddings=qemb,
            medium_embeddings=medium_embeddings,
            config=config,
        )
        allowed = set(routing["selected_coarse_ids"])
        ranking_errors = validate_ranking(
            ranking,
            medium_ids={node["medium_id"] for node in index["medium_nodes"]},
            allowed_coarse_ids=allowed,
            weights=config["weights"],
        )
        if ranking_errors:
            raise PlannerError(f"Ranking validation failed for {question['question_id']}: {ranking_errors}")
        write_json(qdir / "routing_result.json", routing)
        write_json(qdir / "medium_ranking.json", ranking)
        write_json(qdir / "selected_mediums.json", selected)
        write_json(
            qdir / "provenance.json",
            {
                "frozen_index_sha256": before_hash,
                "config": config,
                "question_source": str(questions_path.resolve()),
                "query_texts": texts,
                "siglip_device": encoder.device_name,
                "no_fine_reranking": True,
                "no_final_answer": True,
            },
        )
        usage = load_json(qdir / "api_usage.json")["totals"]
        planner_rows.append(
            {
                "question_id": question["question_id"],
                "scope": plan["scope"],
                "operation": plan["operation"],
                "retrieval_strategy": plan["retrieval_strategy"],
                "selected_storyline_ids": routing["selected_storyline_ids"],
                "selected_coarse_ids": routing["selected_coarse_ids"],
                "selected_medium_ids": [row["medium_id"] for row in selected],
                "planner_calls": usage["calls"],
                "repair_calls": usage["repair_calls"],
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "latency_sec": usage["latency_sec"],
            }
        )
        results[question["question_id"]] = {
            "plan": plan,
            "routing": routing,
            "ranking": ranking,
            "selected": selected,
        }
    after_hash = sha256_file(index_path)
    if after_hash != before_hash:
        raise PlannerError("Frozen input hash changed during run")
    total_usage = {
        "calls": sum(row["planner_calls"] for row in planner_rows),
        "repair_calls": sum(row["repair_calls"] for row in planner_rows),
        "input_tokens": sum(row["input_tokens"] for row in planner_rows),
        "output_tokens": sum(row["output_tokens"] for row in planner_rows),
        "latency_sec": sum(row["latency_sec"] for row in planner_rows),
        "estimated_cost_usd": None,
    }
    write_json(video_dir / "planner_summary.json", {"questions": planner_rows, "totals": total_usage})
    write_json(
        video_dir / "retrieval_summary.json",
        {
            "weights": config["weights"],
            "selection": config["selection"],
            "questions": [
                {
                    "question_id": row["question_id"],
                    "selected_medium_ids": row["selected_medium_ids"],
                    "selected_count": len(row["selected_medium_ids"]),
                }
                for row in planner_rows
            ],
        },
    )
    validation_report = {
        "valid": not validation_errors,
        "errors": validation_errors,
        "warnings": [
            "This diagnostic stops at Medium retrieval; selected Mediums are not final evidence.",
            "API cost is not fabricated because no reliable price table is configured.",
        ],
        "frozen_input_sha256_before": before_hash,
        "frozen_input_sha256_after": after_hash,
        "frozen_input_unchanged": before_hash == after_hash,
        "counts": {"questions": len(questions), "mediums": len(index["medium_nodes"])},
    }
    write_json(video_dir / "validation_report.json", validation_report)
    write_json(
        video_dir / "provenance.json",
        {
            "experiment": "planner_medium_retrieval_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "frozen_input_manifest": "frozen_input_manifest.json",
            "questions": str(questions_path.resolve()),
            "config": config,
            "final_answer_model_run": False,
        },
    )
    _render_review(index=index, questions=questions, results=results, path=video_dir / "review.html")
    return {
        "valid": validation_report["valid"],
        "output_dir": str(video_dir),
        "planner_summary": planner_rows,
        "totals": total_usage,
        "frozen_hash": before_hash,
    }
