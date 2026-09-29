from __future__ import annotations

import base64
import hashlib
import html
import json
import math
import mimetypes
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.experiments.planner_medium_retrieval.core import SiglipTextEncoder

FORBIDDEN_FIELDS = {"answer", "final_answer", "selected_option", "conclusion"}
SELECTION_REASONS = {"top_relevance", "diverse_second"}


class FineRerankingError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_fine_query_text(question: str, search_unit: dict[str, Any]) -> str:
    return (
        f"{question} Retrieval target: {search_unit['description']}. "
        f"Variants: {' ; '.join(search_unit['query_variants'])}"
    )


def resolve_embedding_path(index_path: Path, ref: dict[str, Any]) -> Path:
    path = Path(ref["path"])
    return path if path.is_absolute() else index_path.parent / path


def load_fine_embeddings(
    index: dict[str, Any], index_path: Path
) -> tuple[np.ndarray, dict[str, int], Path]:
    fine_nodes = index["fine_nodes"]
    paths = {resolve_embedding_path(index_path, node["visual_embedding_ref"]) for node in fine_nodes}
    if len(paths) != 1:
        raise FineRerankingError(f"Expected one Fine embedding matrix, got {paths}")
    path = next(iter(paths))
    if not path.exists():
        raise FineRerankingError(f"Missing Fine embedding file: {path}")
    matrix = np.load(path, mmap_mode="r")
    if matrix.shape != (88, 768) or matrix.dtype != np.float32:
        raise FineRerankingError(f"Expected Fine embedding shape (88, 768) float32, got {matrix.shape} {matrix.dtype}")
    row_by_id: dict[str, int] = {}
    for node in fine_nodes:
        ref = node["visual_embedding_ref"]
        if ref["dimension"] != 768:
            raise FineRerankingError(f"Invalid Fine embedding dimension: {node['fine_id']}")
        row = int(ref["row"])
        if row < 0 or row >= matrix.shape[0]:
            raise FineRerankingError(f"Fine embedding row outside matrix: {node['fine_id']}")
        row_by_id[node["fine_id"]] = row
    selected = np.asarray(matrix, dtype=np.float32)
    if not np.isfinite(selected).all():
        raise FineRerankingError("Fine embedding contains NaN/Infinity")
    return selected, row_by_id, path


def validate_frozen_inputs(
    *,
    index: dict[str, Any],
    index_path: Path,
    planner_dir: Path,
) -> None:
    if (
        len(index["fine_nodes"]),
        len(index["medium_nodes"]),
        len(index["coarse_nodes"]),
        len(index["storyline_events"]),
    ) != (88, 30, 10, 3):
        raise FineRerankingError("Frozen hierarchy counts differ from 88/30/10/3")
    fine_by_id = {node["fine_id"]: node for node in index["fine_nodes"]}
    medium_by_id = {node["medium_id"]: node for node in index["medium_nodes"]}
    load_fine_embeddings(index, index_path)
    for fine in index["fine_nodes"]:
        frame = Path(fine["representative_frame_path"])
        if not frame.exists() or not frame.is_file():
            raise FineRerankingError(f"Missing representative frame: {frame}")
    for medium in index["medium_nodes"]:
        for fine_id in medium["child_fine_ids"]:
            if fine_id not in fine_by_id:
                raise FineRerankingError(
                    f"Invalid child Fine ID: {medium['medium_id']}:{fine_id}"
                )
    question_dirs = sorted(path for path in planner_dir.glob("q_*") if path.is_dir())
    if len(question_dirs) != 6:
        raise FineRerankingError(f"Expected six frozen question directories, got {len(question_dirs)}")
    for qdir in question_dirs:
        for filename in (
            "planner_parsed.json",
            "routing_result.json",
            "medium_ranking.json",
            "selected_mediums.json",
        ):
            if not (qdir / filename).exists():
                raise FineRerankingError(f"Missing frozen Planner artifact: {qdir / filename}")
        selected = load_json(qdir / "selected_mediums.json")
        for row in selected:
            medium_id = row["medium_id"]
            if medium_id not in medium_by_id:
                raise FineRerankingError(f"Unknown selected Medium: {medium_id}")
            expected = medium_by_id[medium_id]["child_fine_ids"]
            if row.get("child_fine_ids") != expected:
                raise FineRerankingError(f"Selected Medium child_fine_ids mismatch: {medium_id}")
            for fine_id in expected:
                if fine_id not in fine_by_id:
                    raise FineRerankingError(f"Invalid child Fine ID: {medium_id}:{fine_id}")


def build_frozen_manifest(index_path: Path, planner_dir: Path) -> dict[str, Any]:
    files = [index_path, planner_dir / "questions_226.json"]
    for qdir in sorted(path for path in planner_dir.glob("q_*") if path.is_dir()):
        files.extend(
            qdir / filename
            for filename in (
                "planner_parsed.json",
                "routing_result.json",
                "medium_ranking.json",
                "selected_mediums.json",
            )
        )
    records = [
        {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "fine_reranking_v1",
        "video_id": "copa/2021-1076/540772226",
        "files": records,
        "file_count": len(records),
    }


def _normalize(scores: np.ndarray) -> np.ndarray:
    low, high = float(scores.min()), float(scores.max())
    if math.isclose(low, high):
        return np.ones_like(scores, dtype=np.float64)
    return (scores - low) / (high - low)


def rerank_fines_for_medium(
    *,
    question_id: str,
    search_unit_id: str,
    medium: dict[str, Any],
    fine_by_id: dict[str, dict[str, Any]],
    fine_embeddings: np.ndarray,
    row_by_id: dict[str, int],
    query_embedding: np.ndarray,
) -> list[dict[str, Any]]:
    child_ids = medium["child_fine_ids"]
    if not child_ids:
        raise FineRerankingError(f"Medium has no child Fine: {medium['medium_id']}")
    try:
        rows = [row_by_id[fine_id] for fine_id in child_ids]
    except KeyError as error:
        raise FineRerankingError(f"Invalid child Fine ID: {error.args[0]}") from error
    raw = fine_embeddings[rows] @ query_embedding
    normalized = _normalize(raw.astype(np.float64))
    ranking: list[dict[str, Any]] = []
    for offset, fine_id in enumerate(child_ids):
        fine = fine_by_id[fine_id]
        frame = Path(fine["representative_frame_path"])
        if not frame.exists():
            raise FineRerankingError(f"Missing representative frame: {frame}")
        ranking.append(
            {
                "question_id": question_id,
                "search_unit_id": search_unit_id,
                "medium_id": medium["medium_id"],
                "fine_id": fine_id,
                "start_sec": fine["start_sec"],
                "end_sec": fine["end_sec"],
                "representative_frame_timestamp_sec": fine["representative_frame_timestamp_sec"],
                "representative_frame_path": fine["representative_frame_path"],
                "siglip_score_raw": float(raw[offset]),
                "siglip_score_normalized": float(normalized[offset]),
            }
        )
    ranking.sort(
        key=lambda row: (
            -row["siglip_score_raw"],
            row["representative_frame_timestamp_sec"],
            row["fine_id"],
        )
    )
    for rank, row in enumerate(ranking, 1):
        row["rank_within_medium"] = rank
    return ranking


def temporally_diverse(first: dict[str, Any], candidate: dict[str, Any], minimum_gap: float) -> bool:
    timestamp_gap = abs(
        float(first["representative_frame_timestamp_sec"])
        - float(candidate["representative_frame_timestamp_sec"])
    )
    different_nonidentical_interval = (
        first["fine_id"] != candidate["fine_id"]
        and (
            float(first["start_sec"]) != float(candidate["start_sec"])
            or float(first["end_sec"]) != float(candidate["end_sec"])
        )
        and not (
            float(first["start_sec"]) <= float(candidate["start_sec"])
            and float(first["end_sec"]) >= float(candidate["end_sec"])
            and float(candidate["start_sec"]) <= float(first["start_sec"])
            and float(candidate["end_sec"]) >= float(first["end_sec"])
        )
    )
    return timestamp_gap >= minimum_gap or different_nonidentical_interval


def select_diverse_fines(
    rankings_by_unit: dict[str, list[dict[str, Any]]],
    *,
    strategy: str,
    max_fines: int,
    minimum_gap: float,
) -> tuple[list[dict[str, Any]], str | None]:
    if not rankings_by_unit:
        raise FineRerankingError("No search-unit rankings")
    chosen: list[dict[str, Any]] = []
    support: dict[str, list[str]] = {}
    candidates_by_id: dict[str, dict[str, Any]] = {}
    for unit_id, rows in rankings_by_unit.items():
        for row in rows:
            current = candidates_by_id.get(row["fine_id"])
            if current is None or row["siglip_score_raw"] > current["siglip_score_raw"]:
                candidates_by_id[row["fine_id"]] = row
        if strategy == "multi_target_compare" and rows:
            top = rows[0]
            support.setdefault(top["fine_id"], []).append(unit_id)
            if top["fine_id"] not in {item["fine_id"] for item in chosen}:
                if not chosen or temporally_diverse(chosen[0], top, minimum_gap):
                    chosen.append(top)
            if len(chosen) >= max_fines:
                break
    ordered = sorted(
        candidates_by_id.values(),
        key=lambda row: (
            -row["siglip_score_raw"],
            row["representative_frame_timestamp_sec"],
            row["fine_id"],
        ),
    )
    if not chosen:
        chosen.append(ordered[0])
    for candidate in ordered:
        if len(chosen) >= max_fines:
            break
        if candidate["fine_id"] in {item["fine_id"] for item in chosen}:
            continue
        if temporally_diverse(chosen[0], candidate, minimum_gap):
            chosen.append(candidate)
    for row in chosen:
        if row["fine_id"] not in support:
            best_units = [
                unit_id
                for unit_id, rows in rankings_by_unit.items()
                if rows and any(item["fine_id"] == row["fine_id"] for item in rows[:2])
            ]
            support[row["fine_id"]] = best_units or [row["search_unit_id"]]
    for index, row in enumerate(chosen):
        row["selection_reason"] = "top_relevance" if index == 0 else "diverse_second"
        row["search_unit_ids"] = list(dict.fromkeys(support[row["fine_id"]]))
    if len(chosen) == 1:
        only_reason = (
            "medium_has_one_child_fine"
            if len(candidates_by_id) == 1
            else "no_temporally_diverse_second"
        )
    else:
        only_reason = None
    return chosen, only_reason


def _storyline_ids_for_coarse(index: dict[str, Any], coarse_id: str) -> list[str]:
    return [
        event["storyline_event_id"]
        for event in index["storyline_events"]
        if coarse_id in event["source_coarse_ids"]
    ]


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(FORBIDDEN_FIELDS & set(value)) or any(
            _contains_forbidden_key(child) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(child) for child in value)
    return False


def validate_outputs(
    *,
    index: dict[str, Any],
    planner_dir: Path,
    output_dir: Path,
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}
    for record in manifest["files"]:
        path = Path(record["path"])
        if not path.exists() or sha256_file(path) != record["sha256"]:
            errors.append(f"frozen_input_hash_changed:{path}")
    checks["all_frozen_input_hashes_unchanged"] = not errors
    fine_by_id = {node["fine_id"]: node for node in index["fine_nodes"]}
    medium_by_id = {node["medium_id"]: node for node in index["medium_nodes"]}
    question_dirs = sorted(path for path in output_dir.glob("q_*") if path.is_dir())
    checks["six_question_outputs"] = len(question_dirs) == 6
    for qdir in question_dirs:
        qid = qdir.name
        selected_mediums = load_json(planner_dir / qid / "selected_mediums.json")
        selected = load_json(qdir / "selected_fines.json")
        ranking = load_json(qdir / "fine_ranking.json")
        planner = load_json(planner_dir / qid / "planner_parsed.json")
        counts = Counter(row["medium_id"] for row in selected)
        if any(value > config["max_fine_per_medium"] for value in counts.values()):
            errors.append(f"{qid}:more_than_two_fines_per_medium")
        allowed_mediums = {row["medium_id"] for row in selected_mediums}
        if not {row["medium_id"] for row in selected} <= allowed_mediums:
            errors.append(f"{qid}:fine_from_unselected_medium")
        for row in selected:
            if row["fine_id"] not in fine_by_id:
                errors.append(f"{qid}:unknown_fine:{row['fine_id']}")
                continue
            if row["fine_id"] not in medium_by_id[row["medium_id"]]["child_fine_ids"]:
                errors.append(f"{qid}:fine_not_child_of_medium:{row['fine_id']}")
            if row["selection_reason"] not in SELECTION_REASONS:
                errors.append(f"{qid}:invalid_selection_reason")
            if not Path(row["representative_frame_path"]).exists():
                errors.append(f"{qid}:missing_frame:{row['fine_id']}")
            if not math.isfinite(row["siglip_score_raw"]):
                errors.append(f"{qid}:nonfinite_selected_score")
        for row in ranking:
            if not all(
                math.isfinite(row[field])
                for field in ("siglip_score_raw", "siglip_score_normalized")
            ):
                errors.append(f"{qid}:nonfinite_ranking_score")
        if planner["retrieval_strategy"] == "global_coverage":
            selected_coarse = {
                row["parent_coarse_id"] for row in selected
            }
            expected_coarse = {
                row["parent_coarse_id"] for row in selected_mediums
            }
            if selected_coarse != expected_coarse:
                errors.append(f"{qid}:global_coarse_coverage_lost")
        if planner["retrieval_strategy"] == "multi_target_compare":
            units = {unit["unit_id"] for unit in planner["search_units"]}
            ranking_units = {row["search_unit_id"] for row in ranking}
            if units != ranking_units:
                errors.append(f"{qid}:multi_target_units_not_separate")
    candidates = load_json(output_dir / "evidence_candidates.json")
    checks["evidence_candidates_reload"] = len(candidates["questions"]) == 6
    checks["no_answer_fields"] = not _contains_forbidden_key(candidates)
    if not checks["no_answer_fields"]:
        errors.append("forbidden_answer_field")
    checks["all_constraints_pass"] = not errors
    return {
        "valid": not errors,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "frozen_file_count": manifest["file_count"],
    }


def _image_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def render_review(
    *,
    questions: list[dict[str, Any]],
    planner_dir: Path,
    output_dir: Path,
    index: dict[str, Any],
) -> None:
    medium_by_id = {node["medium_id"]: node for node in index["medium_nodes"]}
    sections: list[str] = []
    for question in questions:
        qid = question["question_id"]
        qdir = output_dir / qid
        planner = load_json(planner_dir / qid / "planner_parsed.json")
        routing = load_json(planner_dir / qid / "routing_result.json")
        ranking = load_json(qdir / "fine_ranking.json")
        selected = load_json(qdir / "selected_fines.json")
        selected_ids = {(row["medium_id"], row["fine_id"]) for row in selected}
        groups: list[str] = []
        for medium_id in [row["medium_id"] for row in load_json(planner_dir / qid / "selected_mediums.json")]:
            medium = medium_by_id[medium_id]
            medium_rows = [row for row in ranking if row["medium_id"] == medium_id]
            table_rows = "".join(
                "<tr><td>{unit}</td><td>{fine}</td><td>{time:.1f}</td><td>{raw:.5f}</td>"
                "<td>{norm:.4f}</td><td>{rank}</td><td>{selected}</td></tr>".format(
                    unit=html.escape(row["search_unit_id"]),
                    fine=html.escape(row["fine_id"]),
                    time=row["representative_frame_timestamp_sec"],
                    raw=row["siglip_score_raw"],
                    norm=row["siglip_score_normalized"],
                    rank=row["rank_within_medium"],
                    selected="✓" if (medium_id, row["fine_id"]) in selected_ids else "",
                )
                for row in medium_rows
            )
            cards = []
            for row in [item for item in selected if item["medium_id"] == medium_id]:
                uri = _image_data_uri(Path(row["representative_frame_path"]))
                cards.append(
                    f"<figure><img src='{uri}'><figcaption><b>{html.escape(row['fine_id'])}</b>"
                    f" @ {row['representative_frame_timestamp_sec']:.1f}s<br>"
                    f"{row['selection_reason']} · units: {html.escape(', '.join(row['search_unit_ids']))}</figcaption></figure>"
                )
            summary = load_json(qdir / "selection_summary.json")
            one = summary["single_selection_by_medium"].get(medium_id)
            groups.append(
                f"<article><h3>{html.escape(medium_id)} · {medium['start_sec']:.1f}–{medium['end_sec']:.1f}s"
                f" · {html.escape(medium['parent_coarse_id'])}</h3><p>{html.escape(medium['qwen_caption'])}</p>"
                f"<p class='note'>Single-selection reason: {html.escape(one or 'not applicable')}</p>"
                f"<div class='cards'>{''.join(cards)}</div><div class='table'><table><thead><tr>"
                "<th>Search unit</th><th>Fine</th><th>Frame time</th><th>Raw</th><th>Norm</th>"
                f"<th>Rank</th><th>Selected</th></tr></thead><tbody>{table_rows}</tbody></table></div></article>"
            )
        unit_columns = "".join(
            f"<li><b>{html.escape(unit['unit_id'])}</b>: {html.escape(unit['description'])}</li>"
            for unit in planner["search_units"]
        )
        sections.append(
            f"<section><h2>{html.escape(qid)}</h2><p class='question'>{html.escape(question['question'])}</p>"
            f"<p>{planner['scope']} · {planner['operation']} · {planner['retrieval_strategy']}</p>"
            f"<p>Coarse: {html.escape(', '.join(routing['selected_coarse_ids']))}</p><ul>{unit_columns}</ul>"
            f"{''.join(groups)}</section>"
        )
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>Fine reranking v1</title>
<style>body{{font:14px system-ui;margin:24px;color:#20242a}}section{{border-top:4px solid #334e68;padding:20px 0}}
article{{border:1px solid #ccd;border-radius:8px;padding:12px;margin:12px 0}}.question{{font-size:17px}}
.cards{{display:flex;gap:12px;flex-wrap:wrap}}figure{{margin:0;width:280px}}img{{width:280px;max-height:190px;object-fit:contain;background:#111}}
figcaption{{padding:5px;background:#eef3f8}}.table{{overflow:auto}}table{{width:100%;border-collapse:collapse}}
th,td{{border:1px solid #ccd;padding:5px}}th{{background:#eef3f8}}.note{{color:#52606d}}</style></head>
<body><h1>Fine reranking v1 — 226</h1><p>Offline Fine evidence diagnostic. No LLM call, dense sampling, caption generation, sufficiency decision, or answer.</p>
{''.join(sections)}</body></html>"""
    (output_dir / "review.html").write_text(document, encoding="utf-8")


def run_experiment(
    *,
    root: Path,
    index_path: Path,
    planner_dir: Path,
    config_path: Path,
    output_root: Path,
    encoder: SiglipTextEncoder | None = None,
) -> dict[str, Any]:
    config = load_json(config_path)
    index = load_json(index_path)
    validate_frozen_inputs(index=index, index_path=index_path, planner_dir=planner_dir)
    manifest = build_frozen_manifest(index_path, planner_dir)
    output_dir = output_root / "226"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "frozen_input_manifest.json", manifest)
    questions_payload = load_json(planner_dir / "questions_226.json")
    questions = questions_payload["questions"]
    fine_embeddings, row_by_id, embedding_path = load_fine_embeddings(index, index_path)
    fine_by_id = {node["fine_id"]: node for node in index["fine_nodes"]}
    medium_by_id = {node["medium_id"]: node for node in index["medium_nodes"]}
    encoder = encoder or SiglipTextEncoder(config["siglip"])
    all_candidates: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    per_question_selected: dict[str, set[str]] = {}
    for question in questions:
        started = time.perf_counter()
        qid = question["question_id"]
        source_qdir = planner_dir / qid
        qdir = output_dir / qid
        qdir.mkdir(parents=True, exist_ok=True)
        planner = load_json(source_qdir / "planner_parsed.json")
        routing = load_json(source_qdir / "routing_result.json")
        selected_mediums = load_json(source_qdir / "selected_mediums.json")
        query_records = [
            {
                "question_id": qid,
                "question": question["question"],
                "search_unit_id": unit["unit_id"],
                "search_unit_description": unit["description"],
                "query_variants": unit["query_variants"],
                "fine_query_text": build_fine_query_text(question["question"], unit),
            }
            for unit in planner["search_units"]
        ]
        encoding_started = time.perf_counter()
        query_embeddings = encoder.encode([row["fine_query_text"] for row in query_records])
        encoding_latency = time.perf_counter() - encoding_started
        query_embedding_by_unit = {
            row["search_unit_id"]: query_embeddings[index_]
            for index_, row in enumerate(query_records)
        }
        write_json(qdir / "fine_query.json", {"queries": query_records})
        all_ranking: list[dict[str, Any]] = []
        all_selected: list[dict[str, Any]] = []
        single_reasons: dict[str, str] = {}
        for selected_medium in selected_mediums:
            medium_id = selected_medium["medium_id"]
            medium = medium_by_id[medium_id]
            unit_ids = selected_medium.get("matched_search_unit_ids") or [
                unit["unit_id"] for unit in planner["search_units"]
            ]
            valid_units = [
                unit["unit_id"] for unit in planner["search_units"] if unit["unit_id"] in unit_ids
            ]
            if not valid_units:
                valid_units = [unit["unit_id"] for unit in planner["search_units"]]
            rankings_by_unit: dict[str, list[dict[str, Any]]] = {}
            for unit_id in valid_units:
                rows = rerank_fines_for_medium(
                    question_id=qid,
                    search_unit_id=unit_id,
                    medium=medium,
                    fine_by_id=fine_by_id,
                    fine_embeddings=fine_embeddings,
                    row_by_id=row_by_id,
                    query_embedding=query_embedding_by_unit[unit_id],
                )
                rankings_by_unit[unit_id] = rows
                all_ranking.extend(rows)
            chosen, single_reason = select_diverse_fines(
                rankings_by_unit,
                strategy=planner["retrieval_strategy"],
                max_fines=config["max_fine_per_medium"],
                minimum_gap=config["minimum_timestamp_gap_sec"],
            )
            if single_reason:
                single_reasons[medium_id] = single_reason
            for chosen_row in chosen:
                fine = fine_by_id[chosen_row["fine_id"]]
                all_selected.append(
                    {
                        "question_id": qid,
                        "search_unit_ids": chosen_row["search_unit_ids"],
                        "medium_id": medium_id,
                        "parent_coarse_id": medium["parent_coarse_id"],
                        "fine_id": fine["fine_id"],
                        "start_sec": fine["start_sec"],
                        "end_sec": fine["end_sec"],
                        "representative_frame_timestamp_sec": fine["representative_frame_timestamp_sec"],
                        "representative_frame_path": fine["representative_frame_path"],
                        "siglip_score_raw": chosen_row["siglip_score_raw"],
                        "siglip_score_by_search_unit": {
                            unit_id: next(
                                row["siglip_score_raw"]
                                for row in rankings_by_unit[unit_id]
                                if row["fine_id"] == fine["fine_id"]
                            )
                            for unit_id in rankings_by_unit
                        },
                        "selection_reason": chosen_row["selection_reason"],
                        "medium_caption": medium["qwen_caption"],
                        "source_storyline_ids": _storyline_ids_for_coarse(
                            index, medium["parent_coarse_id"]
                        ),
                    }
                )
        all_selected.sort(
            key=lambda row: (
                row["start_sec"],
                row["medium_id"],
                row["representative_frame_timestamp_sec"],
                row["fine_id"],
            )
        )
        all_ranking.sort(
            key=lambda row: (
                row["medium_id"],
                row["search_unit_id"],
                row["rank_within_medium"],
            )
        )
        elapsed = time.perf_counter() - started
        expanded_unique = {
            fine_id
            for row in selected_mediums
            for fine_id in medium_by_id[row["medium_id"]]["child_fine_ids"]
        }
        raw_scores = [row["siglip_score_raw"] for row in all_ranking]
        per_unit_counts = Counter(
            unit_id for row in all_selected for unit_id in row["search_unit_ids"]
        )
        timestamps = [row["representative_frame_timestamp_sec"] for row in all_selected]
        summary = {
            "question_id": qid,
            "selected_medium_count": len(selected_mediums),
            "expanded_unique_fine_count": len(expanded_unique),
            "fine_ranking_record_count": len(all_ranking),
            "selected_fine_count": len(all_selected),
            "average_selected_fines_per_medium": len(all_selected) / len(selected_mediums),
            "single_selection_medium_count": len(single_reasons),
            "single_selection_by_medium": single_reasons,
            "selected_fines_by_search_unit": dict(per_unit_counts),
            "duplicate_selected_frame_count": len(all_selected)
            - len({row["representative_frame_path"] for row in all_selected}),
            "score_distribution": {
                "minimum": min(raw_scores),
                "maximum": max(raw_scores),
                "mean": float(np.mean(raw_scores)),
                "std": float(np.std(raw_scores)),
            },
            "timestamp_coverage": {
                "minimum_selected_timestamp_sec": min(timestamps),
                "maximum_selected_timestamp_sec": max(timestamps),
            },
            "siglip_encoding_latency_sec": encoding_latency,
            "reranking_total_latency_sec": elapsed,
        }
        write_json(qdir / "fine_ranking.json", all_ranking)
        write_json(qdir / "selected_fines.json", all_selected)
        write_json(qdir / "selection_summary.json", summary)
        write_json(
            qdir / "provenance.json",
            {
                "frozen_input_manifest": "../frozen_input_manifest.json",
                "planner_source": str((source_qdir / "planner_parsed.json").resolve()),
                "selected_medium_source": str((source_qdir / "selected_mediums.json").resolve()),
                "fine_embedding_source": str(embedding_path.resolve()),
                "config": config,
                "no_api_calls": True,
                "no_medium_retrieval_recompute": True,
                "no_fine_resampling": True,
            },
        )
        all_candidates.append(
            {
                "question_id": qid,
                "question": question["question"],
                "planner_operation": planner["operation"],
                "retrieval_strategy": planner["retrieval_strategy"],
                "required_evidence": planner["required_evidence"],
                "selected_medium_ids": [row["medium_id"] for row in selected_mediums],
                "selected_fine_evidence": all_selected,
            }
        )
        summaries.append(summary)
        per_question_selected[qid] = {row["fine_id"] for row in all_selected}
    evidence_candidates = {
        "schema_version": "fine_evidence_candidates_v1",
        "video_id": index["video"]["video_id"],
        "questions": all_candidates,
    }
    write_json(output_dir / "evidence_candidates.json", evidence_candidates)
    weapon = next(item for item in all_candidates if item["question_id"] == "q_weapon_visible")
    hand = next(item for item in all_candidates if item["question_id"] == "q_handcuffing")
    medical = next(item for item in all_candidates if item["question_id"] == "q_medical_assistance")
    global_item = next(item for item in all_candidates if item["question_id"] == "q_global_summary")
    hand_ids, medical_ids = per_question_selected["q_handcuffing"], per_question_selected["q_medical_assistance"]
    overlap = hand_ids & medical_ids
    diagnostics = {
        "weapon_safe_node_0067": [
            row["fine_id"] for row in weapon["selected_fine_evidence"] if row["medium_id"] == "safe_node_0067"
        ],
        "handcuff_618_625_medium": [
            row["fine_id"]
            for row in hand["selected_fine_evidence"]
            if row["medium_id"] == "comet_style_dinov2_0051"
        ],
        "handcuff_medical_overlap": {
            "intersection_fine_ids": sorted(overlap),
            "intersection_count": len(overlap),
            "union_count": len(hand_ids | medical_ids),
            "jaccard": len(overlap) / max(1, len(hand_ids | medical_ids)),
        },
        "global_coarse_coverage": {
            "selected_coarse_ids": sorted(
                {row["parent_coarse_id"] for row in global_item["selected_fine_evidence"]}
            ),
            "expected_coarse_ids": [node["coarse_id"] for node in index["coarse_nodes"]],
        },
    }
    write_json(
        output_dir / "reranking_summary.json",
        {
            "questions": summaries,
            "diagnostics": diagnostics,
            "totals": {
                "selected_mediums": sum(row["selected_medium_count"] for row in summaries),
                "expanded_unique_fines_per_question_sum": sum(
                    row["expanded_unique_fine_count"] for row in summaries
                ),
                "selected_fines": sum(row["selected_fine_count"] for row in summaries),
                "siglip_encoding_latency_sec": sum(
                    row["siglip_encoding_latency_sec"] for row in summaries
                ),
                "reranking_latency_sec": sum(
                    row["reranking_total_latency_sec"] for row in summaries
                ),
                "api_calls": 0,
            },
        },
    )
    validation = validate_outputs(
        index=index,
        planner_dir=planner_dir,
        output_dir=output_dir,
        manifest=manifest,
        config=config,
    )
    write_json(output_dir / "validation_report.json", validation)
    if not validation["valid"]:
        raise FineRerankingError(f"Output validation failed: {validation['errors']}")
    write_json(
        output_dir / "provenance.json",
        {
            "experiment": "fine_reranking_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "frozen_input_manifest": "frozen_input_manifest.json",
            "config_path": str(config_path.resolve()),
            "api_calls": 0,
            "answer_model_run": False,
            "sufficiency_judge_run": False,
        },
    )
    render_review(
        questions=questions,
        planner_dir=planner_dir,
        output_dir=output_dir,
        index=index,
    )
    return {
        "valid": True,
        "output_dir": str(output_dir),
        "summaries": summaries,
        "diagnostics": diagnostics,
    }
