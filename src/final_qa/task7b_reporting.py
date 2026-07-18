"""Shared post-hoc reporting helpers for Task 7B v0.2/v0.3."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def guarded_load_gold(
    gold_manifest: Path,
    raw_predictions_path: Path,
    validated_predictions_path: Path,
    case_ids: list[str] | tuple[str, ...],
) -> dict[str, str | None]:
    """Load gold only after persisted prediction and validation artifacts exist."""
    if not raw_predictions_path.is_file() or not validated_predictions_path.is_file():
        raise RuntimeError("posthoc_gold_load_requires_saved_raw_and_validated_outputs")
    raw_ids = {row["case_id"] for row in read_jsonl(raw_predictions_path)}
    validated_ids = {row["case_id"] for row in read_jsonl(validated_predictions_path)}
    missing = set(case_ids) - raw_ids.intersection(validated_ids)
    if missing:
        raise RuntimeError(f"posthoc_gold_load_blocked_missing_predictions:{sorted(missing)}")
    records = json.loads(gold_manifest.read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = records.get("cases", records.get("records", []))
    by_case = {record["case_id"]: record.get("answer") for record in records}
    return {case_id: by_case.get(case_id) for case_id in case_ids}


def modality_token_counts(usage: dict[str, Any]) -> dict[str, int | None]:
    values = {"text": 0, "image": 0, "audio": 0, "other": 0}
    seen = False
    for item in usage.get("input_tokens_by_modality") or []:
        modality = str(item.get("modality", item.get("type", "other"))).casefold()
        count = item.get("tokens", item.get("token_count"))
        if not isinstance(count, int):
            continue
        seen = True
        key = next((name for name in ("text", "image", "audio") if name in modality), "other")
        values[key] += count
    return values if seen else {key: None for key in values}


def _short(value: Any, limit: int = 70) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _cell(value: Any) -> str:
    if isinstance(value, bool):
        value = str(value).lower()
    if value is None:
        value = "unavailable"
    return html.escape(str(value))


def _table(headers: list[str], rows: list[list[Any]], css_class: str = "") -> str:
    head = "".join(f"<th>{html.escape(item)}</th>" for item in headers)
    body = "".join("<tr>" + "".join(f"<td>{_cell(value)}</td>" for value in row) + "</tr>" for row in rows)
    return f'<table class="{css_class}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def make_review_html(
    title: str,
    reconstruction_note: str | None,
    case_ids: list[str],
    payloads: dict[str, dict[str, Any]],
    raw_rows: dict[str, dict[str, Any]],
    validated: dict[str, dict[str, Any]],
    usages: dict[str, dict[str, Any]],
    reports: dict[str, dict[str, Any]],
    manifests: dict[str, dict[str, Any]],
    posthoc: dict[str, dict[str, Any]],
    results: dict[str, dict[str, Any]],
) -> str:
    answer_rows, efficiency_rows, modality_rows, status_rows = [], [], [], []
    for case_id in case_ids:
        gold = posthoc.get(case_id, {})
        valid = validated.get(case_id, {})
        usage = usages.get(case_id, {})
        modality = modality_token_counts(usage)
        answer_rows.append([case_id, valid.get("answer_status"), _short(gold.get("gold_dataset_answer")), _short(valid.get("answer"))])
        efficiency_rows.append([
            case_id, valid.get("answer_status"), usage.get("visual_frame_count", 0), usage.get("acoustic_duration_sec", 0),
            usage.get("speech_segment_count", 0), usage.get("input_tokens"), usage.get("output_tokens"),
            usage.get("unattributed_tokens"), usage.get("total_tokens"), usage.get("total_api_calls", 0),
            usage.get("retry_api_calls", 0), round(float(usage.get("total_latency_sec", 0)), 3),
        ])
        modality_rows.append([case_id, modality["text"], modality["image"], modality["audio"], modality["other"]])
        status_rows.append([case_id, valid.get("answer_status"), len(valid.get("immutable_pipeline_uncertainties", [])), len(valid.get("final_uncertainties", [])), valid.get("dataset_audit", {}).get("status"), valid.get("confidence", {}).get("level"), valid.get("abstain")])

    sections = []
    for case_id in case_ids:
        payload, valid, usage = payloads[case_id], validated.get(case_id, {}), usages.get(case_id, {})
        gold, raw, report = posthoc.get(case_id, {}), raw_rows.get(case_id, {}), reports.get(case_id, {})
        answer_panel = _table(["Field", "Value"], [
            ["Question", payload.get("question")], ["Gold / dataset answer", gold.get("gold_dataset_answer")],
            ["Raw model prediction", gold.get("raw_model_prediction")], ["Validated prediction", valid.get("answer")],
            ["Answer status", valid.get("answer_status")], ["Abstain", valid.get("abstain")],
            ["Confidence", valid.get("confidence", {}).get("level")], ["Human review status", gold.get("human_review_status")],
        ])
        per_usage = _table(["Metric", "Value"], [
            ["Model", usage.get("model")], ["Thinking level", usage.get("thinking_level")],
            ["Initial API calls", usage.get("initial_api_calls", 0)], ["Retry API calls", usage.get("retry_api_calls", 0)],
            ["Total API calls", usage.get("total_api_calls", 0)], ["Latency", f"{float(usage.get('total_latency_sec', 0)):.3f} s"],
            ["Input tokens", usage.get("input_tokens")], ["Output tokens", usage.get("output_tokens")],
            ["Unattributed tokens", usage.get("unattributed_tokens")], ["Total tokens", usage.get("total_tokens")],
            ["Image count", usage.get("visual_frame_count", 0)], ["Image bytes", usage.get("visual_asset_bytes", 0)],
            ["Audio clip count", usage.get("acoustic_clip_count", 0)], ["Audio duration", usage.get("acoustic_duration_sec", 0)],
            ["Audio bytes", usage.get("acoustic_asset_bytes", 0)], ["Speech segment count", usage.get("speech_segment_count", 0)],
            ["Transcript characters", usage.get("transcript_character_count", 0)], ["Evidence group count", usage.get("evidence_group_count", 0)],
            ["Relation count", sum(len(g.get("relations", [])) for g in payload.get("evidence_groups", []))],
            ["Inherited uncertainty count", len(valid.get("immutable_pipeline_uncertainties", []))], ["Final uncertainty count", len(valid.get("final_uncertainties", []))],
            ["Validator correction count", len(report.get("corrections", []))], ["Schema valid first call", usage.get("schema_valid_on_first_call")],
            ["Estimated cost", usage.get("estimated_cost")],
        ])
        pre = lambda value: f"<pre>{html.escape(json.dumps(value, ensure_ascii=False, indent=2))}</pre>"
        sections.append(f"""<article><h2>{_cell(case_id)}</h2><h3>Gold Answer vs Prediction</h3>{answer_panel}
<h3>提供给模型的完整 evidence</h3>{pre(payload.get('evidence_groups'))}
<h3>Uncertainty assessment 与 final_uncertainties</h3>{pre({'immutable_pipeline_uncertainties': valid.get('immutable_pipeline_uncertainties'), 'uncertainty_assessments': valid.get('uncertainty_assessments'), 'newly_observed_uncertainties': valid.get('newly_observed_uncertainties'), 'final_uncertainties': valid.get('final_uncertainties'), 'dataset_audit': valid.get('dataset_audit')})}
<h3>Evidence citations 与 timestamp_source</h3>{pre(valid.get('evidence_used'))}
<h3>Raw model output</h3>{pre(raw)}<h3>Validator corrections</h3>{pre(report)}
<h3>API 使用与效率</h3>{per_usage}<h3>Request manifest 与 leakage audit</h3>{pre(manifests.get(case_id))}
<h3>Gold quality / 人工复核占位</h3>{pre(gold)}<h3>Errors</h3>{pre(results.get(case_id, {}).get('error'))}</article>""")

    note = f"<p class=notice>{html.escape(reconstruction_note)}</p>" if reconstruction_note else ""
    return f"""<!doctype html><html lang=zh><meta charset=utf-8><title>{html.escape(title)}</title>
<style>body{{font:15px system-ui;margin:2rem;max-width:1500px;color:#183153}}table{{border-collapse:collapse;width:100%;margin:.7rem 0 1.5rem}}th,td{{border:1px solid #ccd6dd;padding:.5rem;text-align:left;vertical-align:top}}th{{background:#eaf0f6}}article{{border-top:4px solid #627d98;margin-top:3rem}}pre{{white-space:pre-wrap;word-break:break-word;background:#f4f7f9;padding:1rem;max-height:520px;overflow:auto}}.notice{{background:#fff3cd;padding:1rem;border-left:5px solid #d39e00}}</style>
<h1>{html.escape(title)}</h1>{note}<h2>Gold Answer vs Prediction</h2>{_table(['Case','Status','Gold','Prediction'], answer_rows)}
<h2>效率汇总</h2>{_table(['Case','Status','Frames','Audio sec','Speech segs','Input tok','Output tok','Unattributed tok','Total tok','Calls','Retries','Latency'], efficiency_rows)}
<h2>Modality token breakdown</h2>{_table(['Case','Text tokens','Image tokens','Audio tokens','Other / unavailable'], modality_rows)}
<h2>Answer status 与 uncertainty 汇总</h2>{_table(['Case','Status','Inherited','Final','Dataset audit','Confidence','Abstain'], status_rows)}
{''.join(sections)}</html>"""

