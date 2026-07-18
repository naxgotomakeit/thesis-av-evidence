from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from src.question_planner.task5a import (  # noqa: E402
    CODEX_ANALYSIS_INSTRUCTIONS, CORRECTION_TEMPLATE, JSON_OUTPUT_SCHEMA, PLAN_SCHEMA, PROMPT_VERSION,
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, ModelReply, aggregate_semantic_diagnostics,
    extract_cues, plan_with_retry, prompt_hash, safe_question_cases,
    semantic_diagnostic, shared_prompt_text,
)

OUTPUT = ROOT / "outputs/question_planner/v2"
MODEL_TEMPERATURE = 0
MODEL_MAX_TOKENS = 1200


def load_json(path: Path) -> Any: return json.loads(path.read_text(encoding="utf-8"))
def save_json(path: Path, obj: Any) -> None: path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sanitized_error(exc: Exception) -> str:
    text = re.sub(r"sk-ant-[A-Za-z0-9_-]+", "[REDACTED]", str(exc))
    text = re.sub(r"(?i)(x-api-key|authorization)\s*[:=]\s*\S+", r"\1=[REDACTED]", text)
    return f"{type(exc).__name__}: {text}"[:1000]


def esc(value: Any) -> str: return html.escape(str(value))
def pre(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    return f"<pre>{html.escape(text)}</pre>"


def make_html(path: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    preflight_blocks = []
    for audit_path in sorted(OUTPUT.glob("task5a_failed_*_run.jsonl")):
        audit_rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        rendered = []
        for row in audit_rows:
            rendered.append(f"<h4>{esc(row.get('case_id'))}</h4>{pre({'status': row.get('validation_status'), 'error': row.get('non_secret_error'), 'attempts': row.get('attempts', [])})}")
        preflight_blocks.append(f"<details><summary>{esc(audit_path.name)} — preserved unsuccessful implementation preflight</summary>{''.join(rendered)}</details>")
    shared = f"""
    <details open><summary><b>Codex Result Analysis</b></summary>
      <p><b>Schema validity:</b> {summary['valid_plan_count']}/{summary['case_count']} plans valid.</p>
      <p><b>Semantic-plan quality and likely functional impact:</b></p>{pre(summary['codex_result_analysis'])}
      <p>Semantic plausibility is separate from schema validity. Labeling-only issues do not imply functional failure when required resolvers and executable routes remain present.</p>
    </details>
    <details open><summary><b>Shared Prompt Configuration</b></summary>
      <p>Prompt version: <code>{PROMPT_VERSION}</code><br>SHA-256: <code>{prompt_hash()}</code></p>
      <h3>Exact Claude system prompt</h3>{pre(SYSTEM_PROMPT)}
      <h3>Exact user-prompt template</h3>{pre(USER_PROMPT_TEMPLATE)}
      <h3>Fixed JSON schema and field definitions</h3>{pre(PLAN_SCHEMA)}
      <h3>API structured-output JSON Schema</h3>{pre(JSON_OUTPUT_SCHEMA)}
      <h3>Retry/correction template</h3>{pre(CORRECTION_TEMPLATE)}
      <h3>Model configuration</h3>{pre(summary['model_configuration'])}
    </details>
    <details><summary><b>Codex Analysis Instructions</b></summary>{pre(CODEX_ANALYSIS_INSTRUCTIONS)}</details>
    <details><summary><b>Preserved implementation preflight history</b></summary><p>These unsuccessful preflight attempts are not validated plans and are excluded from the validated-run aggregate. They are retained to avoid hiding raw model responses, validation errors, or rejected API-schema attempts.</p>{''.join(preflight_blocks)}</details>
    """
    cases = []
    for record in records:
        plan = record.get("plan") or {}
        answer_req = plan.get("answer_requirement", {})
        attempts = []
        for attempt in record.get("attempts", []):
            attempts.append(f"""
              <details><summary>Request {attempt['request_number']} audit — validation: {esc('valid' if not attempt['validation_errors'] else 'invalid')}</summary>
                <p>Model: <code>{esc(record['model_identifier'])}</code>; temperature: {MODEL_TEMPERATURE}; max tokens: {MODEL_MAX_TOKENS};
                input/output tokens: {attempt['input_tokens']}/{attempt['output_tokens']}; latency: {attempt['latency_sec']:.6f}s.</p>
                <h4>Complete rendered prompt actually sent</h4>{pre(attempt['rendered_prompt'])}
                <h4>Raw Claude text response (unaltered)</h4>{pre(attempt['raw_response'])}
                <h4>Parsed JSON for this attempt</h4>{pre(attempt['parsed_plan'])}
                <h4>Validation errors</h4>{pre(attempt['validation_errors'])}
              </details>""")
        retry_attempt = record.get("attempts", [])[1] if len(record.get("attempts", [])) > 1 else None
        cases.append(f"""
        <article class="case">
          <h2>{esc(record['case_id'])}</h2><p class="badge">MODEL-GENERATED PLAN — AWAITING HUMAN REVIEW</p>
          <h3>Raw question</h3><p>{esc(record['raw_question'])}</p>
          <div class="grid">
            <section><h3>Deterministic parser output</h3>{pre(record['deterministic_cues'])}</section>
            <section><h3>Validated Claude plan</h3>
              <dl><dt>Answer requirement</dt><dd>{esc(answer_req.get('operation'))}: {esc(answer_req.get('description'))}</dd>
              <dt>Anchor cues</dt><dd>{esc(plan.get('anchor_cues'))}</dd><dt>Primary anchor modality</dt><dd>{esc(plan.get('primary_anchor_modality'))}</dd>
              <dt>Resolver modalities</dt><dd>{esc(plan.get('resolver_modalities'))}</dd><dt>Audio role</dt><dd>{esc(plan.get('audio_role'))}</dd>
              <dt>Temporal relation</dt><dd>{esc(plan.get('temporal_relation'))}</dd><dt>Visual route</dt><dd>{esc(plan.get('visual_route'))}</dd>
              <dt>Fallback route</dt><dd>{esc(plan.get('fallback_route'))}</dd><dt>Planner confidence</dt><dd>{esc(plan.get('planner_confidence'))}</dd>
              <dt>Rationale</dt><dd>{esc(plan.get('rationale'))}</dd></dl>{pre(plan)}
            </section>
          </div>
          <p>API totals — latency: {record['api_usage']['latency_sec']:.6f}s; input/output tokens: {record['api_usage']['input_tokens']}/{record['api_usage']['output_tokens']}; retry: {record['retry_required']}; validation: {esc(record['validation_status'])}.</p>
          <details><summary><b>Prompt Audit — deterministic cues, rendered prompts, raw responses, parsed plans</b></summary>
            <h4>Deterministic cues inserted</h4>{pre(record['deterministic_cues'])}
            {''.join(attempts)}
            <h4>Retry/correction prompt and response</h4>{pre({'prompt': retry_attempt['rendered_prompt'], 'raw_response': retry_attempt['raw_response']} if retry_attempt else 'No retry occurred.')}
          </details>
          <details open><summary><b>Separate Codex semantic diagnostic</b></summary>{pre(record['codex_semantic_diagnostic'])}</details>
          <details><summary><b>Human review notes</b></summary><textarea placeholder="Human reviewer notes (not saved automatically)"></textarea></details>
        </article>""")
    css = """body{font:15px system-ui;margin:2rem;max-width:1500px;color:#183153}details{border:1px solid #bcccdc;padding:.8rem;margin:1rem 0;border-radius:7px}summary{cursor:pointer}.case{border-top:4px solid #627d98;margin-top:2.5rem}.badge{display:inline-block;background:#fff3bf;border-left:5px solid #d69e2e;padding:.5rem}.grid{display:grid;grid-template-columns:1fr 1.4fr;gap:1rem}pre{white-space:pre-wrap;word-break:break-word;background:#f2f5f7;padding:.8rem;max-height:500px;overflow:auto}dt{font-weight:bold;margin-top:.5rem}dd{margin-left:0}textarea{width:100%;height:90px}@media(max-width:900px){.grid{grid-template-columns:1fr}}"""
    path.write_text(f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>Task 5A human review</title><style>{css}</style></head><body><h1>Task 5A — Question Understanding and Modality Routing</h1><p class='badge'>All plans are model-generated and awaiting human review. No retrieval, media inspection, final QA, or Task 5B was performed.</p>{shared}{''.join(cases)}</body></html>", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run question-only Task 5A planner")
    parser.add_argument("--cases", default="data/manifests/mvp_cases_6.json")
    args = parser.parse_args()
    key, model = os.environ.get("ANTHROPIC_API_KEY"), os.environ.get("ANTHROPIC_MODEL")
    if not key or not model: raise SystemExit("Required Anthropic environment variables are unavailable.")
    import anthropic
    client = anthropic.Anthropic(api_key=key)
    source_rows = load_json(ROOT / args.cases)
    cases = safe_question_cases(source_rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    def request(system: str, user: str, number: int) -> ModelReply:
        started = time.perf_counter()
        response = client.messages.create(model=model, max_tokens=MODEL_MAX_TOKENS, temperature=MODEL_TEMPERATURE, system=system, messages=[{"role": "user", "content": user}], output_config={"format": {"type": "json_schema", "schema": JSON_OUTPUT_SCHEMA}})
        latency = time.perf_counter() - started
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        return ModelReply(text=text, latency_sec=latency, input_tokens=int(response.usage.input_tokens), output_tokens=int(response.usage.output_tokens))

    for case in cases:
        cues = extract_cues(case["question"])
        try:
            outcome = plan_with_retry(case["question"], cues, request)
            attempts = outcome["attempts"]
            usage = {"latency_sec": sum(x["latency_sec"] for x in attempts), "input_tokens": sum(x["input_tokens"] for x in attempts), "output_tokens": sum(x["output_tokens"] for x in attempts)}
            record = {"case_id": case["case_id"], "raw_question": case["question"], "task5a_prompt_version": PROMPT_VERSION, "shared_prompt_sha256": prompt_hash(), "model_identifier": model, "temperature": MODEL_TEMPERATURE, "max_tokens": MODEL_MAX_TOKENS, "deterministic_cues": cues, **outcome, "api_usage": usage, "non_secret_error": None}
        except Exception as exc:
            record = {"case_id": case["case_id"], "raw_question": case["question"], "task5a_prompt_version": PROMPT_VERSION, "shared_prompt_sha256": prompt_hash(), "model_identifier": model, "temperature": MODEL_TEMPERATURE, "max_tokens": MODEL_MAX_TOKENS, "deterministic_cues": cues, "plan": None, "validation_status": "api_error", "retry_required": False, "attempts": [], "api_usage": {"latency_sec": 0.0, "input_tokens": 0, "output_tokens": 0}, "non_secret_error": sanitized_error(exc)}
        record["codex_semantic_diagnostic"] = semantic_diagnostic(case["question"], cues, record.get("plan") or {}) if record.get("plan") else {"analysis_status":"inconsistent","question_understanding":"No valid plan was available.","anchor_resolver_analysis":"Unavailable.","routing_consistency":"Unavailable.","functional_impact":"may_omit_required_evidence","possible_problem":"Planner output was unavailable or invalid.","recommended_human_check":"Inspect the API/validation failure before any routing use."}
        records.append(record)

    with (OUTPUT / "task5a_plans.jsonl").open("w", encoding="utf-8") as handle:
        for record in records: handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    total_latency = sum(x["api_usage"]["latency_sec"] for x in records); total_in = sum(x["api_usage"]["input_tokens"] for x in records); total_out = sum(x["api_usage"]["output_tokens"] for x in records)
    valid = sum(x["validation_status"] == "valid" for x in records); retries = sum(x["retry_required"] for x in records)
    operation_counts: dict[str, int] = {}; anchor_counts: dict[str, int] = {}; resolver_counts: dict[str, int] = {}
    for row in records:
        plan = row.get("plan") or {}; op = plan.get("answer_requirement", {}).get("operation"); anchor = plan.get("primary_anchor_modality")
        if op: operation_counts[op] = operation_counts.get(op, 0) + 1
        if anchor: anchor_counts[anchor] = anchor_counts.get(anchor, 0) + 1
        for resolver in plan.get("resolver_modalities", []): resolver_counts[resolver] = resolver_counts.get(resolver, 0) + 1
    semantic_summary = aggregate_semantic_diagnostics(records)
    summary = {"task": "Task 5A v2 Question Understanding and Modality Routing", "scope": "question-only planning; no retrieval, media inspection, answers, reference timestamps, or Task 5B", "case_count": len(records), "valid_plan_count": valid, "validation_failure_count": len(records)-valid, "retry_count": retries, "task5a_prompt_version": PROMPT_VERSION, "shared_prompt_sha256": prompt_hash(), "model_configuration": {"model_identifier": model, "temperature": MODEL_TEMPERATURE, "max_tokens": MODEL_MAX_TOKENS, "prompt_caching": False, "requests_per_question": "one, plus one correction retry only on parse/schema failure"}, "aggregate_api_usage": {"total_latency_sec": total_latency, "mean_latency_sec": total_latency/max(len(records),1), "input_tokens": total_in, "output_tokens": total_out}, "planner_summary": {"answer_operation_counts": operation_counts, "primary_anchor_modality_counts": anchor_counts, "resolver_modality_counts": resolver_counts}, "codex_result_analysis": semantic_summary, "failures": [{"case_id":x["case_id"],"error":x["non_secret_error"],"status":x["validation_status"]} for x in records if x["validation_status"] != "valid"], "limitations": ["Plans are model-generated and require human review.", "Deterministic cues come only from raw question text.", "Temporal cues in questions are not precise gold boundaries.", "Task 5A does not establish semantic correctness or execute retrieval."]}
    save_json(OUTPUT / "task5a_summary.json", summary)
    md = f"""# Task 5A summary

## Scope

Question-only evidence planning for {len(records)} frozen MVP questions. No answers, annotation context, reference timestamps, retrieval results, media, retrieval execution, evidence selection, or Task 5B were used.

## Schema

The strict plan schema records answer requirement, anchor cues/modality, resolver modalities, audio role, temporal relation, local-visual requirement, visual and fallback routes, confidence, and a two-sentence rationale. See `task5a_summary.json` and the HTML Prompt Audit for the exact schema.

## Model configuration

- Model: `{model}`
- Temperature: `{MODEL_TEMPERATURE}`
- Max tokens: `{MODEL_MAX_TOKENS}`
- Prompt version: `{PROMPT_VERSION}`
- Shared prompt SHA-256: `{prompt_hash()}`

## Aggregate usage and validation

- Valid plans: {valid}/{len(records)}
- Retries: {retries}
- Total latency: {total_latency:.6f}s (mean {total_latency/max(len(records),1):.6f}s)
- Input/output tokens: {total_in}/{total_out}
- Validation failures: {len(records)-valid}
- Operation counts: `{operation_counts}`
- Primary anchor counts: `{anchor_counts}`
- Resolver counts: `{resolver_counts}`

## Codex result analysis

`{json.dumps(semantic_summary, ensure_ascii=False)}`

## Limitations

- Plans are model-generated and awaiting human review.
- Question timestamps are textual cues, not precise gold boundaries.
- Routing is planned but not executed; semantic correctness is not established.
"""
    (OUTPUT / "task5a_summary.md").write_text(md, encoding="utf-8")
    make_html(OUTPUT / "task5a_human_review.html", records, summary)
    print(json.dumps({"case_count":len(records),"valid":valid,"retries":retries,"total_latency_sec":total_latency,"input_tokens":total_in,"output_tokens":total_out,"failures":summary["failures"]}, ensure_ascii=False))
    return 0 if valid == len(records) else 1


if __name__ == "__main__": raise SystemExit(main())
