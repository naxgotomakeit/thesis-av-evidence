"""Single-turn GenS selected-frame answering with crash-safe route boundaries.

No selector, retrieval, navigation tool, map, or gold answer is used here.
Real transport is disabled unless the caller explicitly constructs the real provider.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from direct_api_v1.pricing import AnthropicCacheAwarePricing

PROHIBITED_INPUT_FIELDS = {
    "answer", "correct_answer", "correct_option", "gold", "gold_answer",
    "winning_option", "selection_reasoning", "selection_provenance",
    "planner", "shared", "fine", "final", "previous_prediction",
}
OPTION_IDS = ("A", "B", "C", "D", "E")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(raw)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def answer_prompt(row: dict[str, Any], template: str | None = None) -> str:
    validate_input_row(row)
    options = row["options"]
    if template is None:
        template = "{question}\nA. {option_A}\nB. {option_B}\nC. {option_C}\nD. {option_D}\nE. {option_E}\nAnswer with the option's letter from the given choices directly.\n"
    return template.format(question=row["question"], **{f"option_{key}": options[key] for key in OPTION_IDS}).rstrip("\n")


def parse_answer(raw_text: str) -> str | None:
    stripped = raw_text.strip()
    return stripped if stripped in OPTION_IDS else None


def validate_input_row(row: dict[str, Any]) -> None:
    forbidden = sorted(PROHIBITED_INPUT_FIELDS.intersection(row))
    if forbidden:
        raise ValueError(f"prohibited GenS answering fields: {forbidden}")
    required = {"qa_uid", "video_id", "question", "options", "selected_frame_count", "selected_frames"}
    if not required.issubset(row):
        raise ValueError(f"missing required fields: {sorted(required - set(row))}")
    if list(row["options"].keys()) != list(OPTION_IDS):
        raise ValueError("options must preserve exact A-E order")
    frames = row["selected_frames"]
    if not 1 <= len(frames) <= 16 or row["selected_frame_count"] != len(frames):
        raise ValueError("invalid frozen selected-frame count")
    if [f["chronological_order"] for f in frames] != list(range(len(frames))):
        raise ValueError("frames are not in chronological_order")
    if [float(f["timestamp_sec"]) for f in frames] != sorted(float(f["timestamp_sec"]) for f in frames):
        raise ValueError("frame timestamps are not chronological")


@dataclass(frozen=True)
class GensAnsweringConfig:
    schema_version: str
    experiment_id: str
    provider: str
    model: str
    max_output_tokens: int
    temperature: float
    timeout_sec: float
    max_retries: int
    concurrency: int
    hard_api_budget_usd: float
    per_request_budget_reserve_usd: float
    api_enabled_by_default: bool
    system_prompt: str | None
    message_layout: str
    image_transport: str
    cache_policy: str
    prompt_template_path: str
    pricing: dict[str, Any]
    direct_reference_config: str
    selector_path: str
    uid_order_path: str
    frame_root: str
    canonical_population_manifest: str

    @classmethod
    def load(cls, path: Path) -> "GensAnsweringConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        config = cls(**data)
        direct = json.loads(Path(config.direct_reference_config).read_text(encoding="utf-8"))
        for field in ("model", "max_output_tokens", "timeout_sec", "max_retries"):
            if getattr(config, field) != direct[field]:
                raise ValueError(f"GenS/Direct mismatch for {field}")
        if config.provider != direct["provider"] or config.temperature != 0.0:
            raise ValueError("provider/temperature mismatch")
        if config.api_enabled_by_default or config.concurrency != 1 or config.per_request_budget_reserve_usd <= 0:
            raise ValueError("unsafe GenS answering execution defaults")
        return config

    @property
    def pricing_object(self) -> AnthropicCacheAwarePricing:
        return AnthropicCacheAwarePricing.from_mapping(self.pricing)


def resolve_frames(row: dict[str, Any], frame_root: Path, verify_sha: bool = True) -> list[dict[str, Any]]:
    validate_input_row(row)
    resolved = []
    for frame in row["selected_frames"]:
        relative = Path(frame["relative_frame_path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != row["video_id"]:
            raise ValueError("unsafe or wrong-video frame path")
        if relative != Path(row["video_id"]) / frame["frame_filename"]:
            raise ValueError("relative frame path/filename mismatch")
        path = frame_root / row["video_id"] / "frames_1fps" / frame["frame_filename"]
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path) if verify_sha else None
        if verify_sha and observed != frame["frame_sha256"]:
            raise ValueError(f"frame SHA mismatch: {path}")
        resolved.append({
            "chronological_order": frame["chronological_order"],
            "timestamp_sec": float(frame["timestamp_sec"]), "path": str(path),
            "expected_sha256": frame["frame_sha256"], "observed_sha256": observed,
        })
    return resolved


def build_request(row: dict[str, Any], frame_root: Path, config: GensAnsweringConfig,
                  *, encode_images: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frames = resolve_frames(row, frame_root, verify_sha=True)
    content: list[dict[str, Any]] = []
    # Frozen layout: original full-resolution JPEG blocks in chronological order, then exact MCQ text.
    for frame in frames:
        data = base64.b64encode(Path(frame["path"]).read_bytes()).decode("ascii") if encode_images else "<validated-jpeg-bytes>"
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}})
    template = Path(config.prompt_template_path).read_text(encoding="utf-8")
    content.append({"type": "text", "text": answer_prompt(row, template)})
    payload: dict[str, Any] = {
        "model": config.model, "max_tokens": config.max_output_tokens,
        "temperature": config.temperature, "messages": [{"role": "user", "content": content}],
    }
    # Empty system means omitted, not an extra hidden instruction.
    if config.system_prompt:
        payload["system"] = config.system_prompt
    return payload, frames


@dataclass
class ProviderOutcome:
    status: str
    raw_text: str | None
    input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0
    latency_sec: float = 0.0
    response_id: str | None = None
    response_model: str | None = None
    stop_reason: str | None = None
    stop_sequence: str | None = None
    error_category: str | None = None


class GensProviderError(RuntimeError):
    pass


class _HttpMessages:
    def __init__(self, api_key: str, timeout: float) -> None:
        self.api_key, self.timeout = api_key, timeout

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=json.dumps(payload).encode("utf-8"), method="POST",
            headers={"content-type": "application/json", "x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))


class GensHaikuProvider:
    """Real single-turn transport. Construction requires an explicit API key."""
    def __init__(self, config: GensAnsweringConfig, *, api_key: str, client: Any | None = None) -> None:
        if not api_key:
            raise GensProviderError("explicit API key required")
        self.config = config
        self.client = client or _HttpMessages(api_key, config.timeout_sec)

    def call(self, payload: dict[str, Any]) -> ProviderOutcome:
        started = time.perf_counter()
        try:
            response = self.client.create(payload)
        except (TimeoutError, urllib.error.URLError) as error:
            raise GensProviderError(type(error).__name__) from error
        content = response.get("content", [])
        if any(block.get("type") != "text" for block in content):
            raw_text = None
            error_category = "non_text_response"
        else:
            raw_text = "".join(str(block.get("text", "")) for block in content)
            error_category = None
        usage = response.get("usage", {})
        return ProviderOutcome(
            status="response_received", raw_text=raw_text,
            input_tokens=int(usage.get("input_tokens", 0)),
            cache_creation_input_tokens=int(usage.get("cache_creation_input_tokens", 0)),
            cache_read_input_tokens=int(usage.get("cache_read_input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            latency_sec=time.perf_counter() - started, response_id=response.get("id"),
            response_model=response.get("model"), stop_reason=response.get("stop_reason"),
            stop_sequence=response.get("stop_sequence"), error_category=error_category,
        )


class GensAnsweringStore:
    def __init__(self, root: Path, experiment_id: str, cap_usd: float) -> None:
        self.root, self.experiment_id, self.cap_usd = root, experiment_id, cap_usd

    @property
    def ledger_path(self) -> Path:
        return self.root / "budget_ledger.json"

    def initialise(self, fingerprint: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        fp_path = self.root / "run_fingerprint.json"
        if fp_path.exists() and json.loads(fp_path.read_text())["fingerprint"] != fingerprint:
            raise RuntimeError("run fingerprint mismatch; refuse mixed run")
        if not fp_path.exists():
            atomic_json(fp_path, {"fingerprint": fingerprint, "created_at": utc_now()})
        if not self.ledger_path.exists():
            atomic_json(self.ledger_path, {"cap_usd": self.cap_usd, "spent_usd": 0.0, "charges": {}, "created_at": utc_now()})

    def status_path(self, qid: str) -> Path:
        return self.root / "route_status" / f"{qid}.json"

    def status(self, qid: str) -> dict[str, Any] | None:
        path = self.status_path(qid)
        return json.loads(path.read_text()) if path.exists() else None

    def start(self, qid: str, input_fingerprint: str) -> None:
        prior = self.status(qid)
        if prior and prior["state"].startswith("terminal_"):
            raise RuntimeError("terminal route cannot restart")
        atomic_json(self.status_path(qid), {"state": "running", "question_id": qid, "input_fingerprint": input_fingerprint, "started_at": utc_now()})

    def append(self, journal: str, row: dict[str, Any]) -> None:
        append_jsonl(self.root / "journals" / f"{journal}.jsonl", {"timestamp": utc_now(), "experiment_id": self.experiment_id, **row})

    def charge_once(self, attempt_id: str, usd: float) -> bool:
        ledger = json.loads(self.ledger_path.read_text())
        if attempt_id in ledger["charges"]:
            return False
        # A completed billed attempt is authoritative and must never disappear,
        # even if its observed charge exceeded the conservative pre-call reserve.
        ledger["charges"][attempt_id] = usd
        ledger["spent_usd"] = float(ledger["spent_usd"]) + usd
        atomic_json(self.ledger_path, ledger)
        return True

    def reconcile_ledger(self) -> int:
        count = 0
        for row in read_jsonl(self.root / "journals" / "attempt_end.jsonl"):
            if self.charge_once(str(row["attempt_id"]), float(row.get("cache_aware_usd", 0.0))):
                count += 1
        return count

    def assert_budget(self, upper_bound_usd: float) -> None:
        ledger = json.loads(self.ledger_path.read_text())
        if float(ledger["spent_usd"]) + upper_bound_usd > float(ledger["cap_usd"]) + 1e-12:
            raise RuntimeError("hard API budget would be exceeded")

    def terminal(self, qid: str, success: bool, artifact: dict[str, Any]) -> None:
        atomic_json(self.root / "routes" / f"{qid}.json", artifact)
        atomic_json(self.status_path(qid), {
            "state": "terminal_success" if success else "terminal_failed", "question_id": qid,
            "prediction": artifact.get("prediction"), "failure_category": artifact.get("failure_category"),
            "terminal_at": utc_now(), "resource_totals": artifact["resource_totals"],
        })

    def reconcile_interrupted(self, qids: Iterable[str]) -> int:
        count = 0
        starts = read_jsonl(self.root / "journals" / "request_start.jsonl")
        ends = read_jsonl(self.root / "journals" / "attempt_end.jsonl")
        ended = {r["attempt_id"] for r in ends}
        for qid in qids:
            status = self.status(qid)
            if status and status.get("state") == "running":
                qstarts = [r for r in starts if r.get("question_id") == qid]
                unmatched = [r["attempt_id"] for r in qstarts if r["attempt_id"] not in ended]
                qends = [r for r in ends if r.get("question_id") == qid]
                totals = aggregate_attempts(qends)
                artifact = {"question_id": qid, "prediction": None, "terminal_status": "interrupted_active_route",
                            "failure_category": "interrupted_active_route", "provider_outcome_unknown": bool(unmatched),
                            "unmatched_attempt_ids": unmatched, "resource_totals": totals}
                self.terminal(qid, False, artifact)
                count += 1
        return count


def attempt_cost(outcome: ProviderOutcome, pricing: AnthropicCacheAwarePricing) -> dict[str, float]:
    return pricing.breakdown(
        ordinary_input_tokens=outcome.input_tokens,
        cache_creation_input_tokens=outcome.cache_creation_input_tokens,
        cache_read_input_tokens=outcome.cache_read_input_tokens,
        output_tokens=outcome.output_tokens,
    )


def aggregate_attempts(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    return {
        "provider_attempts": len(rows),
        "input_tokens": sum(int(r.get("input_tokens", 0)) for r in rows),
        "cache_creation_input_tokens": sum(int(r.get("cache_creation_input_tokens", 0)) for r in rows),
        "cache_read_input_tokens": sum(int(r.get("cache_read_input_tokens", 0)) for r in rows),
        "output_tokens": sum(int(r.get("output_tokens", 0)) for r in rows),
        "cache_aware_usd": sum(float(r.get("cache_aware_usd", 0.0)) for r in rows),
        "api_latency_sec": sum(float(r.get("latency_sec", 0.0)) for r in rows),
    }


class GensRouteRunner:
    def __init__(self, config: GensAnsweringConfig, store: GensAnsweringStore,
                 provider_factory: Callable[[], Any]) -> None:
        self.config, self.store, self.provider_factory = config, store, provider_factory

    def run_one(self, row: dict[str, Any], *, run_fingerprint: str) -> dict[str, Any]:
        qid = str(row["qa_uid"])
        status = self.store.status(qid)
        if status and status["state"].startswith("terminal_"):
            return {"question_id": qid, "skipped_terminal": True}
        input_fp = canonical_sha(row)
        self.store.start(qid, input_fp)
        attempts: list[dict[str, Any]] = []
        route_started = time.perf_counter()
        try:
            payload, frames = build_request(row, Path(self.config.frame_root), self.config, encode_images=True)
            provider = self.provider_factory()
        except Exception as error:
            artifact = {"question_id": qid, "video_id": row.get("video_id"), "prediction": None,
                        "terminal_status": "input_or_provider_initialisation_failure",
                        "failure_category": f"input_or_provider_initialisation_failure:{type(error).__name__}",
                        "selected_frame_count": int(row.get("selected_frame_count", 0)), "selected_frames": [],
                        "resource_totals": aggregate_attempts([]),
                        "route_wall_time_sec": time.perf_counter() - route_started, "run_fingerprint": run_fingerprint}
            self.store.terminal(qid, False, artifact)
            return artifact
        for retry in range(self.config.max_retries + 1):
            attempt_id = f"{qid}:{uuid.uuid4()}"
            # Conservative per-call guard based on configured output only; observed image/input cost is reconciled afterward.
            upper = self.config.per_request_budget_reserve_usd
            try:
                self.store.assert_budget(upper)
            except RuntimeError:
                totals = aggregate_attempts(attempts)
                artifact = {"question_id": qid, "video_id": row["video_id"], "prediction": None,
                            "terminal_status": "budget_guard_refused_no_request", "failure_category": "budget_guard_refused_no_request",
                            "selected_frame_count": len(frames), "selected_frames": frames, "resource_totals": totals,
                            "route_wall_time_sec": time.perf_counter() - route_started, "run_fingerprint": run_fingerprint}
                self.store.terminal(qid, False, artifact)
                return artifact
            self.store.append("request_start", {"question_id": qid, "attempt_id": attempt_id, "retry_index": retry,
                                                 "model": self.config.model, "image_count": len(frames)})
            call_started = time.perf_counter()
            try:
                outcome = provider.call(payload)
            except Exception as error:
                outcome = ProviderOutcome(status="provider_error", raw_text=None,
                                          latency_sec=time.perf_counter() - call_started,
                                          error_category=type(error).__name__)
            costs = attempt_cost(outcome, self.config.pricing_object)
            record = {
                "question_id": qid, "attempt_id": attempt_id, "retry_index": retry, "provider_status": outcome.status,
                "response_id": outcome.response_id, "raw_response_text": outcome.raw_text,
                "response_model": outcome.response_model, "stop_reason": outcome.stop_reason,
                "stop_sequence": outcome.stop_sequence,
                "parsed_prediction": parse_answer(outcome.raw_text) if outcome.raw_text is not None else None,
                "error_category": outcome.error_category,
                "input_tokens": outcome.input_tokens, "cache_creation_input_tokens": outcome.cache_creation_input_tokens,
                "cache_read_input_tokens": outcome.cache_read_input_tokens, "output_tokens": outcome.output_tokens,
                "latency_sec": outcome.latency_sec, **costs, "cache_aware_usd": costs["total_cache_aware_usd"],
            }
            self.store.append("attempt_end", record)
            self.store.charge_once(attempt_id, costs["total_cache_aware_usd"])
            attempts.append(record)
            if outcome.status == "response_received":
                prediction = record["parsed_prediction"]
                failure = None if prediction else (outcome.error_category or "invalid_answer_format")
                totals = aggregate_attempts(attempts)
                artifact = {
                    "question_id": qid, "video_id": row["video_id"], "prediction": prediction,
                    "raw_response_text": outcome.raw_text, "terminal_status": "final_answer" if prediction else "invalid_answer",
                    "response_model": outcome.response_model, "stop_reason": outcome.stop_reason,
                    "stop_sequence": outcome.stop_sequence,
                    "failure_category": failure, "selected_frame_count": len(frames),
                    "selected_frames": frames, "resource_totals": totals,
                    "route_wall_time_sec": time.perf_counter() - route_started, "run_fingerprint": run_fingerprint,
                }
                self.store.terminal(qid, bool(prediction), artifact)
                return artifact
        totals = aggregate_attempts(attempts)
        artifact = {"question_id": qid, "video_id": row["video_id"], "prediction": None,
                    "terminal_status": "provider_retry_exhausted", "failure_category": "provider_retry_exhausted",
                    "selected_frame_count": len(frames), "selected_frames": frames, "resource_totals": totals,
                    "route_wall_time_sec": time.perf_counter() - route_started, "run_fingerprint": run_fingerprint}
        self.store.terminal(qid, False, artifact)
        return artifact
