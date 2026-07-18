"""Explicit live-only external boundaries for the canonical baseline.

This module preserves the final Task 5A and Task 5C behavior. It contains no
credentials, gold data, case-specific routing, or frozen evidence injection.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

import yaml

from src.question_planner.task5a import JSON_OUTPUT_SCHEMA, ModelReply
from src.retrieval.task5c_v1_1 import FALLBACK_MODEL, staged_local_asr_fallback

from .state import CaseState


def environment_presence() -> dict[str, bool]:
    """Return secret presence flags without reading values into logs."""
    return {
        "anthropic_key_present": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "anthropic_model_present": bool(os.environ.get("ANTHROPIC_MODEL")),
        "gemini_key_present": bool(os.environ.get("GEMINI_API_KEY")),
    }


def anthropic_requester() -> tuple[Callable[[str, str, int], ModelReply], str]:
    """Create the exact Task 5A v2 request callback from environment config."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    model = os.environ.get("ANTHROPIC_MODEL")
    if not key or not model:
        raise RuntimeError("Required Anthropic environment configuration is unavailable")
    import anthropic

    client = anthropic.Anthropic(api_key=key)

    def request(system: str, user: str, number: int) -> ModelReply:
        del number
        started = time.perf_counter()
        response = client.messages.create(
            model=model,
            max_tokens=1200,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": JSON_OUTPUT_SCHEMA}},
        )
        latency = time.perf_counter() - started
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        return ModelReply(
            text=text,
            latency_sec=latency,
            input_tokens=int(response.usage.input_tokens),
            output_tokens=int(response.usage.output_tokens),
        )

    return request, model


def _explicit_or_search_interval(state: CaseState) -> tuple[tuple[float, float], tuple[float, float]]:
    for cue in state.deterministic_cues.get("time_cues", []):
        if cue.get("start_sec") is not None:
            start = float(cue["start_sec"])
            end = float(cue.get("end_sec", start))
            return (start, end), (start, end)
    intervals = (state.retrieval_result or {}).get("anchor_resolution", {}).get("final_search_intervals", [])
    if not intervals:
        raise RuntimeError("Task 5C fallback has no question-derived executable interval")
    interval = intervals[0]
    bounds = float(interval["start_sec"]), float(interval["end_sec"])
    return bounds, bounds


class LazyWhisperFallback:
    """Load Whisper-small only if the canonical insufficiency decision triggers it."""

    def __init__(self, project_root: Path):
        config = yaml.safe_load((project_root / "configs/audio_mvp.yaml").read_text(encoding="utf-8"))
        self.project_root = project_root
        self.config = config
        self.model_name = str(config.get("whisper_model", FALLBACK_MODEL))
        self.device = "not_loaded"
        self.model_load_latency_sec = 0.0
        self.model_load_count = 0
        self.fallback_calls = 0
        self._transcriber: Callable[[Any, int], dict[str, Any]] | None = None

    def _get(self) -> Callable[[Any, int], dict[str, Any]]:
        if self._transcriber is not None:
            return self._transcriber
        import torch
        import whisper

        started = time.perf_counter()
        requested = str(self.config.get("whisper_device", "cpu"))
        self.device = "cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu"
        model = whisper.load_model(
            self.model_name,
            device=self.device,
            download_root=str(self.config.get("whisper_cache_dir", "")) or None,
        )
        self.model_load_latency_sec = time.perf_counter() - started
        self.model_load_count += 1

        def transcribe(audio: Any, sample_rate: int) -> dict[str, Any]:
            if sample_rate != 16000:
                raise ValueError("Task 5C canonical local Whisper input must be 16 kHz")
            return model.transcribe(
                audio,
                fp16=self.device == "cuda",
                word_timestamps=False,
                condition_on_previous_text=False,
                verbose=None,
            )

        self._transcriber = transcribe
        return transcribe

    def _source_audio(self, video_id: str) -> Path:
        configured = os.environ.get("EGOSOUND_DATA_ROOT") or str(self.config.get("dataset_root", ""))
        if not configured:
            raise RuntimeError("EgoSound data root is not configured")
        path = Path(configured) / "data" / "Ego4d" / "audios" / f"{video_id}.wav"
        if not path.is_file():
            raise FileNotFoundError(f"Required local fallback WAV is missing: {path}")
        return path

    def execute(self, state: CaseState, context: dict[str, Any]) -> dict[str, Any]:
        """Execute one final-policy fallback; no frozen evidence is accepted."""
        del context
        self.fallback_calls += 1
        hard, decode = _explicit_or_search_interval(state)
        phrases = [str(item["text"]) for item in state.deterministic_cues.get("quoted_phrases", []) if item.get("text")]
        source = self._source_audio(state.video_id)
        total_started = time.perf_counter()
        transcriber = self._get()
        result = staged_local_asr_fallback(
            case_id=state.case_id,
            video_id=state.video_id,
            search_interval=decode,
            hard_question_interval=hard,
            phrases=phrases,
            source_audio=source,
            transcriber=transcriber,
            model_name=self.model_name,
        )
        result["model_device"] = self.device
        result["model_load_latency_sec"] = self.model_load_latency_sec
        result["canonical_local_asr_total_sec"] = time.perf_counter() - total_started
        result["canonical_execution"] = "live_fallback_executed_once"
        return result

    def lifecycle_audit(self) -> dict[str, Any]:
        """Report lazy Whisper lifecycle without exposing model internals."""
        return {
            "model": self.model_name,
            "device": self.device,
            "load_count": self.model_load_count,
            "fallback_calls": self.fallback_calls,
            "model_load_latency_sec": self.model_load_latency_sec if self.model_load_count else None,
            "currently_loaded": self._transcriber is not None,
            "persistent_reuse_invariant": self.model_load_count <= 1,
        }
