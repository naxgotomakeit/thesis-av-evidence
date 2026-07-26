from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image


class LocalQwenVLEngineeringModel:
    """Instrumented local Qwen-VL wrapper for bounded E1 engineering runs.

    The caller selects the frozen model profile.  This wrapper deliberately
    does not alter frame budgets, prompts, preprocessing, or decoding.
    """

    def __init__(
        self, *, model_path: Path, model_id: str, max_pixels: int, seed: int,
        empty_cache_after_call: bool = True, minimum_free_vram_gib: float = 7.0,
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration

        if not model_path.is_dir():
            raise RuntimeError(f"Engineering model is unavailable: {model_path}")
        if not torch.cuda.is_available():
            raise RuntimeError("Engineering dry-run requires CUDA")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Engineering dry-run GPU does not support BF16")
        self.model_id = model_id
        self.pre_model_load_memory = self._memory_snapshot()
        weight_bytes = sum(path.stat().st_size for path in model_path.glob("*.safetensors"))
        free_bytes, _ = torch.cuda.mem_get_info()
        # Retain the frozen BF16/no-quantization model behavior.  The free-VRAM
        # gate is an engineering preflight only, not a protocol change.
        required_free_bytes = max(
            weight_bytes + 2 * 1024**3,
            int(float(minimum_free_vram_gib) * 1024**3),
        )
        if free_bytes < required_free_bytes:
            raise RuntimeError(
                f"Insufficient free VRAM for {model_id}: "
                f"free={free_bytes / 1024**3:.2f} GiB, required>={required_free_bytes / 1024**3:.2f} GiB"
            )
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, max_pixels=max_pixels)
        config_payload = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        architectures = config_payload.get("architectures") or []
        model_class = (
            Qwen2_5_VLForConditionalGeneration
            if "Qwen2_5_VLForConditionalGeneration" in architectures
            else Qwen2VLForConditionalGeneration
        )
        self.model = model_class.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map={"": 0},
        )
        self.model.eval()
        torch.cuda.synchronize()
        self.model_load_time_s = time.perf_counter() - started
        self.actual_model_loading_mode = "bfloat16_unquantized"
        self.empty_cache_after_call = bool(empty_cache_after_call)
        self.post_model_load_memory = self._memory_snapshot()
        self.last_inference_telemetry: dict[str, Any] | None = None

    @staticmethod
    def _bytes_and_gib(value: int) -> dict[str, int | float]:
        return {"bytes": int(value), "gib": float(value / 1024**3)}

    @classmethod
    def _memory_snapshot(cls) -> dict[str, Any]:
        import torch

        free_bytes, total_bytes = torch.cuda.mem_get_info()
        allocated = int(torch.cuda.memory_allocated())
        reserved = int(torch.cuda.memory_reserved())
        snapshot = {
            "allocated": cls._bytes_and_gib(allocated),
            "reserved": cls._bytes_and_gib(reserved),
            "free": cls._bytes_and_gib(int(free_bytes)),
            "total": cls._bytes_and_gib(int(total_bytes)),
        }
        return snapshot

    @staticmethod
    def is_cuda_oom(exception: BaseException) -> bool:
        import torch

        return isinstance(exception, torch.OutOfMemoryError) or "out of memory" in str(exception).lower()

    def recover_after_oom(self) -> dict[str, Any]:
        """Clear only allocator caches; never alter model/protocol state."""
        import torch

        started = time.perf_counter()
        gc.collect()
        torch.cuda.empty_cache()
        telemetry = self._memory_snapshot()
        telemetry["recovery_time_s"] = time.perf_counter() - started
        baseline = self.post_model_load_memory["allocated"]["bytes"]
        telemetry["safe_to_continue"] = bool(
            telemetry["allocated"]["bytes"] <= baseline + 1024**3
            and telemetry["free"]["bytes"] >= 2 * 1024**3
        )
        return telemetry

    def infer(self, *, prompt: str, images: list[Image.Image], max_new_tokens: int) -> dict[str, Any]:
        import torch

        inputs: dict[str, Any] | None = None
        generated: Any = None
        generated_only: Any = None
        raw_output: str | None = None
        telemetry: dict[str, Any] = {
            "empty_cache_called": self.empty_cache_after_call,
            "before_preprocess": self._memory_snapshot(),
        }
        # This reset is intentionally before preprocessing: every persisted peak
        # covers the complete per-call visual/text input and generation path.
        torch.cuda.reset_peak_memory_stats()
        cleanup_started: float | None = None
        content: list[dict[str, Any]] = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt})
        try:
            rendered = self.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
            )
            processor_inputs: dict[str, Any] = {"text": [rendered], "padding": True, "return_tensors": "pt"}
            if images:
                processor_inputs["images"] = images
            preprocess_started = time.perf_counter()
            inputs = self.processor(**processor_inputs)
            device = next(self.model.parameters()).device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            text_tokens = len(self.processor.tokenizer(rendered, add_special_tokens=False)["input_ids"])
            total_input_tokens = int(inputs["input_ids"].shape[1])
            torch.cuda.synchronize()
            preprocess_time_s = time.perf_counter() - preprocess_started
            telemetry["before_generate"] = self._memory_snapshot()
            started = time.perf_counter()
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs, do_sample=False, use_cache=True, max_new_tokens=max_new_tokens
                )
            torch.cuda.synchronize()
            answer_model_time_s = time.perf_counter() - started
            generated_only = generated[:, total_input_tokens:]
            raw_output = self.processor.batch_decode(
                generated_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            telemetry["after_generate_before_cleanup"] = {
                **self._memory_snapshot(),
                "peak_allocated": self._bytes_and_gib(int(torch.cuda.max_memory_allocated())),
                "peak_reserved": self._bytes_and_gib(int(torch.cuda.max_memory_reserved())),
            }
            result = {
                "raw_output": raw_output,
                "answer_model_time_s": answer_model_time_s,
                "answer_preprocess_time_s": preprocess_time_s,
                "text_tokens": text_tokens,
                "visual_tokens": max(0, total_input_tokens - text_tokens),
                "total_input_tokens": total_input_tokens,
                "output_tokens": int(generated_only.shape[1]),
                "peak_vram_gib": float(torch.cuda.max_memory_allocated() / 1024**3),
            }
            return result
        except BaseException as exc:
            telemetry["exception"] = repr(exc)
            telemetry["is_cuda_oom"] = self.is_cuda_oom(exc)
            telemetry["at_exception_before_cleanup"] = {
                **self._memory_snapshot(),
                "peak_allocated": self._bytes_and_gib(int(torch.cuda.max_memory_allocated())),
                "peak_reserved": self._bytes_and_gib(int(torch.cuda.max_memory_reserved())),
            }
            raise
        finally:
            cleanup_started = time.perf_counter()
            # Keep only decoded text/scalars in result. All CUDA tensors and
            # processor containers are explicitly released before the snapshot.
            del inputs, generated, generated_only, content
            gc.collect()
            if self.empty_cache_after_call:
                torch.cuda.empty_cache()
            telemetry["after_cleanup"] = self._memory_snapshot()
            telemetry["cleanup_time_s"] = time.perf_counter() - cleanup_started
            self.last_inference_telemetry = telemetry

    def close(self) -> None:
        import torch

        del self.model
        del self.processor
        gc.collect()
        torch.cuda.empty_cache()


# Compatibility alias retained for the existing 2B engineering runner/tests.
Qwen2VL2BEngineeringModel = LocalQwenVLEngineeringModel
