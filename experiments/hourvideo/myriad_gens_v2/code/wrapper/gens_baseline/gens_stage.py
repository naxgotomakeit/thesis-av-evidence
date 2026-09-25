from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

from .access import AccessPolicy
from .config import sha256_file


class ContextOverflow(RuntimeError):
    pass


def chronological_gens_candidates(
    clip_candidates: Sequence[dict[str, Any]], max_frames: int = 256
) -> list[dict[str, Any]]:
    if len(clip_candidates) > max_frames:
        raise ValueError(f"GenS received {len(clip_candidates)} frames, maximum is {max_frames}")
    if clip_candidates and len({item["video_id"] for item in clip_candidates}) != 1:
        raise ValueError("GenS candidate batch must contain exactly one video")
    return sorted(
        (dict(item) for item in clip_candidates),
        key=lambda item: (item["timestamp_sec"], item["frame_index"]),
    )


def estimated_visual_tokens(frame_count: int, resolution: int = 112) -> int:
    if frame_count < 0 or frame_count > 256:
        raise ValueError("frame_count must be between 0 and 256")
    if resolution % 28:
        raise ValueError("Qwen exact resolution must be divisible by 28")
    return frame_count * (resolution // 28) ** 2


def load_instruction_template(profile: dict[str, Any]) -> str:
    path = Path(profile["_config_dir"]) / profile["gens"]["instruction_template_path"]
    expected = profile["gens"]["instruction_template_sha256"]
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"GenS instruction template hash drift: expected {expected}, got {actual}")
    value = path.read_text(encoding="utf-8")
    if value.count("{query}") != 1:
        raise ValueError("GenS instruction template must contain exactly one {query} placeholder")
    return value


def build_gens_messages(
    chronological_candidates: Sequence[dict[str, Any]],
    query: str,
    instruction_template: str,
    resolution: int = 112,
) -> list[dict[str, Any]]:
    if resolution != 112:
        raise ValueError("frozen GenS resolution is 112")
    ordered = chronological_gens_candidates(chronological_candidates)
    content: list[dict[str, Any]] = []
    for frame_number, candidate in enumerate(ordered, 1):
        content.append({"type": "text", "text": f"Frame Number [{frame_number}]"})
        content.append(
            {
                "type": "image",
                "image": f"file://{candidate['extracted_frame_path']}",
                "resized_height": resolution,
                "resized_width": resolution,
            }
        )
    content.append(
        {"type": "text", "text": instruction_template.replace("{query}", query)}
    )
    return [{"role": "user", "content": content}]


def validate_qwen_interface(model_path: str | Path) -> dict[str, Any]:
    import transformers
    from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

    path = Path(model_path)
    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    processor = AutoProcessor.from_pretrained(
        path, local_files_only=True, trust_remote_code=False
    )
    if config.model_type != "qwen2_5_vl":
        raise RuntimeError(f"unexpected GenS model_type: {config.model_type}")
    if config.architectures != ["Qwen2_5_VLForConditionalGeneration"]:
        raise RuntimeError(f"unexpected GenS architecture: {config.architectures}")
    if processor.__class__.__name__ != "Qwen2_5_VLProcessor":
        raise RuntimeError(f"unexpected GenS processor: {processor.__class__.__name__}")
    return {
        "transformers_version": transformers.__version__,
        "model_type": config.model_type,
        "architecture": Qwen2_5_VLForConditionalGeneration.__name__,
        "processor": processor.__class__.__name__,
        "torch_dtype": str(config.torch_dtype),
    }


class GenSSelector:
    def __init__(self, profile: dict[str, Any], policy: AccessPolicy, device: str = "cuda:0"):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        config = profile["gens"]
        self.profile = profile
        self.policy = policy
        self.device = torch.device(device)
        self.resolution = int(config["resolution"])
        self.context_limit = int(config["context_limit_tokens"])
        self.max_new_tokens = int(config["generation"]["max_new_tokens"])
        model_path = (Path(profile["_config_dir"]) / config["local_path"]).absolute()
        model_path = Path(policy.assert_read_allowed(model_path))
        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(self.device)
        self.model.eval()

    def _sync(self) -> None:
        import torch

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def select(
        self, query: str, clip_candidates: Sequence[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]], dict[str, float], int]:
        import torch
        from qwen_vl_utils import process_vision_info

        candidates = chronological_gens_candidates(clip_candidates)
        instruction = load_instruction_template(self.profile)
        for candidate in candidates:
            self.policy.assert_read_allowed(candidate["extracted_frame_path"])
        messages = build_gens_messages(
            candidates, query, instruction, resolution=self.resolution
        )

        self._sync()
        started = time.perf_counter()
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_read_started = time.perf_counter()
        image_inputs, video_inputs = process_vision_info(messages)
        image_read_ms = (time.perf_counter() - image_read_started) * 1000.0
        processor_started = time.perf_counter()
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if "image_grid_thw" in inputs:
            grids = inputs["image_grid_thw"]
            expected_grid = self.resolution // 14
            if any(int(row[1]) != expected_grid or int(row[2]) != expected_grid for row in grids):
                raise RuntimeError("GenS processor did not produce explicit 112x112 image grids")
        context_tokens = int(inputs["input_ids"].shape[1])
        if context_tokens > self.context_limit:
            raise ContextOverflow(
                f"GenS input has {context_tokens} tokens, limit is {self.context_limit}"
            )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        self._sync()
        preprocess_ms = (time.perf_counter() - started) * 1000.0
        processor_ms = (time.perf_counter() - processor_started) * 1000.0

        self._sync()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
        self._sync()
        generate_ms = (time.perf_counter() - started) * 1000.0
        trimmed = [output[len(input_ids) :] for input_ids, output in zip(inputs["input_ids"], generated)]
        raw_response = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        return raw_response, candidates, {
            "gens_image_read": image_read_ms,
            "gens_processor_preprocess": processor_ms,
            "gens_preprocess": preprocess_ms,
            "gens_generate": generate_ms,
        }, context_tokens

    def select_observed(
        self, query: str, clip_candidates: Sequence[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any], int, list[int], str, bool]:
        """Run the V2 long-output path while preserving the exact model input.

        ``return_dict_in_generate`` changes only the return container.  No scores,
        sampling options, logits processors, or stopping criteria are added.
        """
        import torch
        from qwen_vl_utils import process_vision_info

        candidates = chronological_gens_candidates(clip_candidates)
        instruction = load_instruction_template(self.profile)
        for candidate in candidates:
            self.policy.assert_read_allowed(candidate["extracted_frame_path"])
        messages = build_gens_messages(
            candidates, query, instruction, resolution=self.resolution
        )

        self._sync()
        started = time.perf_counter()
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_read_started = time.perf_counter()
        image_inputs, video_inputs = process_vision_info(messages)
        image_read_ms = (time.perf_counter() - image_read_started) * 1000.0
        processor_started = time.perf_counter()
        inputs = self.processor(
            text=[prompt], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
        if "image_grid_thw" in inputs:
            grids = inputs["image_grid_thw"]
            expected_grid = self.resolution // 14
            if any(int(row[1]) != expected_grid or int(row[2]) != expected_grid for row in grids):
                raise RuntimeError("GenS processor did not produce explicit 112x112 image grids")
        context_tokens = int(inputs["input_ids"].shape[1])
        if context_tokens + self.max_new_tokens > self.context_limit:
            raise ContextOverflow(
                f"GenS input {context_tokens} + output budget {self.max_new_tokens} "
                f"exceeds context limit {self.context_limit}"
            )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        self._sync()
        preprocess_ms = (time.perf_counter() - started) * 1000.0
        processor_ms = (time.perf_counter() - processor_started) * 1000.0

        self._sync()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
            )
        self._sync()
        generate_ms = (time.perf_counter() - started) * 1000.0
        sequence = generated.sequences[0]
        input_length = int(inputs["input_ids"].shape[1])
        generated_ids = [int(value) for value in sequence[input_length:].detach().cpu().tolist()]
        eos_ids = self.model.generation_config.eos_token_id
        if eos_ids is None:
            eos_set: set[int] = set()
        elif isinstance(eos_ids, int):
            eos_set = {eos_ids}
        else:
            eos_set = {int(value) for value in eos_ids}
        eos_observed = any(value in eos_set for value in generated_ids)
        if eos_observed:
            finish_reason = "eos"
        elif len(generated_ids) >= self.max_new_tokens:
            finish_reason = "length"
        else:
            finish_reason = "error"
        raw_response = self.processor.batch_decode(
            [sequence[input_length:]],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return raw_response, candidates, {
            "gens_image_read": image_read_ms,
            "gens_processor_preprocess": processor_ms,
            "gens_preprocess": preprocess_ms,
            "gens_generate": generate_ms,
            "prompt_text_sha256": __import__("hashlib").sha256(prompt.encode("utf-8")).hexdigest(),
        }, context_tokens, generated_ids, finish_reason, eos_observed

    def close(self) -> None:
        import torch

        self.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
