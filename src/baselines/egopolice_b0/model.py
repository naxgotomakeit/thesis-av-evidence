from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image

from .core import BaselineInputError, parse_prediction


def checkpoint_weight_bytes(model_path: Path) -> int:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        names = sorted(set(index.get("weight_map", {}).values()))
        return sum((model_path / name).stat().st_size for name in names if (model_path / name).is_file())
    return sum(path.stat().st_size for path in model_path.glob("*.safetensors"))


SUPPORTED_DTYPES = {"bfloat16", "float16", "float32"}
SUPPORTED_QUANTIZATION_MODES = {"none", "nf4_4bit"}


def validate_model_loading_settings(dtype: str, quantization_mode: str) -> None:
    if dtype not in SUPPORTED_DTYPES:
        raise BaselineInputError(f"Unsupported model dtype: {dtype}")
    if quantization_mode not in SUPPORTED_QUANTIZATION_MODES:
        raise BaselineInputError(f"Unsupported quantization mode: {quantization_mode}")


def preflight_environment(
    model_path: Path, *, dtype: str, quantization_mode: str,
) -> dict[str, Any]:
    import torch
    import transformers

    validate_model_loading_settings(dtype, quantization_mode)
    required_files = ["config.json", "preprocessor_config.json", "model.safetensors.index.json"]
    missing_files = [name for name in required_files if not (model_path / name).is_file()]
    packages: dict[str, bool] = {}
    package_versions: dict[str, str | None] = {}
    package_errors: dict[str, str | None] = {}
    for name in ("bitsandbytes", "accelerate"):
        try:
            module = importlib.import_module(name)
            packages[name] = True
            package_versions[name] = str(getattr(module, "__version__", "unknown"))
            package_errors[name] = None
        except Exception as exc:
            packages[name] = False
            package_versions[name] = None
            package_errors[name] = repr(exc)
    cuda = torch.cuda.is_available()
    vram_bytes = int(torch.cuda.get_device_properties(0).total_memory) if cuda else 0
    free_vram_bytes = int(torch.cuda.mem_get_info()[0]) if cuda else 0
    weight_bytes = checkpoint_weight_bytes(model_path) if model_path.is_dir() else 0
    checkpoint_config = {}
    if (model_path / "config.json").is_file():
        checkpoint_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    checkpoint_quantized = bool(checkpoint_config.get("quantization_config"))
    errors = []
    if not model_path.is_dir():
        errors.append(f"model_path_not_found:{model_path}")
    if missing_files:
        errors.append("missing_checkpoint_files:" + ",".join(missing_files))
    if not cuda:
        errors.append("cuda_unavailable")
    elif dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        errors.append("bfloat16_unsupported_by_gpu")
    if not packages["accelerate"]:
        errors.append("accelerate_unavailable")
    if quantization_mode == "nf4_4bit" and not packages["bitsandbytes"]:
        errors.append("bitsandbytes_unavailable")
    if quantization_mode == "none" and cuda and weight_bytes > vram_bytes:
        errors.append("unquantized_checkpoint_exceeds_total_vram")
    return {
        "passed": not errors,
        "errors": errors,
        "model_path": str(model_path),
        "checkpoint_weight_bytes": weight_bytes,
        "checkpoint_quantized": checkpoint_quantized,
        "requested_dtype": dtype,
        "requested_quantization_mode": quantization_mode,
        "cuda_available": cuda,
        "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
        "total_vram_bytes": vram_bytes,
        "free_vram_bytes": free_vram_bytes,
        "packages": packages,
        "package_import_errors": package_errors,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            **package_versions,
        },
    }


class Qwen25VL7BBaseline:
    def __init__(
        self, *, model_path: Path, max_pixels: int, seed: int, dtype: str,
        quantization: dict[str, Any],
    ):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

        quantization_mode = str(quantization["mode"])
        validate_model_loading_settings(dtype, quantization_mode)
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        load_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "device_map": {"": 0},
            "dtype": torch_dtype,
        }
        if quantization_mode == "nf4_4bit":
            compute_dtype_name = str(quantization["bnb_4bit_compute_dtype"])
            validate_model_loading_settings(compute_dtype_name, quantization_mode)
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=str(quantization["bnb_4bit_quant_type"]),
                bnb_4bit_compute_dtype={
                    "bfloat16": torch.bfloat16,
                    "float16": torch.float16,
                    "float32": torch.float32,
                }[compute_dtype_name],
                bnb_4bit_use_double_quant=bool(quantization["bnb_4bit_use_double_quant"]),
            )
        load_started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True, max_pixels=int(max_pixels)
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            **load_kwargs,
        )
        self.model.eval()
        torch.cuda.synchronize()
        self.model_load_latency_sec = time.perf_counter() - load_started
        self.model_path = model_path
        self.actual_model_loading_mode = (
            "nf4_4bit" if quantization_mode == "nf4_4bit" else f"{dtype}_unquantized"
        )

    def infer(
        self, *, images: list[Image.Image], prompt: str, generation: dict[str, Any]
    ) -> dict[str, Any]:
        import torch

        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[rendered], images=images, padding=True, return_tensors="pt"
        )
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        text_tokens = len(self.processor.tokenizer(rendered, add_special_tokens=False)["input_ids"])
        total_input_tokens = int(inputs["input_ids"].shape[1])
        visual_tokens = max(0, total_input_tokens - text_tokens)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=int(generation["max_new_tokens"]),
                do_sample=False,
                use_cache=bool(generation["use_cache"]),
            )
        torch.cuda.synchronize()
        inference_latency = time.perf_counter() - started
        generated_only = generated[:, total_input_tokens:]
        raw = self.processor.batch_decode(
            generated_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        output_tokens = int(generated_only.shape[1])
        prediction = parse_prediction(raw)
        return {
            "prediction_index": prediction,
            "raw_output": raw,
            "inference_latency_sec": inference_latency,
            "text_token_count": text_tokens,
            "visual_token_count": visual_tokens,
            "total_input_token_count": total_input_tokens,
            "output_token_count": output_tokens,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        }
