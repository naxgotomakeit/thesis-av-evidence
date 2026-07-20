"""Strictly local Qwen2-VL loading and deterministic generation helpers."""

from __future__ import annotations

import gc
import hashlib
import os
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_xsym(path: Path) -> Path:
    data = path.read_bytes()
    if not data.startswith(b"XSym\n"):
        return path
    lines = data.decode("utf-8", errors="strict").splitlines()
    if len(lines) < 4:
        raise RuntimeError(f"Malformed XSym pointer: {path}")
    target = (path.parent / lines[3].strip()).resolve()
    if not target.is_file():
        raise FileNotFoundError(f"Qwen XSym target missing: {path} -> {target}")
    return target


def create_read_only_model_view(snapshot: Path, view: Path) -> tuple[Path, list[dict[str, Any]]]:
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Local Qwen snapshot unavailable: {snapshot}")
    view.mkdir(parents=True, exist_ok=True)
    records = []
    for source in sorted(snapshot.iterdir()):
        if not source.is_file() or source.name.startswith("._"):
            continue
        target = resolve_xsym(source)
        link = view / source.name
        if link.exists() or link.is_symlink():
            if link.resolve() != target.resolve():
                raise RuntimeError(f"Existing model-view link has wrong target: {link}")
        else:
            os.symlink(str(target), str(link))
        records.append(
            {
                "filename": source.name,
                "resolved_path": target.as_posix(),
                "bytes": target.stat().st_size,
                "content_identity": target.name,
                "sha256": sha256_file(target) if target.stat().st_size < 100 * 1024 * 1024 else None,
                "sha256_omitted_reason": None if target.stat().st_size < 100 * 1024 * 1024 else "large weight; immutable blob filename and size recorded",
            }
        )
    required = {"config.json", "preprocessor_config.json", "model.safetensors.index.json"}
    if not required <= {row["filename"] for row in records}:
        raise RuntimeError("Resolved local Qwen model view is incomplete")
    return view, records


class LocalQwen2VL:
    """One persistent, offline-only Qwen2-VL model instance."""

    def __init__(self, *, model_view: Path, seed: int = 0):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(model_view, local_files_only=True)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_view,
            local_files_only=True,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to(self.device)
        self.model.eval()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.model_load_sec = time.perf_counter() - started
        self.load_count = 1
        self.inference_count = 0
        self.model_instance_id = id(self.model)
        self.processor_instance_id = id(self.processor)
        self.model_vram_allocated_bytes = (
            int(torch.cuda.memory_allocated()) if self.device.type == "cuda" else 0
        )
        self.model_load_peak_vram_bytes = (
            int(torch.cuda.max_memory_allocated()) if self.device.type == "cuda" else 0
        )
    def _generate(
        self, *, prompt: str, image_paths: list[Path], max_new_tokens: int
    ) -> tuple[str, dict[str, Any]]:
        total_started = time.perf_counter()
        decode_started = time.perf_counter()
        images = []
        for path in image_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
        decode_sec = time.perf_counter() - decode_started
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        preprocess_started = time.perf_counter()
        processor_kwargs: dict[str, Any] = {
            "text": [rendered],
            "padding": True,
            "return_tensors": "pt",
        }
        if images:
            processor_kwargs["images"] = images
        inputs = self.processor(**processor_kwargs)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        preprocess_sec = time.perf_counter() - preprocess_started
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        inference_started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                use_cache=True,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        inference_sec = time.perf_counter() - inference_started
        input_length = int(inputs["input_ids"].shape[1])
        generated_only = generated[:, input_length:]
        text = self.processor.batch_decode(
            generated_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        output_tokens = int(generated_only.shape[1])
        peak_vram = int(torch.cuda.max_memory_allocated()) if self.device.type == "cuda" else 0
        self.inference_count += 1
        del inputs, generated, generated_only
        for image in images:
            image.close()
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return text, {
            "image_count": len(image_paths),
            "image_decode_sec": decode_sec,
            "image_preprocess_sec": preprocess_sec,
            "model_inference_sec": inference_sec,
            "total_semantic_generation_sec": time.perf_counter() - total_started,
            "input_tokens": input_length,
            "output_tokens": output_tokens,
            "peak_vram_bytes": peak_vram,
        }

    def describe_images(
        self, *, image_paths: list[Path], prompt: str, max_new_tokens: int
    ) -> tuple[str, dict[str, Any]]:
        if not image_paths:
            raise ValueError("Medium semantic generation requires at least one image")
        return self._generate(prompt=prompt, image_paths=image_paths, max_new_tokens=max_new_tokens)

    def summarize_text(self, *, ordered_descriptions: list[str], prompt: str, max_new_tokens: int) -> tuple[str, dict[str, Any]]:
        if not ordered_descriptions:
            raise ValueError("Coarse semantic aggregation requires Medium descriptions")
        numbered = "\n".join(f"{index}. {value}" for index, value in enumerate(ordered_descriptions, start=1))
        return self._generate(
            prompt=f"{prompt}\n\nOrdered Medium events:\n{numbered}",
            image_paths=[],
            max_new_tokens=max_new_tokens,
        )
