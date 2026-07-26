"""Shared C-RADIO text/event retrieval adapter for Formal E2.

The model is loaded only by an explicit later offline/online indexing command;
importing this module performs no model download, inference, or CUDA work.
"""
from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .core import normalize_rows, rank_events


RADIO_HF_REPO = "nvidia/C-RADIOv4-SO400M"
RADIO_HF_REVISION = "c0457f5dc26ca145f954cd4fc5bb6114e5705ad8"
RADIO_CHECKPOINT_FILENAME = "c-radio_v4-so400m_half.pth.tar"
RADIO_GITHUB_REVISION = "c0f37017930e9dda53f93424cf4bf39fc51f287e"
RADIO_ADAPTOR = "siglip2-g"
SIGLIP2_TEXT_MODEL = "google/siglip2-giant-opt-patch16-384"
INPUT_RESOLUTION = (512, 512)


class SharedCRadioAlignedEncoder:
    """One model/adaptor instance used identically by B1, B1′, and B2."""

    def __init__(self, *, cache_root: Path, device_name: str) -> None:
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_root / "huggingface")
        os.environ["TORCH_HOME"] = str(cache_root / "torch")
        import torch
        from huggingface_hub import hf_hub_download
        from src.diagnostics.cradio_v4.runner import _load_model

        if not torch.cuda.is_available():
            raise RuntimeError("Formal E2 C-RADIO retrieval needs CUDA")
        self.torch = torch
        self.device = torch.device(device_name)
        self.checkpoint_path = Path(hf_hub_download(
            repo_id=RADIO_HF_REPO, filename=RADIO_CHECKPOINT_FILENAME,
            revision=RADIO_HF_REVISION, local_files_only=True,
            # Pass the cache explicitly: huggingface_hub may resolve HF_HOME
            # during import, before this adapter receives its cache_root.
            cache_dir=cache_root / "huggingface" / "hub",
        ))
        started = time.perf_counter()
        self.model, _, self.load_warnings, self.model_load_time_s = _load_model(self.device, self.checkpoint_path)
        self.model_load_time_s = time.perf_counter() - started

    def encode_images(self, images: Iterable[Any], *, batch_size: int = 4) -> tuple[np.ndarray, dict[str, Any]]:
        """Return normalized siglip2-g aligned visual embeddings from cached 1FPS RGB images."""
        import torch
        from torch.nn import functional as F

        values = list(images)
        if not values:
            raise ValueError("Cannot encode an empty event-index image batch")
        chunks: list[np.ndarray] = []
        preprocess_s = inference_s = 0.0
        torch.cuda.reset_peak_memory_stats(self.device)
        with torch.inference_mode():
            for offset in range(0, len(values), batch_size):
                started = time.perf_counter()
                rgb = np.stack([np.asarray(item.convert("RGB"), dtype=np.uint8) for item in values[offset:offset + batch_size]])
                batch = torch.from_numpy(rgb).permute(0, 3, 1, 2).contiguous().to(self.device, dtype=torch.float32).div_(255.0)
                batch = F.interpolate(batch, size=INPUT_RESOLUTION, mode="bilinear", align_corners=False)
                preprocess_s += time.perf_counter() - started
                started = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = self.model(batch)
                    vector = output[RADIO_ADAPTOR][0]
                vector = F.normalize(vector.float(), dim=-1)
                torch.cuda.synchronize(self.device)
                inference_s += time.perf_counter() - started
                chunks.append(vector.cpu().numpy().astype(np.float32))
                del batch, output, vector
        embeddings = normalize_rows(np.concatenate(chunks, axis=0))
        return embeddings, {
            "model": RADIO_HF_REPO, "model_revision": RADIO_HF_REVISION, "adaptor": RADIO_ADAPTOR,
            "aligned_text_model": SIGLIP2_TEXT_MODEL, "embedding_dimension": int(embeddings.shape[1]),
            "preprocess_time_s": preprocess_s, "visual_inference_time_s": inference_s,
            "peak_vram_gib": torch.cuda.max_memory_allocated(self.device) / 1024**3,
        }

    def encode_question(self, question: str) -> tuple[np.ndarray, dict[str, Any]]:
        """The shared aligned text encoder; it never receives GT/options for retrieval."""
        torch = self.torch
        started = time.perf_counter()
        tokenized = self.model.adaptors[RADIO_ADAPTOR].tokenizer([str(question)])
        if hasattr(tokenized, "to"):
            tokenized = tokenized.to(self.device)
        else:
            tokenized = {key: value.to(self.device) for key, value in tokenized.items()}
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            vector = self.model.adaptors[RADIO_ADAPTOR].encode_text(tokenized, normalize=True)
        torch.cuda.synchronize(self.device)
        output = normalize_rows(vector.float().cpu().numpy()[0])
        return output, {
            "query_definition": "raw_question_only", "answer_fields_used_for_retrieval": False,
            "gt_used_for_retrieval": False, "retrieval_time_s": time.perf_counter() - started,
            "embedding_dimension": int(output.shape[0]),
        }

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        self.torch.cuda.empty_cache()


def retrieve_events(*, encoder: SharedCRadioAlignedEncoder, question: str, events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query, metadata = encoder.encode_question(question)
    return rank_events(query_embedding=query, events=events), metadata
