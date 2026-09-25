from __future__ import annotations

"""Local SigLIP retrieval over HourVideo frame embedding shards.

The HourVideo shards contain normalized image embeddings produced by
``google/siglip-base-patch16-224``.  Queries are encoded with the matching
SigLIP text tower and compared with those frame embeddings.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


_MODEL = None
_PROCESSOR = None


def _load_model():
    global _MODEL, _PROCESSOR
    if _MODEL is not None and _PROCESSOR is not None:
        return _MODEL, _PROCESSOR

    import torch
    from transformers import AutoProcessor, SiglipModel

    model_path = os.getenv("SIGLIP_MODEL_PATH", "google/siglip-base-patch16-224")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=os.getenv("HF_HUB_OFFLINE", "0") in ("1", "true", "yes"))
    model = SiglipModel.from_pretrained(model_path, local_files_only=os.getenv("HF_HUB_OFFLINE", "0") in ("1", "true", "yes"))
    device = os.getenv("SIGLIP_DEVICE", "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    model = model.to(device).eval()
    _MODEL, _PROCESSOR = model, processor
    return _MODEL, _PROCESSOR


def _shard_path(index_dir: Path, video_id: Optional[str]) -> Path:
    if index_dir.is_file() and index_dir.suffix == ".npz":
        return index_dir
    vid = str(video_id or "").strip()
    candidates = []
    if vid:
        candidates.extend(
            [
                index_dir / "shards" / f"{vid}.npz",
                index_dir / f"{vid}.npz",
                index_dir / vid / f"{vid}.npz",
            ]
        )
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"SigLIP shard not found under {index_dir} for video_id={vid!r}")


def _encode_query(query_text: str) -> np.ndarray:
    import torch

    model, processor = _load_model()
    device = next(model.parameters()).device
    inputs = processor(text=[str(query_text)], padding="max_length", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
    with torch.inference_mode():
        vec = model.get_text_features(**inputs)
    # Transformers versions differ: older releases return a tensor, while
    # newer releases may return BaseModelOutputWithPooling.
    if not hasattr(vec, "detach"):
        pooled = getattr(vec, "pooler_output", None)
        vec = pooled if pooled is not None else getattr(vec, "text_embeds", None)
    if vec is None or not hasattr(vec, "detach"):
        raise TypeError(f"Unsupported SigLIP text feature output: {type(vec)!r}")
    vec = vec.detach().float().cpu().numpy()[0]
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, 1e-12)


def query_siglip(index_dir: Path | str, query_text: str, *, video_id: Optional[str] = None, topk: int = 50) -> List[Tuple[str, float]]:
    """Return ``(start_end_doc_id, score)`` hits for VideoSEAL retrieval."""

    shard = _shard_path(Path(index_dir), video_id)
    data = np.load(shard, allow_pickle=False)
    vectors = np.asarray(data["embedding"], dtype=np.float32)
    timestamps = np.asarray(data["timestamps_sec"], dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) != len(timestamps):
        raise ValueError(f"Invalid SigLIP shard shapes in {shard}")

    q = _encode_query(query_text)
    scores = vectors @ q
    k = max(1, min(int(topk), len(scores)))
    # Keep the best score per integer-second anchor, then rank globally.
    best: dict[int, float] = {}
    for ts, score in zip(timestamps.tolist(), scores.tolist()):
        sec = max(0, int(round(float(ts))))
        best[sec] = max(float(score), best.get(sec, -float("inf")))
    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)[:k]
    # A one-second doc window is sufficient; VideoSEAL's diversity selector
    # later expands/selects non-overlapping visual inspection windows.
    return [(f"{sec}_{sec + 1}", score) for sec, score in ranked]
