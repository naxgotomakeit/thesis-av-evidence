"""Shared, reproducible DINOv2 frame-feature cache for Methods B and C."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


MODEL_NAME = "facebook/dinov2-small"
CACHE_SCHEMA_VERSION = "coarse-segmentation-dinov2-cache-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_identity(frame_paths: list[Path]) -> str:
    """Return the deterministic identity bound by a per-video feature cache."""
    return hashlib.sha256(
        json.dumps(
            [
                {
                    "path": path.as_posix(),
                    "bytes": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                }
                for path in frame_paths
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class DINOv2FeatureExtractor:
    """Load frozen ViT-S/14 once and cache normalized CLS features per video."""

    def __init__(self, *, device: str, batch_size: int = 32):
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        started = time.perf_counter()
        self.processor = AutoImageProcessor.from_pretrained(MODEL_NAME, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            MODEL_NAME, local_files_only=True, use_safetensors=True
        ).to(self.device)
        self.model.eval()
        self.model_load_sec = time.perf_counter() - started
        self.load_count = 1

    def extract_or_load(
        self,
        *,
        video_id: str,
        frame_paths: list[Path],
        timestamps: np.ndarray,
        cache_dir: Path,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Return one exact cache consumed by both experimental segmenters."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        features_path = cache_dir / "features.npy"
        metadata_path = cache_dir / "metadata.json"
        identity = frame_identity(frame_paths)
        expected = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "video_id": str(video_id),
            "model": MODEL_NAME,
            "feature": "last_hidden_state_cls_token",
            "normalization": "l2",
            "frame_count": len(frame_paths),
            "frame_identity_sha256": identity,
            "timestamps": np.asarray(timestamps, dtype=np.float64).tolist(),
        }
        if metadata_path.is_file() and features_path.is_file():
            saved = json.loads(metadata_path.read_text(encoding="utf-8"))
            comparable = {key: saved.get(key) for key in expected}
            if comparable == expected and saved.get("features_sha256") == sha256_file(features_path):
                features = np.load(features_path)
                if features.shape == (len(frame_paths), 384) and np.isfinite(features).all():
                    return features, {**saved, "cache_hit": True, "extraction_sec": 0.0}
        started = time.perf_counter()
        batches: list[np.ndarray] = []
        with torch.inference_mode():
            for offset in range(0, len(frame_paths), self.batch_size):
                images = [
                    Image.open(path).convert("RGB")
                    for path in frame_paths[offset : offset + self.batch_size]
                ]
                inputs = self.processor(images=images, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                output = self.model(**inputs).last_hidden_state[:, 0, :]
                output = torch.nn.functional.normalize(output.float(), dim=-1)
                batches.append(output.cpu().numpy().astype(np.float32))
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        extraction_sec = time.perf_counter() - started
        features = np.concatenate(batches, axis=0)
        np.save(features_path, features)
        metadata = {
            **expected,
            "dtype": str(features.dtype),
            "dimension": int(features.shape[1]),
            "device": str(self.device),
            "batch_size": self.batch_size,
            "features_sha256": sha256_file(features_path),
            "cache_hit": False,
            "extraction_sec": extraction_sec,
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return features, metadata

