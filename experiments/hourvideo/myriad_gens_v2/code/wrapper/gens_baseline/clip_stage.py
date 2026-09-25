from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

from .access import AccessPolicy


def rank_clip_candidates(
    candidates: Sequence[dict[str, Any]], scores: Sequence[float], top_k: int = 256
) -> list[dict[str, Any]]:
    if len(candidates) != len(scores):
        raise ValueError("candidate and score counts differ")
    if not candidates:
        return []
    video_ids = {candidate["video_id"] for candidate in candidates}
    if len(video_ids) != 1:
        raise ValueError("CLIP candidate batch must contain exactly one video")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    deduplicated: dict[tuple[str, int], dict[str, Any]] = {}
    for candidate, score in zip(candidates, scores):
        item = dict(candidate)
        item["clip_score"] = float(score)
        key = (item["video_id"], int(item["frame_index"]))
        previous = deduplicated.get(key)
        if previous is None or (
            -item["clip_score"], item["timestamp_sec"], item["frame_index"]
        ) < (
            -previous["clip_score"], previous["timestamp_sec"], previous["frame_index"]
        ):
            deduplicated[key] = item

    ranked = sorted(
        deduplicated.values(),
        key=lambda item: (-item["clip_score"], item["timestamp_sec"], item["frame_index"]),
    )[:top_k]
    for rank, item in enumerate(ranked, 1):
        item["clip_rank"] = rank
    return ranked


class ClipRetriever:
    def __init__(self, profile: dict[str, Any], policy: AccessPolicy, device: str = "cuda:0"):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        config = profile["clip"]
        self.profile = profile
        self.policy = policy
        self.device = torch.device(device)
        self.resolution = int(config["resolution"])
        self.batch_size = int(config["batch_size"])
        self.top_k = int(config["top_k"])
        self.max_length = int(config["text_max_length"])
        model_path = Path(profile["_config_dir"]) / config["local_path"]
        model_path = Path(policy.assert_read_allowed(model_path.absolute()))

        torch.manual_seed(int(profile["determinism"]["seed"]))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(profile["determinism"]["seed"]))
        torch.use_deterministic_algorithms(
            bool(profile["determinism"]["torch_deterministic_algorithms"])
        )
        torch.backends.cudnn.benchmark = bool(profile["determinism"]["cudnn_benchmark"])
        torch.backends.cuda.matmul.allow_tf32 = bool(profile["determinism"]["cuda_tf32"])

        self.processor = CLIPProcessor.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
        image_processor = self.processor.image_processor
        crop_size = image_processor.crop_size
        crop_height = crop_size["height"] if isinstance(crop_size, dict) else crop_size
        crop_width = crop_size["width"] if isinstance(crop_size, dict) else crop_size
        if (crop_height, crop_width) != (self.resolution, self.resolution):
            raise RuntimeError(
                f"CLIP processor crop is {(crop_height, crop_width)}, expected 224x224"
            )
        self.model = CLIPModel.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.model.eval()

    def _sync(self) -> None:
        import torch

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def retrieve(
        self, query: str, candidates: Sequence[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, float]]:
        import torch
        import torch.nn.functional as functional
        from PIL import Image

        if not candidates:
            return [], {"text_token_count": 0, "text_was_truncated": False}, {
                "clip_text": 0.0,
                "clip_images": 0.0,
                "clip_sort": 0.0,
            }
        if len({candidate["video_id"] for candidate in candidates}) != 1:
            raise ValueError("cross-video CLIP retrieval is forbidden")

        full_ids = self.processor.tokenizer(
            query, add_special_tokens=True, truncation=False
        )["input_ids"]
        text_metadata = {
            "text_token_count": len(full_ids),
            "text_max_length": self.max_length,
            "text_was_truncated": len(full_ids) > self.max_length,
        }

        self._sync()
        started = time.perf_counter()
        text_inputs = self.processor(
            text=[query],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        text_inputs = {
            key: value.to(self.device)
            for key, value in text_inputs.items()
            if key in {"input_ids", "attention_mask"}
        }
        with torch.inference_mode():
            text_features = self.model.get_text_features(**text_inputs)
            text_features = functional.normalize(text_features.float(), p=2, dim=-1)
        self._sync()
        text_ms = (time.perf_counter() - started) * 1000.0

        image_features = []
        self._sync()
        started = time.perf_counter()
        for offset in range(0, len(candidates), self.batch_size):
            batch = candidates[offset : offset + self.batch_size]
            images = []
            try:
                for candidate in batch:
                    image_path = self.policy.assert_read_allowed(candidate["extracted_frame_path"])
                    images.append(Image.open(image_path).convert("RGB"))
                inputs = self.processor(images=images, return_tensors="pt")
                pixels = inputs["pixel_values"]
                if tuple(pixels.shape[-2:]) != (self.resolution, self.resolution):
                    raise RuntimeError(
                        f"CLIP pixel_values are {tuple(pixels.shape[-2:])}, expected 224x224"
                    )
                with torch.inference_mode():
                    features = self.model.get_image_features(
                        pixel_values=pixels.to(self.device, dtype=torch.float32)
                    )
                    features = functional.normalize(features.float(), p=2, dim=-1)
                image_features.append(features.cpu())
            finally:
                for image in images:
                    image.close()
        self._sync()
        image_ms = (time.perf_counter() - started) * 1000.0

        matrix = torch.cat(image_features, dim=0)
        scores = (matrix @ text_features.cpu().T).squeeze(-1).tolist()
        started = time.perf_counter()
        ranked = rank_clip_candidates(candidates, scores, self.top_k)
        sort_ms = (time.perf_counter() - started) * 1000.0
        return ranked, text_metadata, {
            "clip_text": text_ms,
            "clip_images": image_ms,
            "clip_sort": sort_ms,
        }

    def close(self) -> None:
        import torch

        self.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

