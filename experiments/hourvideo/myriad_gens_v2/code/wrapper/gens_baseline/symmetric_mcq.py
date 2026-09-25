from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .access import AccessPolicy
from .config import resolve_profile_path, sha256_file
from .query import OPTION_KEYS, normalize_field


INTERNAL_CLIP_KEYS = {
    "clip_option_scores",
    "clip_winning_option_label",
}


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_symmetric_template(profile: dict[str, Any]) -> str:
    config = profile["symmetric_mcq"]
    path = resolve_profile_path(profile, config["clip_query_template_path"])
    actual = sha256_file(path)
    if actual != config["clip_query_template_sha256"]:
        raise ValueError("symmetric CLIP query template hash drift")
    template = path.read_text(encoding="utf-8")
    if template.count("{question}") != 1 or template.count("{option}") != 1:
        raise ValueError("symmetric template must contain question and option exactly once")
    return template


def build_option_queries(record: dict[str, Any], template: str) -> list[dict[str, str]]:
    question = normalize_field(record["question"])
    options = record["options"]
    if not isinstance(options, dict) or tuple(options.keys()) != OPTION_KEYS:
        raise ValueError("options must contain exactly A-E in order")
    result = []
    for label in OPTION_KEYS:
        option = normalize_field(options[label])
        query = template.format(question=question, option=option)
        result.append(
            {
                "option_label": label,
                "option_text": option,
                "option_text_sha256": text_sha256(option),
                "query": query,
                "query_sha256": text_sha256(query),
            }
        )
    return result


def aggregate_option_scores(
    candidates: Sequence[dict[str, Any]],
    scores_by_label: dict[str, Sequence[float]],
    top_k: int = 256,
) -> list[dict[str, Any]]:
    if tuple(scores_by_label.keys()) != OPTION_KEYS:
        raise ValueError("option score maps must be ordered A-E")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if candidates and len({item["video_id"] for item in candidates}) != 1:
        raise ValueError("cross-video symmetric retrieval is forbidden")
    for label in OPTION_KEYS:
        if len(scores_by_label[label]) != len(candidates):
            raise ValueError(f"score count mismatch for option {label}")

    deduplicated: dict[tuple[str, int], dict[str, Any]] = {}
    for row_index, candidate in enumerate(candidates):
        option_scores = {
            label: float(scores_by_label[label][row_index]) for label in OPTION_KEYS
        }
        if not all(math.isfinite(value) for value in option_scores.values()):
            raise ValueError("NaN/Inf in option score")
        maximum = max(option_scores.values())
        winner = next(label for label in OPTION_KEYS if option_scores[label] == maximum)
        item = dict(candidate)
        item["clip_option_scores"] = option_scores
        item["clip_score"] = maximum
        item["clip_winning_option_label"] = winner
        key = (str(item["video_id"]), int(item["frame_index"]))
        previous = deduplicated.get(key)
        if previous is None or (
            -item["clip_score"], item["timestamp_sec"], item["frame_index"]
        ) < (-previous["clip_score"], previous["timestamp_sec"], previous["frame_index"]):
            deduplicated[key] = item
    ranked = sorted(
        deduplicated.values(),
        key=lambda item: (-item["clip_score"], item["timestamp_sec"], item["frame_index"]),
    )[:top_k]
    for rank, item in enumerate(ranked, 1):
        item["clip_rank"] = rank
    return ranked


def gens_candidates_without_clip_option_signals(
    candidates: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in candidate.items() if key not in INTERNAL_CLIP_KEYS}
        for candidate in candidates
    ]


class CachedSymmetricClipRetriever:
    def __init__(self, profile: dict[str, Any], policy: AccessPolicy, device: str = "cuda:0"):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        config = profile["clip"]
        cache = profile["cached_image_embeddings"]
        self.profile = profile
        self.policy = policy
        self.device = torch.device(device)
        self.max_length = int(config["text_max_length"])
        self.top_k = int(config["top_k"])
        self.embedding_root = Path(
            policy.assert_read_allowed(resolve_profile_path(profile, cache["root"]))
        )
        sums = self.embedding_root / "CACHE_SHA256SUMS.txt"
        if sha256_file(sums) != cache["global_tree_sha256"]:
            raise RuntimeError("cached image embedding tree SHA drift")
        model_path = Path(
            policy.assert_read_allowed(resolve_profile_path(profile, config["local_path"]))
        )
        torch.manual_seed(int(profile["determinism"]["seed"]))
        torch.cuda.manual_seed_all(int(profile["determinism"]["seed"]))
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.processor = CLIPProcessor.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
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
            torch.cuda.synchronize(0)

    def _load_matrix(
        self, video_id: str, candidates: Sequence[dict[str, Any]]
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        directory = self.embedding_root / "videos" / video_id
        matrix_path = Path(self.policy.assert_read_allowed(directory / "image_embeddings.float32.npy"))
        registry_path = Path(self.policy.assert_read_allowed(directory / "frame_registry.json"))
        matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        if matrix.shape != (len(candidates), 768) or matrix.dtype != np.float32:
            raise RuntimeError("cached embedding shape/dtype mismatch")
        if not np.isfinite(matrix).all():
            raise RuntimeError("NaN/Inf in cached image embeddings")
        for index, (candidate, row) in enumerate(zip(candidates, registry)):
            if (
                row["row_index"] != index
                or row["frame_index"] != candidate["frame_index"]
                or row["timestamp_sec"] != candidate["timestamp_sec"]
                or row["frame_sha256"] != candidate["cache_sha256"]
            ):
                raise RuntimeError(f"candidate/embedding registry mismatch at row {index}")
        return matrix, registry

    def retrieve(
        self,
        option_queries: Sequence[dict[str, str]],
        candidates: Sequence[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        import torch
        import torch.nn.functional as functional

        if [item["option_label"] for item in option_queries] != list(OPTION_KEYS):
            raise ValueError("five ordered option queries are required")
        matrix, _ = self._load_matrix(candidates[0]["video_id"], candidates)
        scores: dict[str, list[float]] = {}
        metadata = []
        query_latencies: dict[str, float] = {}
        similarity_latencies: dict[str, float] = {}
        for item in option_queries:
            query = item["query"]
            full = self.processor.tokenizer(
                query, add_special_tokens=True, truncation=False, return_offsets_mapping=True
            )
            truncated = self.processor.tokenizer(
                query,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                return_offsets_mapping=True,
            )
            covered_end = max((end for _, end in truncated["offset_mapping"]), default=0)
            question_start = query.index("Question: ") + len("Question: ")
            question_end = query.index("\nCandidate answer:")
            option_start = query.index("Candidate answer: ") + len("Candidate answer: ")
            option_end = len(query.rstrip("\n"))
            def retention(start: int, end: int) -> str:
                return "full" if covered_end >= end else ("absent" if covered_end <= start else "partial")

            self._sync()
            started = time.perf_counter()
            inputs = self.processor(
                text=[query],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
                if key in {"input_ids", "attention_mask"}
            }
            with torch.inference_mode():
                feature = self.model.get_text_features(**inputs)
                feature = functional.normalize(feature.float(), p=2, dim=-1)
            self._sync()
            latency_ms = (time.perf_counter() - started) * 1000.0
            label = item["option_label"]
            query_latencies[label] = latency_ms
            similarity_started = time.perf_counter()
            matrix_tensor = torch.from_numpy(np.asarray(matrix))
            scores[label] = (matrix_tensor @ feature.cpu().T).squeeze(-1).tolist()
            similarity_latencies[label] = (time.perf_counter() - similarity_started) * 1000.0
            metadata.append(
                {
                    **item,
                    "untruncated_token_count": len(full["input_ids"]),
                    "truncated_token_count": len(truncated["input_ids"]),
                    "text_was_truncated": len(full["input_ids"]) > self.max_length,
                    "decoded_truncated_text": self.processor.tokenizer.decode(
                        truncated["input_ids"],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ),
                    "question_retention": retention(question_start, question_end),
                    "option_retention": retention(option_start, option_end),
                    "text_encoding_latency_ms": latency_ms,
                }
            )
        started = time.perf_counter()
        ranked = aggregate_option_scores(candidates, scores, self.top_k)
        aggregate_ms = (time.perf_counter() - started) * 1000.0
        return ranked, metadata, {
            "per_option_text_encoding_ms": query_latencies,
            "total_text_encoding_ms": sum(query_latencies.values()),
            "per_option_similarity_ms": similarity_latencies,
            "total_similarity_ms": sum(similarity_latencies.values()),
            "aggregation_top256_ms": aggregate_ms,
            "image_reencoding_count": 0,
        }

    def close(self) -> None:
        import torch

        self.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
