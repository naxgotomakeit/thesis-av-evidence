"""Fresh question-conditioned scoring over reusable offline indexes.

Source lineage is the validated Task 4 implementation in
``scripts/run_three_channel_retrieval.py``.  The raw question is encoded with
OpenAI CLIP ViT-B/32, Sentence-T5, or CLAP text encoding exactly as Task 4 did;
cosine ranking reuses ``src.retrieval.three_channel.rank_index``.  Historical
Task 4 score JSON files are regression references, never runtime inputs here.
"""

from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml

from src.retrieval.three_channel import rank_index, safe_retrieval_case


MODALITIES = ("visual", "speech", "acoustic")


class IndexContractError(ValueError):
    """Raised when an offline index is incompatible with its shared encoder."""


def _schema_hash(metadata: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    shape = {
        "metadata_fields": sorted(metadata),
        "row_fields": sorted({key for row in rows for key in row}),
    }
    return hashlib.sha256(
        json.dumps(shape, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_index_bundle_metadata(
    *,
    modality: str,
    video_id: str,
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
    embeddings: np.ndarray,
    encoder_config: dict[str, Any],
    prior_signature: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one video index without constructing or reloading an encoder."""
    if modality not in MODALITIES:
        raise IndexContractError(f"Unsupported index modality: {modality}")
    if str(metadata.get("video_id")) != str(video_id):
        raise IndexContractError(
            f"{modality} index video identity mismatch: expected {video_id}, "
            f"found {metadata.get('video_id')}"
        )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(rows):
        raise IndexContractError(
            f"{modality} index row/dimension mismatch for {video_id}: "
            f"shape={embeddings.shape}, rows={len(rows)}"
        )
    expected_model = encoder_config["name"]
    actual_model = metadata.get("encoder") if modality == "visual" else metadata.get("model")
    if actual_model != expected_model:
        raise IndexContractError(
            f"{modality} encoder mismatch for {video_id}: expected {expected_model}, found {actual_model}"
        )
    expected_revision = encoder_config.get("revision") if modality == "acoustic" else None
    actual_revision = metadata.get("model_revision") if modality == "acoustic" else None
    if expected_revision != actual_revision:
        raise IndexContractError(
            f"{modality} encoder revision mismatch for {video_id}: "
            f"expected {expected_revision}, found {actual_revision}"
        )
    if modality in {"speech", "acoustic"} and metadata.get("embedding_normalized") is not True:
        raise IndexContractError(f"{modality} index does not declare normalized embeddings: {video_id}")
    required_row_field = {
        "visual": "region_id",
        "speech": "transcript_segment_id",
        "acoustic": "acoustic_region_id",
    }[modality]
    if any(required_row_field not in row for row in rows):
        raise IndexContractError(f"{modality} index rows lack {required_row_field}: {video_id}")
    validation = metadata.get("validation") or {}
    declared_dimension = validation.get("dimension")
    if declared_dimension is not None and int(declared_dimension) != int(embeddings.shape[1]):
        raise IndexContractError(f"{modality} declared embedding dimension is inconsistent: {video_id}")
    signature = {
        "modality": modality,
        "encoder": actual_model,
        "revision": actual_revision,
        "embedding_dimension": int(embeddings.shape[1]),
        "embedding_dtype": str(embeddings.dtype),
        "metadata_declares_normalized": metadata.get("embedding_normalized"),
        "runtime_similarity": "normalized_cosine",
        "schema_hash": _schema_hash(metadata, rows),
    }
    if prior_signature is not None and signature != prior_signature:
        differences = {
            key: {"expected": prior_signature.get(key), "actual": signature.get(key)}
            for key in sorted(set(signature) | set(prior_signature))
            if prior_signature.get(key) != signature.get(key)
        }
        raise IndexContractError(
            f"{modality} index contract changed across videos at {video_id}: {differences}"
        )
    return signature


@dataclass
class QueryScoreResult:
    """Fresh scores and online timing for one modality and question."""

    modality: str
    query_text: str
    query_vector: np.ndarray
    score_map: dict[str, dict[str, Any]]
    ranked_results: list[dict[str, Any]]
    top_k_results: list[dict[str, Any]]
    index_files: list[str]
    model: str
    model_load_sec: float
    query_encode_sec: float
    similarity_search_sec: float
    historical_score_dependency: bool = False

    def audit_dict(self) -> dict[str, Any]:
        """Return JSON-safe metadata without serializing the query vector."""
        return {
            "modality": self.modality,
            "query_text": self.query_text,
            "score_map": self.score_map,
            "ranked_results": self.ranked_results,
            "top_k_results": self.top_k_results,
            "index_files": self.index_files,
            "model": self.model,
            "model_load_sec": self.model_load_sec,
            "query_encode_sec": self.query_encode_sec,
            "similarity_search_sec": self.similarity_search_sec,
            "historical_score_dependency": False,
            "query_vector_dimension": int(self.query_vector.size),
            "all_scores_finite": all(np.isfinite(item["similarity_score"]) for item in self.ranked_results),
        }


@dataclass
class FreshQueryScorer:
    """Lazily load validated local text encoders and score new questions.

    Model instances are process-local reusable runtime resources.  Loading and
    every question encode/search are timed separately; offline embeddings are
    read but never rebuilt or modified.
    """

    project_root: Path
    config_path: Path | None = None
    device: str | None = None
    top_k: int = 3
    _config: dict[str, Any] = field(init=False, repr=False)
    _models: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _processors: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _load_times: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _load_counts: dict[str, int] = field(default_factory=lambda: {name: 0 for name in MODALITIES}, init=False, repr=False)
    _query_counts: dict[str, int] = field(default_factory=lambda: {name: 0 for name in MODALITIES}, init=False, repr=False)
    _instance_ids: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _index_contracts: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _validated_index_bundles: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        path = self.config_path or self.project_root / "configs/retrieval_mvp.yaml"
        self._config = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _load_visual(self) -> tuple[Any, float]:
        if "visual" in self._models:
            return self._models["visual"], 0.0
        try:
            import pkg_resources  # noqa: F401
        except ModuleNotFoundError:
            import packaging
            import packaging.version  # make pkg_resources.packaging.version explicit
            import types
            shim = types.ModuleType("pkg_resources")
            shim.packaging = packaging
            sys.modules["pkg_resources"] = shim
        import clip
        cfg = self._config["encoders"]["visual"]
        started = time.perf_counter()
        model, _ = clip.load(cfg["model_name"], device=self.device, download_root=str(Path(cfg["cache_path"]).parent))
        model.eval()
        elapsed = time.perf_counter() - started
        self._models["visual"] = model
        self._processors["visual"] = clip
        self._load_times["visual"] = elapsed
        self._load_counts["visual"] += 1
        self._instance_ids["visual"] = id(model)
        return model, elapsed

    def _load_speech(self, video_id: str) -> tuple[Any, float]:
        if "speech" in self._models:
            return self._models["speech"], 0.0
        from sentence_transformers import SentenceTransformer
        metadata = json.loads((self.project_root / "outputs/audio_index" / video_id / "transcript_embedding_index.json").read_text(encoding="utf-8"))
        started = time.perf_counter()
        model = SentenceTransformer(metadata["model_cache_path"], device=self.device, local_files_only=True)
        elapsed = time.perf_counter() - started
        self._models["speech"] = model
        self._load_times["speech"] = elapsed
        self._load_counts["speech"] += 1
        self._instance_ids["speech"] = id(model)
        return model, elapsed

    def _load_acoustic(self, video_id: str) -> tuple[Any, Any, float]:
        if "acoustic" in self._models:
            return self._models["acoustic"], self._processors["acoustic"], 0.0
        from transformers import ClapModel, ClapProcessor
        metadata = json.loads((self.project_root / "outputs/audio_index" / video_id / "acoustic_embedding_index.json").read_text(encoding="utf-8"))
        started = time.perf_counter()
        processor = ClapProcessor.from_pretrained(metadata["model"], revision=metadata.get("model_revision"), local_files_only=True)
        model = ClapModel.from_pretrained(metadata["model"], revision=metadata.get("model_revision"), use_safetensors=True, local_files_only=True).to(self.device).eval()
        elapsed = time.perf_counter() - started
        self._models["acoustic"] = model
        self._processors["acoustic"] = processor
        self._load_times["acoustic"] = elapsed
        self._load_counts["acoustic"] += 1
        self._instance_ids["acoustic"] = id(model)
        return model, processor, elapsed

    def _encode(self, modality: str, question: str, video_id: str) -> tuple[np.ndarray, float, float, str]:
        if modality == "visual":
            model, loaded = self._load_visual()
            clip = self._processors["visual"]
            started = time.perf_counter()
            with torch.inference_mode():
                tokens = clip.tokenize([question]).to(self.device)
                vector = model.encode_text(tokens).float()
                vector = vector / vector.norm(dim=-1, keepdim=True)
                if self.device == "cuda":
                    torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            return vector.cpu().numpy()[0].astype(np.float32), loaded, elapsed, self._config["encoders"]["visual"]["name"]
        if modality == "speech":
            model, loaded = self._load_speech(video_id)
            started = time.perf_counter()
            vector = model.encode([question], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]
            if self.device == "cuda":
                torch.cuda.synchronize()
            return np.asarray(vector, dtype=np.float32), loaded, time.perf_counter() - started, "sentence-transformers/sentence-t5-base"
        if modality == "acoustic":
            model, processor, loaded = self._load_acoustic(video_id)
            started = time.perf_counter()
            with torch.inference_mode():
                inputs = processor(text=[question], return_tensors="pt", padding=True)
                output = model.get_text_features(**{key: value.to(self.device) for key, value in inputs.items()})
                vector = output.pooler_output if hasattr(output, "pooler_output") else output
                vector = vector.float()
                vector = vector / vector.norm(dim=-1, keepdim=True)
                if self.device == "cuda":
                    torch.cuda.synchronize()
            return vector.cpu().numpy()[0].astype(np.float32), loaded, time.perf_counter() - started, "laion/clap-htsat-unfused"
        raise ValueError(f"Unsupported modality: {modality}")

    def preload(self, video_id: str, modalities: Iterable[str] = MODALITIES) -> dict[str, float]:
        """Load requested encoders without encoding a question.

        This exposes process startup cost separately from per-question online
        query encoding.  It does not read a question, score an index, or alter
        retrieval output.
        """
        timings: dict[str, float] = {}
        for modality in dict.fromkeys(modalities):
            if modality == "visual":
                _, elapsed = self._load_visual()
            elif modality == "speech":
                _, elapsed = self._load_speech(video_id)
            elif modality == "acoustic":
                _, _, elapsed = self._load_acoustic(video_id)
            else:
                raise ValueError(f"Unsupported modality: {modality}")
            timings[modality] = elapsed
        return timings

    def score_case(self, case: dict[str, Any], modalities: Iterable[str]) -> dict[str, QueryScoreResult]:
        """Encode one allowlisted raw question and rank required offline indexes."""
        safe = safe_retrieval_case(case)
        results: dict[str, QueryScoreResult] = {}
        for modality in dict.fromkeys(modalities):
            if modality not in MODALITIES:
                continue
            folder = self.project_root / ("outputs/visual_index" if modality == "visual" else "outputs/audio_index") / safe["video_id"]
            if modality == "visual":
                embedding_path = folder / "region_embeddings.npy"
                index_path = folder / "visual_state_regions.json"
                metadata = json.loads(index_path.read_text(encoding="utf-8"))
                rows = metadata["visual_state_regions"]
                id_key = "region_id"
            elif modality == "speech":
                embedding_path = folder / "transcript_embeddings.npy"
                index_path = folder / "transcript_embedding_index.json"
                metadata = json.loads(index_path.read_text(encoding="utf-8"))
                rows = metadata["rows"]
                id_key = "transcript_segment_id"
            else:
                embedding_path = folder / "acoustic_embeddings.npy"
                index_path = folder / "acoustic_embedding_index.json"
                metadata = json.loads(index_path.read_text(encoding="utf-8"))
                rows = metadata["rows"]
                id_key = "acoustic_region_id"
            embeddings = np.load(embedding_path)
            signature = validate_index_bundle_metadata(
                modality=modality,
                video_id=safe["video_id"],
                metadata=metadata,
                rows=rows,
                embeddings=embeddings,
                encoder_config=self._config["encoders"][modality],
                prior_signature=self._index_contracts.get(modality),
            )
            self._index_contracts.setdefault(modality, signature)
            self._validated_index_bundles[f"{safe['video_id']}:{modality}"] = {
                "video_id": safe["video_id"],
                **signature,
                "model_reloaded": False,
            }
            self._query_counts[modality] += 1
            vector, load_sec, encode_sec, model_name = self._encode(modality, safe["question"], safe["video_id"])
            if vector.size != embeddings.shape[1]:
                raise IndexContractError(
                    f"{modality} query/index dimension mismatch for {safe['video_id']}: "
                    f"query={vector.size}, index={embeddings.shape[1]}"
                )
            started = time.perf_counter()
            ranked = rank_index(vector, embeddings, rows, len(rows))
            search_sec = time.perf_counter() - started
            flattened = []
            for item in ranked:
                metadata = item["metadata"]
                record = {
                    "rank": item["rank"], "similarity_score": item["similarity_score"],
                    id_key: str(metadata[id_key]), "start_time": float(metadata["start_time"]),
                    "end_time": float(metadata["end_time"]),
                }
                if modality == "visual":
                    record.update({"representative_keyframe_path": metadata.get("representative_keyframe_path"), "representative_frame_timestamp": metadata.get("representative_frame_timestamp")})
                elif modality == "speech":
                    record.update({"transcript_text": metadata.get("text", ""), "language": metadata.get("language")})
                else:
                    record.update({"low_information_or_silence": metadata.get("low_information_or_silence"), "speech_overlap_ratio": metadata.get("speech_overlap_ratio")})
                flattened.append(record)
            score_map = {item[id_key]: item for item in flattened}
            results[modality] = QueryScoreResult(modality, safe["question"], vector, score_map, flattened, flattened[: self.top_k], [str(embedding_path.relative_to(self.project_root)).replace("\\", "/"), str(index_path.relative_to(self.project_root)).replace("\\", "/")], model_name, load_sec, encode_sec, search_sec)
        return results

    def lifecycle_audit(self) -> dict[str, Any]:
        """Report non-sensitive process-local model reuse instrumentation."""
        return {
            "device": self.device,
            "models": {
                modality: {
                    "load_count": self._load_counts[modality],
                    "queries_served": self._query_counts[modality],
                    "instance_id": self._instance_ids.get(modality),
                    "cold_load_sec": self._load_times.get(modality),
                    "currently_loaded": modality in self._models,
                }
                for modality in MODALITIES
            },
            "persistent_reuse_invariant": all(count <= 1 for count in self._load_counts.values()),
            "index_contracts": getattr(self, "_index_contracts", {}),
            "validated_index_bundles": getattr(self, "_validated_index_bundles", {}),
        }

    def close(self) -> None:
        """Release local encoder resources; no indexes or source files change."""
        self._models.clear()
        self._processors.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
