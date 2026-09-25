from __future__ import annotations

import hashlib
import ast
import importlib
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class RetrieverError(RuntimeError):
    pass


class AssetIntegrityError(RetrieverError):
    pass


class TextEmbeddingProvider(Protocol):
    def embed_once(self, query: str) -> tuple[np.ndarray, dict[str, Any]]: ...


class SiglipQueryProvider(Protocol):
    def encode_once(self, query: str) -> np.ndarray: ...


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def flat_cosine_scores(vectors: np.ndarray, norms: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Exact float32 expression used by frozen Flat rag_query_embed.query."""
    V = np.asarray(vectors, dtype=np.float32)
    N = np.asarray(norms, dtype=np.float32)
    q = np.asarray(query, dtype=np.float32)
    denom = (N * (np.linalg.norm(q) or 1.0)).astype(np.float32)
    denom = np.where(denom == 0, 1.0, denom)
    return (V @ q) / denom


def flat_top_b_indices(scores: np.ndarray, b: int) -> np.ndarray:
    """Exact argpartition + argsort sequence used by frozen Flat."""
    sim = np.asarray(scores, dtype=np.float32)
    if sim.ndim != 1 or sim.size == 0:
        return np.asarray([], dtype=np.int64)
    k = max(1, min(int(b), int(sim.shape[0])))
    idx = np.argpartition(-sim, k - 1)[:k]
    return idx[np.argsort(-sim[idx])]


@dataclass
class VideoAssets:
    video_uid: str
    coarse_meta: list[dict[str, Any]]
    coarse_vectors: np.ndarray
    coarse_norms: np.ndarray
    medium_meta: list[dict[str, Any]]
    medium_vectors: np.ndarray
    medium_norms: np.ndarray
    flat_doc_ids: list[str]
    flat_vectors: np.ndarray
    flat_norms: np.ndarray
    flat_captions: dict[str, Any]
    frame_ids: np.ndarray
    frame_timestamps: np.ndarray
    frame_vectors: np.ndarray
    duration_sec: float


class DenseSemanticBeamBRetriever:
    def __init__(self, experiment_root: Path, version_root: Path, text_provider: TextEmbeddingProvider, siglip_provider: SiglipQueryProvider):
        self.experiment_root = Path(experiment_root).resolve()
        self.version_root = Path(version_root).resolve()
        self.config = json.loads((self.version_root / "config.json").read_text(encoding="utf-8"))
        self.text_provider = text_provider
        self.siglip_provider = siglip_provider
        self._cache: dict[str, VideoAssets] = {}
        self._load_flat_references()
        self._validate_global_inputs()

    def _hourvideo_root(self) -> Path:
        return self.experiment_root.parent.parent

    def _resolve_exp(self, rel: str) -> Path:
        return (self.experiment_root / rel).resolve()

    def _resolve_hourvideo(self, rel: str) -> Path:
        return (self._hourvideo_root() / rel).resolve()

    def _load_flat_references(self) -> None:
        src = self._resolve_hourvideo(self.config["flat_source_directory"])
        expected = {
            src / "videoseal/tools/visual_tools.py": self.config["flat_visual_tools_sha256"],
            src / "videoseal/utils/RAG/rag_query_embed.py": self.config["flat_rag_query_embed_sha256"],
            src / "videoseal/utils/video/tooling.py": self.config["flat_tooling_sha256"],
            src / "videoseal/utils/video/time.py": self.config["flat_time_sha256"],
            src / "videoseal/prompts/visual_tool_prompts.py": self.config["flat_visual_tool_prompts_sha256"],
        }
        for path, sha in expected.items():
            if not path.is_file() or file_sha256(path) != sha:
                raise AssetIntegrityError(f"Frozen Flat source SHA mismatch: {path}")
        def exact_functions(path: Path, names: set[str], namespace: dict[str, Any]) -> dict[str, Any]:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            future = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "__future__"]
            selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
            if {n.name for n in selected} != names:
                raise AssetIntegrityError(f"Frozen Flat functions missing from {path}: {names - {n.name for n in selected}}")
            mod = ast.fix_missing_locations(ast.Module(body=future + selected, type_ignores=[]))
            exec(compile(mod, str(path), "exec"), namespace)
            return namespace

        tooling_ns = exact_functions(src / "videoseal/utils/video/tooling.py", {"select_time_diverse_windows"}, {"List": list, "Tuple": tuple})
        visual_ns = exact_functions(src / "videoseal/tools/visual_tools.py", {"_parse_doc_id_time_window"}, {})
        time_ns = exact_functions(src / "videoseal/utils/video/time.py", {"sec_to_hhmmss"}, {})
        rag_ns = exact_functions(
            src / "videoseal/utils/RAG/rag_query_embed.py",
            {"_load_index_npy_from_dir", "load_index", "query"},
            {"np": np, "json": json, "Path": Path, "List": list, "Tuple": tuple, "DEFAULT_EMBED_MODEL": "text-embedding-3-large", "embed_texts": None},
        )
        prompt_path = src / "videoseal/prompts/visual_tool_prompts.py"
        spec = importlib.util.spec_from_file_location("frozen_flat_visual_tool_prompts", prompt_path)
        if spec is None or spec.loader is None:
            raise AssetIntegrityError("Cannot load frozen Flat prompt module")
        prompt_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prompt_mod)
        self.flat_select_time_diverse_windows = tooling_ns["select_time_diverse_windows"]
        self.flat_parse_doc_id_time_window = visual_ns["_parse_doc_id_time_window"]
        self.flat_sec_to_hhmmss = time_ns["sec_to_hhmmss"]
        self.flat_load_index = rag_ns["load_index"]
        self.flat_query_namespace = rag_ns
        self.flat_query = rag_ns["query"]
        self.flat_build_summary_prompt = prompt_mod.build_visual_retrieve_summary_prompt

    def _rank_with_frozen_flat(self, doc_ids: list[str], vectors: np.ndarray, norms: np.ndarray, query_vector: np.ndarray, b: int) -> tuple[np.ndarray, np.ndarray]:
        """Execute the SHA-verified frozen Flat query function against an in-memory candidate set."""
        ns = self.flat_query_namespace
        old_load, old_embed = ns["load_index"], ns["embed_texts"]
        ns["load_index"] = lambda _path: (doc_ids, vectors, norms, self.config["text_embedding_model"])
        ns["embed_texts"] = lambda _texts, model=None: [query_vector]
        try:
            ranked = self.flat_query(Path("__in_memory_frozen_candidates__"), "__preembedded_query__", topk=b)
        finally:
            ns["load_index"], ns["embed_texts"] = old_load, old_embed
        position = {doc_id: i for i, doc_id in enumerate(doc_ids)}
        return (
            np.asarray([position[doc_id] for doc_id, _score in ranked], dtype=np.int64),
            np.asarray([score for _doc_id, score in ranked], dtype=np.float32),
        )

    def _validate_global_inputs(self) -> None:
        checks = [
            (self._resolve_exp(self.config["protocol_directory"]) / "MANIFEST.sha256", self.config["protocol_manifest_sha256"]),
            (self._resolve_exp(self.config["amendment_directory"]) / "MANIFEST.sha256", self.config["amendment_manifest_sha256"]),
            (self._resolve_exp(self.config["embedding_directory"]) / "MANIFEST.sha256", self.config["embedding_manifest_sha256"]),
            (self._resolve_hourvideo(self.config["flat_config"]), self.config["flat_config_sha256"]),
        ]
        for path, sha in checks:
            if not path.is_file() or file_sha256(path) != sha:
                raise AssetIntegrityError(f"Frozen input SHA mismatch: {path}")
        amendment = self._resolve_exp(self.config["amendment_directory"])
        if not (amendment / "MANIFEST.sha256").is_file():
            raise AssetIntegrityError("Amendment manifest missing")
        for root in (self._resolve_exp(self.config["protocol_directory"]), self._resolve_exp(self.config["embedding_directory"]), amendment):
            self._verify_manifest(root)
        videos = self.available_videos()
        if len(videos) != 12:
            raise AssetIntegrityError(f"Expected 12 embedding videos, found {len(videos)}")
        flat_root = self._resolve_hourvideo(self.config["flat_semantic_index_root"])
        names = ["semantic_vectors.npy", "semantic_norms.npy", "semantic_doc_ids.txt", "semantic_captions.json", "semantic_meta.json", "semantic_segments.txt"]
        combined = hashlib.sha256()
        for video in videos:
            for name in names:
                path = flat_root / video / name
                if not path.is_file():
                    raise AssetIntegrityError(f"Missing Flat index asset: {path}")
                combined.update(f"{video}/{name}\0{file_sha256(path)}\n".encode())
        if combined.hexdigest() != self.config["flat_semantic_index_12video_aggregate_sha256"]:
            raise AssetIntegrityError("Flat 12-video semantic-index aggregate SHA mismatch")

    @staticmethod
    def _verify_manifest(root: Path) -> None:
        manifest = root / "MANIFEST.sha256"
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sha, rel = line.split(None, 1)
            rel = rel.strip()
            if rel.startswith("./"):
                rel = rel[2:]
            path = root / rel
            if not path.is_file() or file_sha256(path) != sha:
                raise AssetIntegrityError(f"Manifest verification failed: {path}")

    def available_videos(self) -> list[str]:
        root = self._resolve_exp(self.config["embedding_directory"]) / "cases"
        return sorted(p.name for p in root.iterdir() if p.is_dir())

    def load_video(self, video_uid: str, verify_canonical_frames: bool = True) -> VideoAssets:
        if video_uid in self._cache:
            return self._cache[video_uid]
        emb_case = self._resolve_exp(self.config["embedding_directory"]) / "cases" / video_uid
        hierarchy_case = self._resolve_exp(self.config["hierarchy_root"]) / video_uid
        flat_case = self._resolve_hourvideo(self.config["flat_semantic_index_root"]) / video_uid
        siglip_root = self._resolve_hourvideo(self.config["siglip_shard_root"])
        frame_root = self._resolve_hourvideo(self.config["canonical_frame_root"]) / video_uid / "frames_1fps"
        required = [emb_case, hierarchy_case, flat_case, siglip_root / f"{video_uid}.npz", siglip_root / f"{video_uid}.json"]
        if any(not p.exists() for p in required):
            raise AssetIntegrityError(f"Missing assets for {video_uid}")

        def load_level(level: str):
            meta = json.loads((emb_case / f"{level}_nodes.json").read_text(encoding="utf-8"))
            vec = np.load(emb_case / f"{level}_text_embedding_3_large.float32.npy", allow_pickle=False).astype(np.float32)
            if vec.shape != (len(meta), int(self.config["text_embedding_dimensions"])) or not np.isfinite(vec).all():
                raise AssetIntegrityError(f"Invalid {level} vectors for {video_uid}")
            if any(x["model"] != self.config["text_embedding_model"] or x["dimensions"] != self.config["text_embedding_dimensions"] for x in meta):
                raise AssetIntegrityError(f"Invalid {level} metadata for {video_uid}")
            return meta, vec, np.linalg.norm(vec, axis=1).astype(np.float32)

        coarse_meta, coarse_vec, coarse_norm = load_level("coarse")
        medium_meta, medium_vec, medium_norm = load_level("medium")
        coarse_ids = {x["node_id"] for x in coarse_meta}
        if len(coarse_ids) != len(coarse_meta) or any(x["parent_node_id"] not in coarse_ids for x in medium_meta):
            raise AssetIntegrityError(f"Invalid parent mapping for {video_uid}")

        doc_ids, flat_vec, flat_norm, flat_model = self.flat_load_index(flat_case)
        if flat_model != self.config["text_embedding_model"] or flat_vec.dtype != np.float32:
            raise AssetIntegrityError(f"Invalid Flat index model/dtype for {video_uid}")
        captions = json.loads((flat_case / "semantic_captions.json").read_text(encoding="utf-8"))
        if any(doc_id not in captions for doc_id in doc_ids):
            raise AssetIntegrityError(f"Flat caption mapping missing for {video_uid}")

        sig = np.load(siglip_root / f"{video_uid}.npz", allow_pickle=False)
        frame_ids = np.asarray(sig["frame_ids"])
        frame_ts = np.asarray(sig["timestamps_sec"], dtype=np.float32)
        frame_vec = np.asarray(sig["embedding"], dtype=np.float32)
        meta_sig = json.loads((siglip_root / f"{video_uid}.json").read_text(encoding="utf-8"))
        if file_sha256(siglip_root / f"{video_uid}.npz") != meta_sig["sha256"]:
            raise AssetIntegrityError(f"SigLIP shard SHA mismatch for {video_uid}")
        if frame_vec.shape != (len(frame_ids), int(self.config["siglip_dimensions"])) or len(frame_ts) != len(frame_ids):
            raise AssetIntegrityError(f"SigLIP shape mismatch for {video_uid}")
        if not np.isfinite(frame_vec).all() or not np.isfinite(frame_ts).all() or len(set(frame_ids.tolist())) != len(frame_ids):
            raise AssetIntegrityError(f"Invalid SigLIP values/IDs for {video_uid}")
        if np.any(np.diff(frame_ts) < 0):
            raise AssetIntegrityError(f"Non-monotonic frame timestamps for {video_uid}")
        if verify_canonical_frames:
            missing = [fid for fid, ts in zip(frame_ids.tolist(), frame_ts.tolist()) if not (frame_root / f"frame_{int(round(float(ts))):05d}.jpg").is_file()]
            if missing:
                raise AssetIntegrityError(f"Missing canonical frames for {video_uid}: {len(missing)}")
        hierarchy = json.loads((hierarchy_case / "shared_hierarchy.json").read_text(encoding="utf-8"))
        medium_ids = {x["medium_id"] for x in hierarchy["medium_nodes"]}
        if len(medium_ids) != len(hierarchy["medium_nodes"]) or any(x["parent_medium_id"] not in medium_ids for x in hierarchy["fine_nodes"]):
            raise AssetIntegrityError(f"Invalid 15-second Fine lineage for {video_uid}")
        assets = VideoAssets(video_uid, coarse_meta, coarse_vec, coarse_norm, medium_meta, medium_vec, medium_norm,
                             list(doc_ids), np.asarray(flat_vec, dtype=np.float32), np.asarray(flat_norm, dtype=np.float32), captions,
                             frame_ids, frame_ts, frame_vec, float(hierarchy["duration_sec"]))
        self._cache[video_uid] = assets
        return assets

    def retrieve(self, video_uid: str, query: str, b: int, original_question: str = "") -> dict[str, Any]:
        if b not in self.config["allowed_b"]:
            raise RetrieverError(f"Unsupported B={b}")
        if not str(query).strip():
            raise RetrieverError("Empty retrieval query")
        t_total = time.perf_counter()
        assets = self.load_video(video_uid)

        t0 = time.perf_counter()
        q_text, usage = self.text_provider.embed_once(query)
        q_text = np.asarray(q_text, dtype=np.float32)
        if q_text.shape != (self.config["text_embedding_dimensions"],) or not np.isfinite(q_text).all():
            raise RetrieverError("Invalid text query embedding")
        embedding_sec = time.perf_counter() - t0

        t0 = time.perf_counter()
        coarse_idx, coarse_ranked_scores = self._rank_with_frozen_flat(
            [str(x["node_id"]) for x in assets.coarse_meta], assets.coarse_vectors, assets.coarse_norms, q_text, b
        )
        selected_coarse = {assets.coarse_meta[int(i)]["node_id"] for i in coarse_idx}
        coarse_sec = time.perf_counter() - t0

        t0 = time.perf_counter()
        eligible_medium_idx = np.asarray([i for i, m in enumerate(assets.medium_meta) if m["parent_node_id"] in selected_coarse], dtype=np.int64)
        if eligible_medium_idx.size == 0:
            raise RetrieverError("No Medium candidates after Coarse routing")
        local_med_idx, medium_ranked_scores = self._rank_with_frozen_flat(
            [str(assets.medium_meta[int(i)]["node_id"]) for i in eligible_medium_idx],
            assets.medium_vectors[eligible_medium_idx], assets.medium_norms[eligible_medium_idx], q_text, b,
        )
        selected_medium_idx = eligible_medium_idx[local_med_idx]
        selected_medium = [assets.medium_meta[int(i)] for i in selected_medium_idx]
        medium_sec = time.perf_counter() - t0

        t0 = time.perf_counter()
        q_siglip = np.asarray(self.siglip_provider.encode_once(query), dtype=np.float32)
        if q_siglip.shape != (self.config["siglip_dimensions"],) or not np.isfinite(q_siglip).all():
            raise RetrieverError("Invalid SigLIP query embedding")
        qn = float(np.linalg.norm(q_siglip))
        if qn <= 0:
            raise RetrieverError("Zero SigLIP query embedding")
        q_siglip = q_siglip / qn
        siglip_encode_sec = time.perf_counter() - t0

        def center_in_selected(s: float, e: float) -> bool:
            c = 0.5 * (s + e)
            return any(float(m["start_sec"]) <= c < float(m["end_sec"]) or (c == assets.duration_sec and float(m["end_sec"]) == assets.duration_sec) for m in selected_medium)

        t0 = time.perf_counter()
        eligible = []
        scored_frame_indices: set[int] = set()
        for doc_id in assets.flat_doc_ids:
            s, e = self.flat_parse_doc_id_time_window(doc_id)
            if not center_in_selected(float(s), float(e)):
                continue
            frame_idx = np.where((assets.frame_timestamps >= float(s)) & (assets.frame_timestamps < float(e)))[0]
            if frame_idx.size == 0:
                raise RetrieverError(f"Eligible Flat segment has no canonical 1fps frame: {video_uid}/{doc_id}")
            scored_frame_indices.update(int(x) for x in frame_idx.tolist())
            scores = assets.frame_vectors[frame_idx] @ q_siglip
            best_local = int(np.argmax(scores))
            best_global = int(frame_idx[best_local])
            eligible.append({
                "doc_id": doc_id,
                "start_sec": int(s),
                "end_sec": int(e),
                "score": float(scores[best_local]),
                "max_score_frame_timestamp": float(assets.frame_timestamps[best_global]),
                "max_score_frame_id": str(assets.frame_ids[best_global]),
            })
        if not eligible:
            raise RetrieverError("No eligible Flat segments after Medium routing")
        segment_scores = np.asarray([x["score"] for x in eligible], dtype=np.float32)
        ranked_idx, segment_ranked_scores = self._rank_with_frozen_flat(
            [str(i) for i in range(len(eligible))],
            np.eye(len(eligible), dtype=np.float32),
            np.ones(len(eligible), dtype=np.float32),
            segment_scores,
            b,
        )
        ranked = [eligible[int(i)] for i in ranked_idx]
        windows = [(float(x["start_sec"]), float(x["end_sec"])) for x in ranked]
        keep = self.flat_select_time_diverse_windows(windows, k=b, min_gap_sec=float(self.config["min_time_gap_sec"]))
        selected_segments = [ranked[int(i)] for i in keep]
        fine_sec = time.perf_counter() - t0

        evidence = []
        for item in selected_segments:
            meta = assets.flat_captions[item["doc_id"]]
            rec = {
                "start_time": self.flat_sec_to_hhmmss(item["start_sec"]),
                "end_time": self.flat_sec_to_hhmmss(item["end_sec"]),
            }
            cap = meta.get("caption") or meta.get("clip_caption") or meta.get("ocr_text")
            if cap:
                rec["caption"] = str(cap)
            evidence.append(rec)
        prompt = self.flat_build_summary_prompt(query_text=str(query).strip(), user_question=str(original_question or query).strip(), candidates=evidence, max_useful=100)
        audit_segments = [{k: x[k] for k in ("doc_id", "score", "max_score_frame_timestamp", "max_score_frame_id")} for x in eligible]
        telemetry = {
            "video_uid": video_uid,
            "b": b,
            "text_embedding_calls": 1,
            "text_embedding_usage": usage,
            "siglip_query_encode_calls": 1,
            "total_coarse": len(assets.coarse_meta),
            "scored_coarse": len(assets.coarse_meta),
            "selected_coarse": len(coarse_idx),
            "selected_coarse_ids": [assets.coarse_meta[int(i)]["node_id"] for i in coarse_idx],
            "coarse_saturated": len(assets.coarse_meta) <= b,
            "medium_candidates": int(eligible_medium_idx.size),
            "scored_medium": int(eligible_medium_idx.size),
            "selected_medium": int(len(selected_medium_idx)),
            "selected_medium_ids": [assets.medium_meta[int(i)]["node_id"] for i in selected_medium_idx],
            "eligible_flat_segments": len(eligible),
            "unique_1fps_frames_scored": len(scored_frame_indices),
            "fine_candidate_count": len(eligible),
            "returned_segments": len(evidence),
            "candidate_shortfall": len(evidence) < b,
            "segment_max_frames": audit_segments,
            "latency_sec": {
                "text_embedding": embedding_sec,
                "coarse": coarse_sec,
                "medium": medium_sec,
                "siglip_query_encode": siglip_encode_sec,
                "fine_and_flat_postprocess": fine_sec,
                "total": time.perf_counter() - t_total,
            },
        }
        return {"evidence": evidence, "summarizer_prompt": prompt, "telemetry": telemetry}
