from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .retriever import DenseSemanticBeamBRetriever


EXPERIMENT_ROOT = Path("/home/naxucl/data/HourVideo/experiments/hourvideo_v7_4_variant_c_budgets_v1")
ORIGINAL_RETRIEVER_ROOT = EXPERIMENT_ROOT / "outputs/dense_semantic_beam_b_retriever_v1_20260827T200942Z"
EXPECTED = {
    "retriever.py": "47b8dcfea459f9edacf650e9975043714635c8c3b5fd054f78715535f99cd46e",
    "config.json": "4a20cabe418a611dc1b21c44f6879293484e2e97183bc34c466b43d4a0496ab7",
    "original_manifest": "e23ae0a1f703ac82ccfe756330de772b4d7952112415b5efc0056c071681ca0a",
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def runtime_gate() -> dict[str, Any]:
    here = Path(__file__).resolve().parent
    checks = {
        "overlay_retriever": (_sha(here / "retriever.py"), EXPECTED["retriever.py"]),
        "overlay_config": (_sha(here / "config.json"), EXPECTED["config.json"]),
        "original_retriever_manifest": (_sha(ORIGINAL_RETRIEVER_ROOT / "MANIFEST.sha256"), EXPECTED["original_manifest"]),
    }
    failed = [name for name, (actual, expected) in checks.items() if actual != expected]
    if failed:
        raise RuntimeError(f"Dense Beam-B runtime SHA gate failed: {failed}")
    return {name: {"actual": actual, "expected": expected, "pass": True} for name, (actual, expected) in checks.items()}


class OpenAITextEmbeddingProvider:
    def __init__(self) -> None:
        base = str(os.getenv("EMBEDDING_API_BASE") or "").strip()
        key = str(os.getenv("EMBEDDING_API_KEY") or "").strip()
        if not base or not key:
            raise RuntimeError("EMBEDDING_API_BASE and EMBEDDING_API_KEY are required")
        from openai import OpenAI

        self.client = OpenAI(api_key=key, base_url=base)
        self.calls = 0

    def embed_once(self, query: str) -> tuple[np.ndarray, dict[str, Any]]:
        self.calls += 1
        response = self.client.embeddings.create(
            model="text-embedding-3-large",
            input=[query],
            dimensions=3072,
            encoding_format="float",
            timeout=float(os.getenv("EMBED_TIMEOUT") or "60"),
        )
        if len(response.data) != 1:
            raise RuntimeError("Expected exactly one query embedding")
        usage = response.usage.model_dump() if getattr(response, "usage", None) is not None else {}
        usage.update({"provider_calls": 1, "model": "text-embedding-3-large", "dimensions": 3072})
        return np.asarray(response.data[0].embedding, dtype=np.float32), usage


class LocalSiglipQueryProvider:
    def __init__(self) -> None:
        model_path = Path(str(os.getenv("DENSE_BEAM_SIGLIP_MODEL") or "")).expanduser().resolve()
        if not model_path.is_dir() or not (model_path / "spiece.model").is_file():
            raise RuntimeError(f"Invalid local SigLIP model path: {model_path}")
        self.model_path = model_path
        self._model = None
        self._tokenizer = None
        self.calls = 0

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import SiglipModel, SiglipTokenizer

        self._tokenizer = SiglipTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
        self._model = SiglipModel.from_pretrained(str(self.model_path), local_files_only=True).eval().to("cpu")

    def encode_once(self, query: str) -> np.ndarray:
        self.calls += 1
        self._load()
        import torch

        inputs = self._tokenizer([query], padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        with torch.inference_mode():
            output = self._model.get_text_features(**inputs)
            features = output.pooler_output if hasattr(output, "pooler_output") else output
            features = torch.nn.functional.normalize(features.float(), dim=-1)
        return features.cpu().numpy()[0].astype(np.float32)


_retriever: DenseSemanticBeamBRetriever | None = None


def get_retriever() -> DenseSemanticBeamBRetriever:
    global _retriever
    if _retriever is None:
        runtime_gate()
        here = Path(__file__).resolve().parent
        _retriever = DenseSemanticBeamBRetriever(
            EXPERIMENT_ROOT,
            here,
            OpenAITextEmbeddingProvider(),
            LocalSiglipQueryProvider(),
        )
    return _retriever


def gate_as_json() -> str:
    return json.dumps(runtime_gate(), sort_keys=True)
