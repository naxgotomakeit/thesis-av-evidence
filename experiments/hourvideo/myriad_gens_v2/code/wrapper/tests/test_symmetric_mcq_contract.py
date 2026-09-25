from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from gens_baseline.access import AccessViolation
from gens_baseline.cache import load_frozen_cache_candidates
from gens_baseline.config import load_profile, resolve_profile_path, sha256_file, verify_frozen_artifacts
from gens_baseline.contract import cap_gens_frames
from gens_baseline.gens_stage import build_gens_messages, load_instruction_template
from gens_baseline.pipeline import build_access_policy
from gens_baseline.query import build_retrieval_query, load_selector
from gens_baseline.symmetric_mcq import (
    aggregate_option_scores,
    build_option_queries,
    gens_candidates_without_clip_option_signals,
    load_symmetric_template,
)


ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "config/gens_hybrid_symmetric_mcq_cap16_v1.json"
PROFILE_SHA = "e817014952c76d5cf770062a6ca5d2324f5595b560c7baec1f2db40a0ce45b31"
EMBEDDING_TREE_SHA = "0a2d41faca48e52f558860220aad2c14c6452e52e0125a8f7daa8502693910c7"
OPTION_KEYS = ("A", "B", "C", "D", "E")


class SymmetricMCQContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = load_profile(PROFILE_PATH)
        cls.policy, _, _ = build_access_policy(cls.profile)
        cls.records = load_selector(cls.profile, cls.policy)
        cls.template = load_symmetric_template(cls.profile)

    def test_01_profile_templates_and_frozen_artifacts(self) -> None:
        self.assertEqual(sha256_file(PROFILE_PATH), PROFILE_SHA)
        self.assertEqual(self.profile["method"], "gens_hybrid_symmetric_mcq_cap16_v1")
        self.assertEqual(
            self.profile["symmetric_mcq"]["clip_query_template_sha256"],
            "8391f7c5dbd15fb122f270fdbb8d652820cd2d6d6507a6606d3ff0430adb716f",
        )
        self.assertEqual(
            self.profile["gens_full_query"]["template_sha256"],
            "f32b2f1b38241811e461c1c5e6c47f0b5e54abdb9710867e87ec557bfa788985",
        )
        verify_frozen_artifacts(self.profile, full_manifests=False)

    def test_02_five_queries_are_label_free_and_identically_configured(self) -> None:
        queries = build_option_queries(self.records[0], self.template)
        self.assertEqual([item["option_label"] for item in queries], list(OPTION_KEYS))
        for item in queries:
            self.assertEqual(item["query"].count("Question: "), 1)
            self.assertEqual(item["query"].count("Candidate answer: "), 1)
            self.assertNotIn(f"Candidate answer {item['option_label']}", item["query"])
        self.assertEqual(self.profile["symmetric_mcq"]["max_length"], 77)
        self.assertTrue(self.profile["symmetric_mcq"]["truncation"])

    def test_03_same_option_at_A_or_E_tokenizes_identically(self) -> None:
        from transformers import CLIPTokenizerFast

        tokenizer = CLIPTokenizerFast.from_pretrained(
            resolve_profile_path(self.profile, self.profile["clip"]["local_path"]),
            local_files_only=True,
        )
        record = self.records[0]
        moved = dict(record)
        moved["options"] = dict(record["options"])
        moved["options"]["E"] = record["options"]["A"]
        original = build_option_queries(record, self.template)[0]["query"]
        relocated = build_option_queries(moved, self.template)[4]["query"]
        self.assertEqual(original, relocated)
        self.assertEqual(
            tokenizer(original, truncation=True, max_length=77)["input_ids"],
            tokenizer(relocated, truncation=True, max_length=77)["input_ids"],
        )

    def test_04_max_aggregation_and_tie_break_are_exact(self) -> None:
        candidates = [
            {"video_id": "v", "frame_index": 2, "timestamp_sec": 2.0},
            {"video_id": "v", "frame_index": 1, "timestamp_sec": 1.0},
        ]
        scores = {
            "A": [0.5, 0.5],
            "B": [0.5, 0.4],
            "C": [0.2, 0.3],
            "D": [0.1, 0.2],
            "E": [0.0, 0.1],
        }
        ranked = aggregate_option_scores(candidates, scores, 256)
        self.assertEqual([item["frame_index"] for item in ranked], [1, 2])
        self.assertEqual(ranked[1]["clip_winning_option_label"], "A")
        self.assertEqual(ranked[0]["clip_score"], 0.5)

    def test_05_option_permutation_preserves_aggregated_ranking(self) -> None:
        candidates = [
            {"video_id": "v", "frame_index": i, "timestamp_sec": float(i)} for i in range(300)
        ]
        text_scores = {
            "one": [i / 1000 for i in range(300)],
            "two": [(299 - i) / 900 for i in range(300)],
            "three": [0.2] * 300,
            "four": [0.1] * 300,
            "five": [0.0] * 300,
        }
        first_map = dict(zip(OPTION_KEYS, text_scores.values()))
        reversed_map = dict(zip(OPTION_KEYS, reversed(list(text_scores.values()))))
        first = aggregate_option_scores(candidates, first_map, 256)
        second = aggregate_option_scores(candidates, reversed_map, 256)
        self.assertEqual([x["frame_index"] for x in first], [x["frame_index"] for x in second])
        self.assertEqual([x["clip_score"] for x in first], [x["clip_score"] for x in second])

    def test_06_normalized_cosine_and_top256_repeat_exactly(self) -> None:
        image = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        texts = {
            "A": [1.0, 0.0], "B": [0.0, 1.0], "C": [0.0, 0.0],
            "D": [-1.0, 0.0], "E": [0.0, -1.0],
        }
        scores = {label: (image @ np.asarray(vector, dtype=np.float32)).tolist() for label, vector in texts.items()}
        candidates = [
            {"video_id": "v", "frame_index": i, "timestamp_sec": float(i)} for i in range(2)
        ]
        self.assertEqual(
            aggregate_option_scores(candidates, scores),
            aggregate_option_scores(candidates, scores),
        )

    def test_07_embedding_cache_is_12_of_12_hash_shape_order(self) -> None:
        candidates = load_frozen_cache_candidates(
            profile=self.profile,
            video_id=self.records[0]["video_id"],
            duration_sec=float(self.records[0]["video_duration_sec"]),
            policy=self.policy,
        )
        self.assertEqual(len(candidates), 3598)
        root = resolve_profile_path(self.profile, self.profile["cached_image_embeddings"]["root"])
        self.assertEqual(sha256_file(root / "CACHE_SHA256SUMS.txt"), EMBEDDING_TREE_SHA)
        summary = json.loads(
            resolve_profile_path(
                self.profile, self.profile["cached_image_embeddings"]["summary"]
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["video_count"], 12)
        self.assertEqual(summary["embedding_shape_global"], [37561, 768])
        for video in summary["videos"]:
            matrix = np.load(root / "videos" / video["video_id"] / "image_embeddings.float32.npy", mmap_mode="r")
            registry = json.loads((root / "videos" / video["video_id"] / "frame_registry.json").read_text())
            self.assertEqual(matrix.shape, tuple(video["shape"]))
            self.assertEqual(matrix.dtype, np.float32)
            self.assertEqual([row["row_index"] for row in registry], list(range(len(registry))))

    def test_08_gens_receives_full_mcq_without_clip_internal_signals(self) -> None:
        record = self.records[0]
        full_template = resolve_profile_path(
            self.profile, self.profile["gens_full_query"]["template_path"]
        ).read_text(encoding="utf-8")
        full_query = build_retrieval_query(record, full_template)
        candidate = {
            "video_id": record["video_id"], "frame_index": 0, "timestamp_sec": 0.0,
            "extracted_frame_path": "/allowed/frame.jpg", "clip_option_scores": {k: 0.1 for k in OPTION_KEYS},
            "clip_winning_option_label": "A", "clip_score": 0.1, "clip_rank": 1,
        }
        clean = gens_candidates_without_clip_option_signals([candidate])
        self.assertFalse(any(key in clean[0] for key in ("clip_option_scores", "clip_winning_option_label")))
        self.assertEqual(clean[0]["clip_score"], 0.1)
        self.assertEqual(clean[0]["clip_rank"], 1)
        messages = build_gens_messages(clean, full_query, load_instruction_template(self.profile), 112)
        self.assertIn(full_query, messages[0]["content"][-1]["text"])
        self.assertTrue(all(key not in json.dumps(messages) for key in ("clip_option_scores", "clip_winning_option_label")))

    def test_09_private_gold_and_api_boundaries(self) -> None:
        with self.assertRaises(AccessViolation):
            self.policy.assert_read_allowed("/myriadfs/home/ucemxna/Scratch/evaluation_private/never")
        self.assertFalse(self.profile["outputs"]["api_downstream_run"])
        forbidden = {"gold", "gold_label", "answer", "prediction", "trajectory"}
        self.assertTrue(all(not forbidden.intersection(record) for record in self.records))

    def test_10_final_cap_remains_at_most_16_without_padding(self) -> None:
        selected, _ = cap_gens_frames([], 16)
        self.assertEqual(selected, [])
        self.assertEqual(self.profile["final_selection"]["frame_cap"], 16)
        self.assertFalse(self.profile["final_selection"]["pad_to_cap"])
        self.assertIsNone(self.profile["final_selection"]["fallback"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
