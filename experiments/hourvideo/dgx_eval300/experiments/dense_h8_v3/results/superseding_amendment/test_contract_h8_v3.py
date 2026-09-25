from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from retriever import AssetIntegrityError, DenseSemanticBeamBRetriever, RetrieverError, flat_cosine_scores, flat_top_b_indices


HERE = Path('/home/naxucl/data/HourVideo/experiments/hourvideo_v7_4_variant_c_budgets_v1/outputs/dense_semantic_beam_b_h8_budget_overlay_v3_20260831T201510Z/runtime_overlay/videoseal/tools/dense_beam_b')
EXPERIMENT = Path('/home/naxucl/data/HourVideo/experiments/hourvideo_v7_4_variant_c_budgets_v1')


class MockTextProvider:
    def __init__(self, fail: bool = False):
        self.calls = 0
        self.fail = fail

    def embed_once(self, query: str):
        self.calls += 1
        if self.fail:
            raise RuntimeError("mock embedding failure")
        rng = np.random.default_rng(abs(hash(query)) % (2**32))
        return rng.normal(size=3072).astype(np.float32), {"prompt_tokens": 7, "total_tokens": 7, "mock": True}


class MockSiglipProvider:
    def __init__(self, fail: bool = False):
        self.calls = 0
        self.fail = fail

    def encode_once(self, query: str):
        self.calls += 1
        if self.fail:
            raise RuntimeError("mock SigLIP failure")
        rng = np.random.default_rng((abs(hash(query)) + 17) % (2**32))
        q = rng.normal(size=768).astype(np.float32)
        return q / np.linalg.norm(q)


def make_retriever(text=None, siglip=None):
    return DenseSemanticBeamBRetriever(EXPERIMENT, HERE, text or MockTextProvider(), siglip or MockSiglipProvider())


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = MockTextProvider()
        cls.siglip = MockSiglipProvider()
        cls.r = make_retriever(cls.text, cls.siglip)
        cls.videos = cls.r.available_videos()

    def test_01_exactly_12_videos_and_all_assets_load(self):
        self.assertEqual(len(self.videos), 12)
        totals = {"coarse": 0, "medium": 0}
        for video in self.videos:
            a = self.r.load_video(video, verify_canonical_frames=True)
            totals["coarse"] += len(a.coarse_meta)
            totals["medium"] += len(a.medium_meta)
            coarse = {x["node_id"] for x in a.coarse_meta}
            self.assertTrue(all(x["parent_node_id"] in coarse for x in a.medium_meta))
        self.assertEqual(totals, {"coarse": 186, "medium": 838})

    def test_02_flat_cosine_and_topk_exact_parity(self):
        video = self.videos[0]
        a = self.r.load_video(video, verify_canonical_frames=False)
        q = np.linspace(-1, 1, 3072, dtype=np.float32)
        ours = flat_top_b_indices(flat_cosine_scores(a.flat_vectors, a.flat_norms, q), 30)
        ns = self.r.flat_query_namespace
        original = ns["embed_texts"]
        try:
            ns["embed_texts"] = lambda texts, model: [q.tolist()]
            flat_root = self.r._resolve_hourvideo(self.r.config["flat_semantic_index_root"]) / video
            hits = ns["query"](flat_root, "mock", topk=30)
        finally:
            ns["embed_texts"] = original
        self.assertEqual([a.flat_doc_ids[int(i)] for i in ours], [x[0] for x in hits])

    def test_03_one_embedding_and_one_siglip_encoding_per_retrieval(self):
        text, sig = MockTextProvider(), MockSiglipProvider()
        r = make_retriever(text, sig)
        result = r.retrieve(self.videos[0], "open drawer", 8, "question")
        self.assertEqual(text.calls, 1)
        self.assertEqual(sig.calls, 1)
        self.assertEqual(result["telemetry"]["text_embedding_calls"], 1)
        self.assertEqual(result["telemetry"]["siglip_query_encode_calls"], 1)

    def test_04_global_top_b_for_all_b_and_parent_scope(self):
        for b in (8, 15, 30):
            result = self.r.retrieve(self.videos[0], f"query-{b}", b)
            t = result["telemetry"]
            self.assertEqual(t["selected_coarse"], min(b, t["total_coarse"]))
            self.assertEqual(t["selected_medium"], min(b, t["medium_candidates"]))
            a = self.r.load_video(self.videos[0], verify_canonical_frames=False)
            parents = {m["node_id"]: m["parent_node_id"] for m in a.medium_meta}
            self.assertTrue(all(parents[mid] in set(t["selected_coarse_ids"]) for mid in t["selected_medium_ids"]))
            self.assertLessEqual(t["returned_segments"], b)

    def test_05_coarse_saturation(self):
        saturated_video = "41a86310-2cc1-48f9-b5b5-6b495a95fbac"
        r6 = self.r.retrieve(saturated_video, "query", 8)
        self.assertTrue(r6["telemetry"]["coarse_saturated"])
        self.assertEqual(r6["telemetry"]["selected_coarse"], 4)
        for video in self.videos:
            self.assertTrue(self.r.retrieve(video, "query", 30)["telemetry"]["coarse_saturated"])

    def test_06_h30_still_medium_gates_fine(self):
        video = self.videos[0]
        a = self.r.load_video(video, verify_canonical_frames=False)
        result = self.r.retrieve(video, "query", 30)
        t = result["telemetry"]
        self.assertEqual(t["selected_medium"], 30)
        self.assertLess(t["unique_1fps_frames_scored"], len(a.frame_ids))
        self.assertLess(t["eligible_flat_segments"], len(a.flat_doc_ids))

    def test_07_flat_time_diversity_refill_exact(self):
        windows = [(0.0, 16.0), (1.0, 17.0), (32.0, 48.0), (33.0, 49.0)]
        self.assertEqual(self.r.flat_select_time_diverse_windows(windows, 4, 15.0), [0, 2, 1, 3])

    def test_08_shortfall_is_logged(self):
        video = self.videos[0]
        a = copy.copy(self.r.load_video(video, verify_canonical_frames=False))
        a.coarse_meta = a.coarse_meta[:1]; a.coarse_vectors = a.coarse_vectors[:1]; a.coarse_norms = a.coarse_norms[:1]
        cid = a.coarse_meta[0]["node_id"]
        mids = [i for i,m in enumerate(a.medium_meta) if m["parent_node_id"] == cid][:1]
        a.medium_meta = [a.medium_meta[mids[0]]]; a.medium_vectors = a.medium_vectors[mids]; a.medium_norms = a.medium_norms[mids]
        m = a.medium_meta[0]
        docs = []
        for doc in a.flat_doc_ids:
            s,e = self.r.flat_parse_doc_id_time_window(doc); c=(s+e)/2
            if float(m["start_sec"]) <= c < float(m["end_sec"]): docs.append(doc)
        a.flat_doc_ids = docs[:2]
        self.r._cache[video] = a
        try:
            result = self.r.retrieve(video, "shortfall", 8)
            self.assertTrue(result["telemetry"]["candidate_shortfall"])
            self.assertLess(result["telemetry"]["returned_segments"], 6)
        finally:
            self.r._cache.pop(video, None)

    def test_09_no_frame_is_hard_failure(self):
        video = self.videos[0]
        a = copy.copy(self.r.load_video(video, verify_canonical_frames=False))
        a.frame_timestamps = np.asarray([999999.0], dtype=np.float32)
        a.frame_ids = np.asarray(["missing"])
        a.frame_vectors = np.zeros((1,768), dtype=np.float32)
        self.r._cache[video] = a
        try:
            with self.assertRaisesRegex(RetrieverError, "no canonical 1fps frame"):
                self.r.retrieve(video, "query", 8)
        finally:
            self.r._cache.pop(video, None)

    def test_10_provider_failure_is_hard_failure(self):
        r = make_retriever(MockTextProvider(fail=True), MockSiglipProvider())
        with self.assertRaises(RuntimeError):
            r.retrieve(self.videos[0], "query", 8)
        r = make_retriever(MockTextProvider(), MockSiglipProvider(fail=True))
        with self.assertRaises(RuntimeError):
            r.retrieve(self.videos[0], "query", 8)

    def test_11_bad_asset_hash_is_hard_failure(self):
        cfg = json.loads((HERE / "config.json").read_text())
        cfg["flat_config_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config.json").write_text(json.dumps(cfg))
            with self.assertRaises(AssetIntegrityError):
                DenseSemanticBeamBRetriever(EXPERIMENT, root, MockTextProvider(), MockSiglipProvider())

    def test_12_deterministic_outputs_and_no_audit_leak(self):
        video = self.videos[0]
        a = self.r.retrieve(video, "same query", 15, "same question")
        b = self.r.retrieve(video, "same query", 15, "same question")
        self.assertEqual(a["evidence"], b["evidence"])
        self.assertEqual(a["summarizer_prompt"], b["summarizer_prompt"])
        self.assertEqual(a["telemetry"]["segment_max_frames"], b["telemetry"]["segment_max_frames"])
        for item in a["evidence"]:
            self.assertTrue(set(item).issubset({"start_time", "end_time", "caption"}))

    def test_13_complete_telemetry(self):
        t = self.r.retrieve(self.videos[0], "telemetry", 8)["telemetry"]
        required = {"total_coarse","scored_coarse","selected_coarse","coarse_saturated","medium_candidates","scored_medium","selected_medium",
                    "eligible_flat_segments","unique_1fps_frames_scored","fine_candidate_count","returned_segments","segment_max_frames",
                    "text_embedding_usage","latency_sec","candidate_shortfall"}
        self.assertTrue(required.issubset(t))
        self.assertEqual(len(t["segment_max_frames"]), t["eligible_flat_segments"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
