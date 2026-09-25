from __future__ import annotations

import hashlib
import inspect
import json
import unittest
from pathlib import Path

import formal_eval300_selector as v1
import formal_eval300_selector_v2 as v2
from gens_baseline.config import load_profile, verify_frozen_artifacts
from gens_baseline.gens_stage import GenSSelector


class V2LongOutputContractTests(unittest.TestCase):
    def test_01_profile_sha_and_only_allowed_diff(self) -> None:
        self.assertEqual(v2.sha256_file(v2.PROFILE_PATH), v2.EXPECTED_PROFILE_SHA)
        changes = v2.validate_profile_diff()
        self.assertIn("gens.generation.max_new_tokens", {item["path"] for item in changes})
        self.assertTrue(all(
            item["path"] == "gens.generation.max_new_tokens"
            or item["path"].startswith("amendment")
            or item["path"].startswith("observability")
            for item in changes
        ))

    def test_02_generation_change_is_exactly_512_to_4096(self) -> None:
        old = load_profile(v2.V1_PROFILE_PATH)
        new = load_profile(v2.PROFILE_PATH)
        self.assertEqual(old["gens"]["generation"], {"do_sample": False, "max_new_tokens": 512})
        self.assertEqual(new["gens"]["generation"], {"do_sample": False, "max_new_tokens": 4096})

    def test_03_all_frozen_artifacts_validate_under_v2(self) -> None:
        profile = load_profile(v2.PROFILE_PATH)
        verify_frozen_artifacts(profile, full_manifests=False)
        self.assertEqual(profile["gens"]["revision"], "6454c5d7b034b57d61c84f4303f43610233eca80")
        self.assertEqual(profile["cached_image_embeddings"]["global_tree_sha256"], v1.EXPECTED_EMBEDDING_TREE_SHA)
        self.assertEqual(profile["candidate_cache"]["total_tree_sha256"], v1.EXPECTED_FRAME_TREE_SHA)

    def test_04_observed_generate_changes_only_return_container(self) -> None:
        source = inspect.getsource(GenSSelector.select_observed)
        self.assertIn("do_sample=False", source)
        self.assertIn("max_new_tokens=self.max_new_tokens", source)
        self.assertIn("return_dict_in_generate=True", source)
        self.assertNotIn("output_scores=True", source)
        self.assertNotIn("temperature=", source)
        self.assertNotIn("top_p=", source)
        self.assertNotIn("top_k=", source)

    def test_05_v1_raw_response_manifest_is_immutable(self) -> None:
        count, digest = v2.raw_manifest_sha(v2.V1_ROOT)
        self.assertEqual(count, 600)
        self.assertEqual(digest, v2.EXPECTED_V1_RAW_MANIFEST_SHA)

    def test_06_v1_stage_a_is_complete_and_hash_valid_for_300(self) -> None:
        records = []
        for line in (v2.V1_ROOT / "shards/gpu0_150.jsonl").read_text(encoding="utf-8").splitlines():
            records.append(json.loads(line))
        for line in (v2.V1_ROOT / "shards/gpu1_150.jsonl").read_text(encoding="utf-8").splitlines():
            records.append(json.loads(line))
        self.assertEqual(len(records), 300)
        self.assertEqual(len({row["qa_uid"] for row in records}), 300)
        for record in records:
            self.assertTrue(v1.stage_a_complete(v2.V1_ROOT / "questions" / record["qa_uid"], record))

    def test_07_frozen_shards_have_exact_identity(self) -> None:
        expected = {
            "gpu0_150.jsonl": "6248191483a485c04872615f03e2f05b47e18ef5f839f45a0a10bce342375922",
            "gpu1_150.jsonl": "895255e8340917b6208328b9cb97e0edc03768217d41cbe56979c5e21bd7cdea",
            "gpu0_uids.txt": "9a099180677203777ea1cd30e9a2c619f5e9b64023174dc3521229d7b0497747",
            "gpu1_uids.txt": "82e4fafa7b470ffb97688276eb2a48e252f3fbffe764a554b5930aabf2060583",
        }
        for name, digest in expected.items():
            self.assertEqual(v2.sha256_file(v2.V1_ROOT / "shards" / name), digest)


if __name__ == "__main__":
    unittest.main()
