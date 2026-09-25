from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from gens_baseline.access import AccessViolation
from gens_baseline.cache import (
    CANONICAL_RUNTIME_CACHE,
    FORBIDDEN_FIRST_CACHE,
    load_frozen_cache_candidates,
)
from gens_baseline.clip_stage import rank_clip_candidates
from gens_baseline.config import load_profile, resolve_profile_path, sha256_file, verify_frozen_artifacts
from gens_baseline.contract import build_package, cap_gens_frames, validate_package
from gens_baseline.gens_stage import (
    build_gens_messages,
    chronological_gens_candidates,
    estimated_visual_tokens,
    load_instruction_template,
    validate_qwen_interface,
)
from gens_baseline.parser import ParserFailure, map_parsed_frames, parse_gens_response
from gens_baseline.pipeline import build_access_policy
from gens_baseline.query import (
    OPTION_KEYS,
    build_retrieval_query,
    load_selector,
    load_template,
    query_sha256,
    uid_order_sha256,
)
from gens_baseline.sampling import SourceFrame, choose_one_fps_source_frames, integer_second_targets
from smoke_telemetry import canonical_json_bytes, timed_call
from smoke_runner import canonical_output_run_dir


ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "config" / "gens_hybrid_cap16_cache_reuse_v1.json"


def candidate(index: int, timestamp: float | None = None, video_id: str = "video-a") -> dict:
    timestamp = float(index) if timestamp is None else timestamp
    return {
        "video_id": video_id,
        "target_timestamp_sec": int(timestamp),
        "timestamp_sec": timestamp,
        "frame_index": index,
        "source_video_path": f"/allowed/{video_id}.mp4",
        "extracted_frame_path": f"/allowed/frames/{index}.png",
    }


class OfflineContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = load_profile(PROFILE_PATH)
        cls.policy, cls.allowlist_path, cls.output_root = build_access_policy(cls.profile)
        cls.records = load_selector(cls.profile, cls.policy)
        cls.template = load_template(cls.profile)

    def test_01_selector_and_frozen_uid_order_are_300_and_equal(self) -> None:
        self.assertEqual(len(self.records), 300)
        self.assertEqual(len({row["qa_uid"] for row in self.records}), 300)
        self.assertEqual(uid_order_sha256(self.records), self.profile["inputs"]["uid_order_sha256"])
        self.assertEqual(
            self.profile["inputs"]["uid_order_sha256"],
            "6db7b3cc672919a60c4b0c4a84f9f2e039e1673bdeddcfe619fdce92f1cb6fa1",
        )

    def test_02_input_policy_is_positive_allowlist_only(self) -> None:
        self.assertEqual(self.policy.assert_read_allowed(self.profile["inputs"]["selector"]), self.policy.selector_path)
        self.assertEqual(len(self.policy.video_paths), 12)
        with self.assertRaises(AccessViolation):
            self.policy.assert_read_allowed("/myriadfs/home/ucemxna/Scratch/workspace/HourVideo")

    def test_03_private_path_is_rejected_lexically_without_open_or_scan(self) -> None:
        private_string = self.policy.forbidden_prefix + "/never-open-this.jsonl"
        with self.assertRaises(AccessViolation):
            self.policy.assert_read_allowed(private_string)

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            allowed = base / "config" / ".." / "outputs"
            allowed.resolve(strict=False).mkdir()
            canonical_root = (base / "outputs").resolve(strict=False)

            self.assertEqual(canonical_output_run_dir(allowed, allowed), canonical_root)
            self.assertEqual(
                canonical_output_run_dir(base / "outputs" / "child", allowed),
                canonical_root / "child",
            )
            with self.assertRaises(ValueError):
                canonical_output_run_dir(base / "outputs_evil", allowed)
            with self.assertRaises(ValueError):
                canonical_output_run_dir(base / "outputs" / ".." / ".." / "outside", allowed)
            with self.assertRaises(ValueError):
                canonical_output_run_dir(Path("/") / "absolute-escape", allowed)

            outside = base / "outside"
            outside.mkdir()
            escape = canonical_root / "escape"
            escape.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                canonical_output_run_dir(escape / "child", allowed)
            with self.assertRaises(ValueError):
                canonical_output_run_dir("", allowed)

            first = canonical_output_run_dir(base / "outputs" / "child", allowed)
            second = canonical_output_run_dir(base / "outputs" / "child", allowed)
            self.assertEqual(first, second)

    def test_04_query_template_hash_is_frozen(self) -> None:
        self.assertEqual(
            sha256_file(ROOT / "config" / "retrieval_query_template.txt"),
            "f32b2f1b38241811e461c1c5e6c47f0b5e54abdb9710867e87ec557bfa788985",
        )

    def test_05_query_contains_question_and_all_A_to_E_options_but_no_gold(self) -> None:
        row = self.records[0]
        query = build_retrieval_query(row, self.template)
        self.assertIn(row["question"], query)
        for option in OPTION_KEYS:
            self.assertIn(f"{option}. {row['options'][option]}", query)
        lowered = query.lower()
        for forbidden in ("gold_label", "prediction", "trajectory"):
            self.assertNotIn(forbidden, lowered)
        self.assertEqual(len(query_sha256(query)), 64)

    def test_06_integer_second_targets_have_no_duration_off_by_one(self) -> None:
        self.assertEqual(integer_second_targets(3.0), [0, 1, 2])
        self.assertEqual(integer_second_targets(3.000001), [0, 1, 2, 3])
        self.assertEqual(integer_second_targets(0.1), [0])

    def test_07_one_fps_uses_nearest_pts_with_earlier_tie_and_deduplicates(self) -> None:
        frames = [
            SourceFrame.from_seconds(0, "0.2"),
            SourceFrame.from_seconds(1, "0.8"),
            SourceFrame.from_seconds(2, "1.2"),
            SourceFrame.from_seconds(3, "2.8"),
        ]
        selected = choose_one_fps_source_frames(frames, 3.0)
        self.assertEqual(
            [(target, frame.frame_index) for target, frame in selected],
            [(0, 0), (1, 1), (2, 2)],
        )

        exact_tie = choose_one_fps_source_frames(
            [
                SourceFrame.from_seconds(0, "0.0"),
                SourceFrame.from_seconds(1, "1.0"),
                SourceFrame.from_seconds(2, "1.2"),
                SourceFrame.from_seconds(3, "2.8"),
            ],
            "2.9",
        )
        self.assertEqual(exact_tie[-1][0], 2)
        self.assertEqual(exact_tie[-1][1].timestamp_sec, 1.2)

        non_tie = choose_one_fps_source_frames(
            [
                SourceFrame.from_seconds(0, "0.0"),
                SourceFrame.from_seconds(1, "1.0"),
                SourceFrame.from_seconds(2, "1.1"),
                SourceFrame.from_seconds(3, "2.7"),
            ],
            "2.9",
        )
        self.assertEqual(non_tie[-1][0], 2)
        self.assertEqual(non_tie[-1][1].timestamp_sec, 2.7)

        same_timestamp = choose_one_fps_source_frames(
            [
                SourceFrame.from_seconds(0, "0.0"),
                SourceFrame.from_seconds(1, "1.0"),
                SourceFrame.from_seconds(3, "2.0"),
                SourceFrame.from_seconds(8, "2.0"),
            ],
            "2.1",
        )
        self.assertEqual(same_timestamp[-1][1].frame_index, 3)

        repeated = choose_one_fps_source_frames(frames, 3.0)
        self.assertEqual(selected, repeated)

        end_bounded = choose_one_fps_source_frames(
            [
                SourceFrame.from_seconds(0, "0.0"),
                SourceFrame.from_seconds(1, "2.1"),
                SourceFrame.from_seconds(2, "2.3"),
            ],
            "2.2",
        )
        self.assertTrue(all(target < 2.2 for target, _ in end_bounded))
        self.assertTrue(all(frame.timestamp < SourceFrame.from_seconds(0, "2.2").timestamp for _, frame in end_bounded))

    def test_08_clip_checkpoint_is_exact_vit_l14_224(self) -> None:
        path = resolve_profile_path(self.profile, self.profile["clip"]["local_path"])
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        processor = json.loads((path / "preprocessor_config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["architectures"], ["CLIPModel"])
        self.assertEqual(config["vision_config"]["patch_size"], 14)
        self.assertEqual(config["vision_config"]["image_size"], 224)
        self.assertEqual(processor["size"], 224)
        self.assertEqual(processor["crop_size"], 224)

    def test_09_clip_top256_sort_and_tie_break_are_deterministic(self) -> None:
        items = [candidate(30, 3.0), candidate(10, 1.0), candidate(20, 2.0)]
        result = rank_clip_candidates(items, [0.5, 0.5, 0.7], top_k=256)
        self.assertEqual([item["frame_index"] for item in result], [20, 10, 30])
        self.assertEqual([item["clip_rank"] for item in result], [1, 2, 3])
        self.assertEqual(result, rank_clip_candidates(items, [0.5, 0.5, 0.7], top_k=256))

    def test_10_less_than_256_clip_candidates_are_not_padded(self) -> None:
        items = [candidate(index) for index in range(7)]
        result = rank_clip_candidates(items, [float(index) for index in range(7)], top_k=256)
        self.assertEqual(len(result), 7)

    def test_11_cross_video_clip_or_gens_mix_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rank_clip_candidates([candidate(0, video_id="a"), candidate(1, video_id="b")], [1.0, 0.5])
        with self.assertRaises(ValueError):
            chronological_gens_candidates([candidate(0, video_id="a"), candidate(1, video_id="b")])

    def test_12_gens_input_is_chronological_and_explicitly_112(self) -> None:
        items = [candidate(20, 2.0), candidate(10, 1.0)]
        ordered = chronological_gens_candidates(items)
        self.assertEqual([item["frame_index"] for item in ordered], [10, 20])
        messages = build_gens_messages(ordered, "query", load_instruction_template(self.profile), 112)
        content = messages[0]["content"]
        images = [item for item in content if item["type"] == "image"]
        labels = [item["text"] for item in content if item["type"] == "text"][:-1]
        self.assertEqual(labels, ["Frame Number [1]", "Frame Number [2]"])
        self.assertTrue(all(item["resized_height"] == 112 and item["resized_width"] == 112 for item in images))

    def test_13_256_frames_fit_frozen_structural_visual_token_budget(self) -> None:
        self.assertEqual(estimated_visual_tokens(256, 112), 4096)
        self.assertLess(estimated_visual_tokens(256, 112), self.profile["gens"]["context_limit_tokens"])

    def test_14_strict_parser_maps_frames_and_inclusive_spans(self) -> None:
        chronological = []
        for index in range(6):
            item = candidate(index, float(index))
            item.update({"clip_rank": index + 1, "clip_score": 1.0 - index / 10.0})
            chronological.append(item)
        parsed = parse_gens_response('{"2": 5, "4-5": 3}', 6)
        self.assertEqual([item.gens_input_index for item in parsed], [2, 4, 5])
        mapped = map_parsed_frames(parsed, chronological)
        self.assertEqual([item["frame_index"] for item in mapped], [1, 3, 4])
        self.assertEqual([item["gens_span"] for item in mapped], ["2", "4-5", "4-5"])

    def test_15_parser_rejects_invalid_overlap_range_duplicate_and_fences(self) -> None:
        invalid = [
            "not json",
            '{"1-3": 5, "3-4": 4}',
            '{"0": 5}',
            '{"5": 4}',
            '{"2": 4, "2": 3}',
            '```json\n{"1": 5}\n```',
            '{"1": 5.0}',
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ParserFailure):
                parse_gens_response(value, 4)

    def test_16_cap16_uses_at_most_actual_unique_frames_without_padding(self) -> None:
        mapped = []
        for index in range(5):
            item = candidate(index, float(4 - index))
            item.update(
                {
                    "clip_rank": index + 1,
                    "clip_score": 0.9 - index / 10,
                    "gens_input_index": index + 1,
                    "gens_relevance_score": 5 - index,
                    "gens_span": str(index + 1),
                    "gens_response_order": index,
                }
            )
            mapped.append(item)
        selected, relevance = cap_gens_frames(mapped)
        self.assertEqual(len(selected), 5)
        self.assertEqual(len(relevance), 5)
        self.assertEqual(
            [(item["timestamp_sec"], item["frame_index"]) for item in selected],
            sorted((item["timestamp_sec"], item["frame_index"]) for item in selected),
        )
        self.assertEqual(cap_gens_frames([]), ([], []))

    def test_17_provenance_and_shared_downstream_contract_are_compatible(self) -> None:
        provenance = {
            "profile": "gens_hybrid_cap16",
            "profile_sha256": "0" * 64,
            "query_template_sha256": "1" * 64,
            "query_sha256": "2" * 64,
            "clip": {},
            "gens": {},
            "candidate_pool": [],
            "clip_candidates": [],
            "gens_raw_response": "{}",
            "gens_parser": {"status": "ok"},
            "gens_relevance_order": [],
        }
        package = build_package(
            profile=self.profile,
            qa_uid="uid",
            video_id="vid",
            status="gens_no_valid_frames",
            error={"type": "EmptyGenSSelection"},
            selected_frames=[],
            latencies={},
            provenance=provenance,
        )
        for method in ("uniform_16", "ours_cap16", "gens_hybrid_cap16"):
            shared = copy.deepcopy(package)
            shared["method"] = method
            validate_package(shared, allow_shared_methods=True)

    def test_18_repeated_pure_pipeline_steps_are_identical(self) -> None:
        frames = [SourceFrame.from_seconds(i, str(i * 0.51)) for i in range(20)]
        first_candidates = choose_one_fps_source_frames(frames, 8.2)
        second_candidates = choose_one_fps_source_frames(frames, 8.2)
        self.assertEqual(first_candidates, second_candidates)
        items = [candidate(frame.frame_index, frame.timestamp_sec) for _, frame in first_candidates]
        scores = [0.25] * len(items)
        self.assertEqual(
            rank_clip_candidates(items, scores), rank_clip_candidates(items, scores)
        )

        fixture_candidates = [candidate(2, 2.0), candidate(0, 0.0), candidate(1, 1.0)]
        fixture_scores = [0.7, 0.8, 0.7]
        direct_clip = rank_clip_candidates(fixture_candidates, fixture_scores, 256)
        direct_chronological = chronological_gens_candidates(direct_clip)
        direct_parsed = parse_gens_response('{"1": 5, "2-3": 3}', 3)
        direct_mapped = map_parsed_frames(direct_parsed, direct_chronological)
        direct_final, direct_relevance = cap_gens_frames(direct_mapped, 16)

        observed_timings: dict[str, float] = {}
        observed_clip = timed_call(
            observed_timings,
            "clip",
            rank_clip_candidates,
            fixture_candidates,
            fixture_scores,
            256,
        )
        observed_chronological = timed_call(
            observed_timings, "chronological", chronological_gens_candidates, observed_clip
        )
        observed_parsed = timed_call(
            observed_timings, "parse", parse_gens_response, '{"1": 5, "2-3": 3}', 3
        )
        observed_mapped = timed_call(
            observed_timings,
            "map",
            map_parsed_frames,
            observed_parsed,
            observed_chronological,
        )
        observed_final, observed_relevance = timed_call(
            observed_timings, "cap", cap_gens_frames, observed_mapped, 16
        )
        direct_bytes = canonical_json_bytes(
            {
                "clip": direct_clip,
                "parsed": [item.to_dict() for item in direct_parsed],
                "final": direct_final,
                "relevance": direct_relevance,
            }
        )
        observed_bytes = canonical_json_bytes(
            {
                "clip": observed_clip,
                "parsed": [item.to_dict() for item in observed_parsed],
                "final": observed_final,
                "relevance": observed_relevance,
            }
        )
        self.assertEqual(direct_bytes, observed_bytes)
        self.assertTrue(all(value >= 0 for value in observed_timings.values()))

    def test_19_official_source_commit_and_worktree_are_unchanged(self) -> None:
        source = ROOT / "source" / "GenS"
        commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)
        self.assertEqual(commit, self.profile["source"]["commit"])
        self.assertEqual(status, "")
        verify_frozen_artifacts(self.profile, full_manifests=False)

    def test_20_qwen25vl_native_interface_and_processor_are_available_cpu_only(self) -> None:
        result = validate_qwen_interface(
            resolve_profile_path(self.profile, self.profile["gens"]["local_path"])
        )
        self.assertEqual(result["model_type"], "qwen2_5_vl")
        self.assertEqual(result["architecture"], "Qwen2_5_VLForConditionalGeneration")
        self.assertEqual(result["processor"], "Qwen2_5_VLProcessor")

    def test_21_canonical_clip_checkpoint_disclosure_is_exact(self) -> None:
        self.assertEqual(self.profile["clip"]["model_id"], "openai/clip-vit-large-patch14")
        self.assertIn("does not disclose the exact checkpoint identifier", self.profile["implementation_disclosure"])
        self.assertNotIn("336", json.dumps(self.profile["clip"]))

    def test_22_runtime_cache_manifest_is_12_of_12_and_matches_every_file_hash(self) -> None:
        cache = self.profile["candidate_cache"]
        manifest_path = resolve_profile_path(self.profile, cache["manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["video_count"], 12)
        self.assertEqual(manifest["total_image_count"], 37561)
        self.assertTrue(manifest["structural_quality_pass"])
        self.assertEqual(manifest["cache_root"], str(CANONICAL_RUNTIME_CACHE))
        self.assertEqual(manifest["forbidden_cache_root"], str(FORBIDDEN_FIRST_CACHE))

        global_digest = hashlib.sha256()
        for video in manifest["videos"]:
            self.assertTrue(video["structural_quality_pass"])
            self.assertEqual(video["missing_numbers"], [])
            self.assertEqual(video["duplicate_numbers"], [])
            self.assertEqual(video["zero_byte_files"], [])
            self.assertEqual(video["corrupt_files"], [])
            self.assertEqual(video["out_of_bounds_files"], [])
            self.assertEqual(video["nonincreasing_timestamp_pairs"], 0)
            self.assertEqual(len(video["frames"]), video["frame_count"])
            video_digest = hashlib.sha256()
            for index, frame in enumerate(video["frames"]):
                self.assertEqual(frame["frame_number"], index)
                self.assertEqual(frame["ordering_index"], index)
                self.assertEqual(frame["timestamp_sec"], float(index))
                path = CANONICAL_RUNTIME_CACHE / frame["relative_path"]
                self.assertEqual(path.stat().st_size, frame["size_bytes"])
                self.assertEqual(sha256_file(path), frame["sha256"])
                line = f"{frame['sha256']}  {frame['relative_path']}\n".encode()
                video_digest.update(line)
                global_digest.update(line)
            self.assertEqual(video_digest.hexdigest(), video["tree_sha256"])
        self.assertEqual(global_digest.hexdigest(), manifest["total_tree_sha256"])
        self.assertEqual(global_digest.hexdigest(), cache["total_tree_sha256"])

    def test_23_cache_anomalies_are_source_verified_and_not_corruption(self) -> None:
        cache = self.profile["candidate_cache"]
        verification_path = resolve_profile_path(self.profile, cache["source_verification"])
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        self.assertTrue(verification["source_verification_pass"])
        self.assertEqual(verification["status"], "pass")
        self.assertEqual(verification["private_reads"], 0)
        self.assertEqual(verification["video_count"], 3)
        self.assertTrue(
            all(
                interval["source_verification_pass"]
                for video in verification["videos"]
                for interval in video["intervals"]
            )
        )

    def test_24_loader_uses_only_runtime_cache_without_video_decode_or_fallback(self) -> None:
        row = self.records[0]
        candidates = load_frozen_cache_candidates(
            profile=self.profile,
            video_id=row["video_id"],
            duration_sec=float(row["video_duration_sec"]),
            policy=self.policy,
        )
        self.assertEqual(len(candidates), 3598)
        self.assertEqual(candidates[0]["timestamp_sec"], 0.0)
        self.assertEqual(candidates[-1]["timestamp_sec"], 3597.0)
        self.assertTrue(
            all(
                Path(item["extracted_frame_path"]).is_relative_to(CANONICAL_RUNTIME_CACHE)
                and not Path(item["extracted_frame_path"]).is_relative_to(FORBIDDEN_FIRST_CACHE)
                and item["source_video_path"] is None
                and item["candidate_source"] == "runtime_frames_1fps_cache_reuse_v1"
                for item in candidates
            )
        )
        source = (ROOT / "wrapper" / "gens_baseline" / "cache.py").read_text(encoding="utf-8")
        for forbidden_decoder in ("import av", "import decord", "subprocess", "ffmpeg"):
            self.assertNotIn(forbidden_decoder, source.lower())
        with self.assertRaises(AccessViolation):
            self.policy.assert_read_allowed(FORBIDDEN_FIRST_CACHE / "never-read.jpg")

    def test_25_profile_supersession_and_frozen_components_are_explicit(self) -> None:
        supersession = json.loads(
            (ROOT / "manifests" / "profile_supersession_cache_reuse_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            supersession["old_profile"]["status"], "superseded_before_live_inference"
        )
        self.assertFalse(supersession["formal_inference_completed_under_old_profile"])
        self.assertEqual(
            sha256_file(ROOT / supersession["old_profile"]["path"]),
            supersession["old_profile"]["sha256"],
        )
        self.assertEqual(
            sha256_file(ROOT / supersession["new_profile"]["path"]),
            supersession["new_profile"]["sha256"],
        )
        verify_frozen_artifacts(self.profile, full_manifests=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
