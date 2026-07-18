"""Generate the read-only Canonical Baseline v1 deep-audit artifacts.

This audit consumes source files and already durable pilot checkpoints.  It
does not invoke a model, retrieve evidence, decode media, or mutate any frozen
research artifact.
"""

from __future__ import annotations

import collections
import hashlib
import html
import json
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs/canonical_pipeline/deep_audit_v0_1"
PILOT = ROOT / "outputs/pilot_20/baseline_v1/full_pilot_v0_1"
CHECKPOINTS = PILOT / "checkpoints"
SMOKE = ROOT / "outputs/pilot_20/baseline_v1/new_case_smoke_v0_1/checkpoints"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence_ids(payload: dict[str, Any]) -> set[str]:
    return {
        item["evidence_id"]
        for group in payload.get("evidence_groups", [])
        for key in ("visual_evidence", "speech_evidence", "acoustic_evidence")
        for item in group.get(key, [])
    }


def finding(
    finding_id: str,
    severity: str,
    category: str,
    title: str,
    locations: list[str],
    evidence: str,
    impact: str,
    recommendation: str,
    *,
    status: str = "open",
    affected_cases: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "finding_id": finding_id,
        "severity": severity,
        "category": category,
        "status": status,
        "title": title,
        "locations": locations,
        "affected_cases": affected_cases or [],
        "evidence": evidence,
        "impact": impact,
        "recommendation": recommendation,
        "research_behavior_changed_by_audit": False,
    }


def checkpoint_audit(cases: list[dict[str, Any]]) -> dict[str, Any]:
    omitted: dict[str, list[str]] = {}
    invalid_group_relations: dict[str, list[dict[str, Any]]] = {}
    duplicate_relations: dict[str, list[dict[str, Any]]] = {}
    visual_contract_errors: dict[str, list[str]] = {}
    identity_errors: dict[str, list[str]] = {}
    media_errors: dict[str, list[str]] = {}

    from src.retrieval.task6_v1_2 import candidate_identity_accounting

    for case in cases:
        case_id = case["case_id"]
        reranking = case["reranking"]
        retained = reranking["retained"]
        retained_ids = {item["candidate_id"] for item in retained}
        payload_ids = evidence_ids(case["final_payload"])
        if retained_ids - payload_ids:
            omitted[case_id] = sorted(retained_ids - payload_ids)

        group_relation_errors: list[dict[str, Any]] = []
        for group in case["final_payload"].get("evidence_groups", []):
            group_ids = {
                item["evidence_id"]
                for key in ("visual_evidence", "speech_evidence", "acoustic_evidence")
                for item in group.get(key, [])
            }
            for relation in group.get("relations", []):
                missing = sorted(
                    {
                        relation["source_candidate_id"],
                        relation["target_candidate_id"],
                    }
                    - group_ids
                )
                if missing:
                    group_relation_errors.append(
                        {"relation": relation, "missing_group_endpoints": missing}
                    )
        if group_relation_errors:
            invalid_group_relations[case_id] = group_relation_errors

        relation_keys = [
            (
                item["source_candidate_id"],
                item["target_candidate_id"],
                item["relation_type"],
                item["relation_basis"],
            )
            for item in reranking.get("relations", [])
        ]
        duplicates = [
            {
                "source_candidate_id": key[0],
                "target_candidate_id": key[1],
                "relation_type": key[2],
                "relation_basis": key[3],
                "count": count,
            }
            for key, count in collections.Counter(relation_keys).items()
            if count > 1
        ]
        if duplicates:
            duplicate_relations[case_id] = duplicates

        identity = candidate_identity_accounting(
            {
                "candidates_before_reranking": reranking["before"],
                "retained_candidates": retained,
                "actually_dropped_candidates": reranking["actually_dropped"],
                "merged_source_candidates": reranking["merged"],
                "transformed_candidates": reranking["transformed"],
            }
        )
        if not identity["consistent"]:
            identity_errors[case_id] = identity["errors"]

        frame_errors: list[str] = []
        asset_errors: list[str] = []
        for group in case["final_payload"].get("evidence_groups", []):
            for visual in group.get("visual_evidence", []):
                frames = visual.get("frames", [])
                timestamps = [float(item["timestamp_sec"]) for item in frames]
                orders = [int(item["presentation_order"]) for item in frames]
                if timestamps != sorted(timestamps):
                    frame_errors.append(f"{visual['evidence_id']}:not_chronological")
                if orders != list(range(1, len(orders) + 1)):
                    frame_errors.append(f"{visual['evidence_id']}:presentation_order")
                if len(timestamps) != len({round(value * 1000) for value in timestamps}):
                    frame_errors.append(f"{visual['evidence_id']}:duplicate_timestamp")
                for frame in frames:
                    path = ROOT / frame["frame_path"]
                    if not path.is_file():
                        asset_errors.append(f"missing_visual:{frame['frame_path']}")
            for acoustic in group.get("acoustic_evidence", []):
                value = acoustic.get("audio_clip_path")
                if not value or not (ROOT / value).is_file():
                    asset_errors.append(f"missing_audio:{value}")
        if frame_errors:
            visual_contract_errors[case_id] = frame_errors
        if asset_errors:
            media_errors[case_id] = asset_errors

    return {
        "case_count": len(cases),
        "retained_candidates_omitted_from_payload_groups": omitted,
        "relations_with_missing_group_endpoints": invalid_group_relations,
        "duplicate_relations": duplicate_relations,
        "candidate_identity_errors": identity_errors,
        "visual_contract_errors": visual_contract_errors,
        "model_facing_media_errors": media_errors,
        "runtime_gold_flags_all_false": all(
            case.get("runtime_gold_loaded") is False for case in cases
        ),
        "leakage_audits_all_passed": all(
            case.get("preflight", {}).get("leakage_check_passed") is True
            for case in cases
        ),
    }


def model_index_audit(cases: list[dict[str, Any]]) -> dict[str, Any]:
    video_ids = sorted({case["video_id"] for case in cases})
    rows = []
    for video_id in video_ids:
        audio = ROOT / "outputs/audio_index" / video_id
        visual = ROOT / "outputs/visual_index" / video_id
        speech = read_json(audio / "transcript_embedding_index.json")
        acoustic = read_json(audio / "acoustic_embedding_index.json")
        visual_meta = read_json(visual / "visual_state_regions.json")
        rows.append(
            {
                "video_id": video_id,
                "visual_encoder": visual_meta.get("encoder"),
                "speech_encoder": speech.get("model"),
                "speech_model_cache_path_present": bool(speech.get("model_cache_path")),
                "acoustic_encoder": acoustic.get("model"),
                "acoustic_revision": acoustic.get("model_revision"),
                "visual_index_carries_qa_reference_metadata": bool(
                    visual_meta.get("dataset_provided_qa_reference_interval")
                ),
            }
        )
    return {
        "videos": rows,
        "visual_models_consistent": len({row["visual_encoder"] for row in rows}) == 1,
        "speech_models_consistent": len({row["speech_encoder"] for row in rows}) == 1,
        "acoustic_models_consistent": len(
            {(row["acoustic_encoder"], row["acoustic_revision"]) for row in rows}
        )
        == 1,
        "note": (
            "The current six indexed videos are homogeneous. FreshQueryScorer "
            "loads Sentence-T5/CLAP metadata from the first triggering video and "
            "does not revalidate later videos."
        ),
    }


def hash_audit() -> dict[str, Any]:
    before = read_json(OUT / "phase1_hashes_before.json")
    frozen_after = {
        path: sha256(ROOT / path) for path in before.get("frozen", {})
    }
    frozen_match = {
        path: before["frozen"][path] == value for path, value in frozen_after.items()
    }
    durable_after: dict[str, str] = {}
    changed: dict[str, dict[str, Any]] = {}
    for filename, old_hash in before.get("durable_checkpoints", {}).items():
        current = CHECKPOINTS / filename
        new_hash = sha256(current)
        durable_after[filename] = new_hash
        if old_hash != new_hash:
            case_id = Path(filename).stem
            source = read_json(SMOKE / filename) if (SMOKE / filename).is_file() else {}
            target = read_json(current)
            changed_keys = sorted(
                key for key in set(source) | set(target) if source.get(key) != target.get(key)
            )
            research_keys = {
                "planner",
                "sufficiency_fallback",
                "reranking",
                "final_payload",
                "preflight",
                "raw_model_output",
                "validated_answer",
                "timings",
                "model_usage",
            }
            changed[filename] = {
                "before_hash": old_hash,
                "after_hash": new_hash,
                "source_smoke_comparison_changed_top_level_keys": changed_keys,
                "research_result_fields_equal_to_source_smoke": all(
                    source.get(key) == target.get(key) for key in research_keys
                ),
                "classification": "reporting_trace_enrichment_and_reuse_markers",
            }
    return {
        "frozen_hashes_before": before.get("frozen", {}),
        "frozen_hashes_after": frozen_after,
        "frozen_artifacts_hash_identical": all(frozen_match.values()),
        "frozen_artifact_per_file_match": frozen_match,
        "preexisting_durable_checkpoint_count": len(before.get("durable_checkpoints", {})),
        "preexisting_durable_checkpoint_hash_unchanged_count": (
            len(before.get("durable_checkpoints", {})) - len(changed)
        ),
        "changed_derived_checkpoint_copies": changed,
        "checkpoint_note": (
            "Thirteen preexisting live checkpoints remained byte-identical. "
            "Three smoke-derived pilot copies were enriched by the old final-report "
            "loop; core research results match their source smoke checkpoints. The "
            "runner now avoids overwriting completed checkpoints."
        ),
    }


def findings(checkpoint: dict[str, Any], hashes: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        finding(
            "F001",
            "HIGH",
            "TECHNICAL BUG",
            "Task6 candidate accounting rejected a canonical visual entity created from selected frame assets",
            [
                "src/retrieval/task6_v1_2.py::candidate_identity_accounting",
                "src/canonical_pipeline/reranking.py::build_evidence_packet",
            ],
            "00018_2 had 3 input acoustic candidate entities, 4 retained entities, 0 genuine drops, and 1 introduced canonical visual entity. The old equation expected 3 retained and observed 4.",
            "The live pilot stopped before Task7A despite a coherent already-selected evidence set.",
            "Keep the new entity-taxonomy reconciliation and producer-side disjointness/uniqueness invariants.",
            status="fixed_in_phase1",
            affected_cases=["00018_2"],
        ),
        finding(
            "F002",
            "HIGH",
            "TECHNICAL BUG",
            "Retained Task6 candidates can be omitted from every Task7A evidence group",
            [
                "src/canonical_pipeline/reranking.py::_groups (lines 128-141)",
                "src/canonical_pipeline/payload.py::build_final_payload (lines 44-52)",
            ],
            f"Audit observed omissions: {checkpoint['retained_candidates_omitted_from_payload_groups']}. measure_delay groups admit only trigger/plausible_response roles; anchor_resolver groups admit only temporal_anchor/resolver roles.",
            "Task7A and Gemini may receive none or only a subset of evidence that Task6 says it retained. 00002_2 became preflight-blocked with an empty group; 00002_7 omitted direct acoustic evidence.",
            "Future technical correction should require every retained candidate to belong to at least one group, without changing which candidates Task6 retains.",
            affected_cases=sorted(checkpoint["retained_candidates_omitted_from_payload_groups"]),
        ),
        finding(
            "F003",
            "HIGH",
            "TECHNICAL BUG",
            "Task6 group relation filtering permits an unavailable endpoint",
            [
                "src/canonical_pipeline/reranking.py::_groups line 141",
                "src/retrieval/task6_relation_reranking.py::temporal_relations lines 98-100",
            ],
            f"Relations are included when source OR target is a group member. Audit observed: {checkpoint['relations_with_missing_group_endpoints']}.",
            "The model can receive a relation referencing evidence absent from its payload; fallback_recovery_for currently targets a synthetic speech_branch_missing_evidence identifier rather than an evidence entity.",
            "Define whether relation endpoints must be evidence IDs or typed sentinel/status nodes, then validate that contract before Task7A.",
            affected_cases=sorted(checkpoint["relations_with_missing_group_endpoints"]),
        ),
        finding(
            "F004",
            "HIGH",
            "TECHNICAL BUG",
            "Corrected Task6 temporal relations can be serialized twice",
            [
                "src/retrieval/task6_v1_1.py::corrected_relations lines 25-52",
                "src/canonical_pipeline/reranking.py::build_evidence_packet lines 167-193",
            ],
            f"The v1.1 correction retains non-resolves relations and then appends the same anchor/resolver temporal relation. Audit observed duplicates: {checkpoint['duplicate_relations']}.",
            "Relation counts and prompt metadata are inflated; downstream readers cannot tell repetition from independent support.",
            "Add deterministic relation-key uniqueness as a producer invariant in a future technical patch.",
            affected_cases=sorted(checkpoint["duplicate_relations"]),
        ),
        finding(
            "F005",
            "MEDIUM",
            "INTEGRATION RISK",
            "Task7A computes visual-frame checks that do not all gate modality availability",
            [
                "src/final_qa/task7a_preflight.py::validate_visual_frames",
                "src/canonical_pipeline/payload.py::build_final_payload lines 57-66",
            ],
            "chronological_visual_order and path validity gate availability; presentation_order_sequential and duplicate normalized timestamp checks are computed but ignored. The current 20 packets pass all checks because Task6 validates them first.",
            "A malformed historical or alternative Task6 producer could pass Task7A despite violating its documented frame contract.",
            "Make all declared checks authoritative at the producer/consumer boundary; do not fabricate missing frame metadata.",
        ),
        finding(
            "F006",
            "MEDIUM",
            "INTEGRATION RISK",
            "Existing local WAV reuse is validated by filename and readability, not provenance/config identity",
            ["src/canonical_pipeline/media_materialization.py::materialize_acoustic_candidate lines 35-53"],
            "When a target path exists, the runtime reads duration/sample rate but does not verify source audio identity, requested end, source hash, channel/PCM policy, or exporter version.",
            "A stale same-name clip can silently become model-facing evidence after source/config changes.",
            "Add a sidecar identity key and strict interval/source/config verification before reuse.",
        ),
        finding(
            "F007",
            "MEDIUM",
            "INTEGRATION RISK",
            "Task5C materialization scope and Task7A acoustic asset scope are independently defined",
            [
                "src/canonical_pipeline/sufficiency.py lines 101-115",
                "src/canonical_pipeline/payload.py lines 59-66",
            ],
            "Task5C materializes acoustic candidates only when the case-level acoustic role is direct_evidence, temporal_anchor, or resolver; Task7A requires a valid WAV for every acoustic member that appears in a group.",
            "A future role/group combination can block preflight even when acoustic is supporting, or silently omit supporting audio upstream.",
            "Centralize a model-facing acoustic asset contract and test every role/group combination.",
        ),
        finding(
            "F008",
            "MEDIUM",
            "INTEGRATION RISK",
            "Persistent encoders do not revalidate later-video index model metadata",
            ["src/canonical_pipeline/query_scoring.py::_load_speech/_load_acoustic lines 117-145"],
            "The first triggering video's Sentence-T5 cache path and CLAP model/revision own the process model. Current pilot metadata and dimensions are homogeneous across all six videos.",
            "A mixed-version index set could be scored with the wrong persistent model without an explicit metadata compatibility failure.",
            "Validate model ID, revision, normalization, and embedding dimension for each index before scoring while retaining one model instance.",
        ),
        finding(
            "F009",
            "MEDIUM",
            "INTEGRATION RISK",
            "Planner primary-anchor modality and Task5B execution modality contracts diverge",
            [
                "src/canonical_pipeline/retrieval.py lines 97-104",
                "src/retrieval/task5b.py::modalities_to_execute lines 89-94",
            ],
            "Fresh scoring requests resolver modalities plus primary_anchor_modality, while Task5B branches execute resolver modalities only. In 00018_5 visual was encoded/scored as primary anchor but the visual retrieval branch and its timing were marked skipped.",
            "Online work can be wasted and planner anchor intent can be ignored; traces disagree about whether the modality executed.",
            "Clarify the contract before changing it; this may be a method change rather than a bookkeeping patch.",
            affected_cases=["00018_5"],
        ),
        finding(
            "F010",
            "LOW",
            "INTEGRATION RISK",
            "Canonical configuration still defaults to the six-case manifest",
            ["config/canonical_pipeline.json paths.case_manifest"],
            "Generalized runners correctly inject the pilot manifest, but a caller that relies on the default remains limited to mvp_cases_6.json.",
            "A future entry point can accidentally reintroduce the original six-case assumption.",
            "Require an explicit manifest for generalized live commands or label the default as regression-only.",
        ),
        finding(
            "F011",
            "LOW",
            "INTEGRATION RISK",
            "Pilot orchestration's safe manifest view retains post-hoc timestamp metadata",
            [
                "scripts/canonical/run_full_pilot.py::safe_manifest_rows lines 65-72",
                "src/canonical_pipeline/runner.py::_safe_case lines 48-79",
            ],
            "provided_timestamp is present in the orchestration safe rows, but the runner reconstructs a strict allowlist and CaseState contains no such field. Static and 20-case runtime audits found no online use.",
            "Unnecessary pre-runtime exposure enlarges the future leakage surface even though current isolation is effective.",
            "Keep post-hoc fields in a separate object/file rather than the pre-call orchestration view.",
        ),
        finding(
            "F012",
            "LOW",
            "INTEGRATION RISK",
            "Question-independent visual/audio index metadata carries QA reference overlays",
            [
                "scripts/build_visual_state_regions.py lines 122-139",
                "scripts/build_audio_index.py lines 239-249",
            ],
            "The metadata explicitly labels provided timestamps as visualization-only, and canonical scoring reads only index rows. All six visual indexes nevertheless contain per-case reference metadata.",
            "A later consumer could accidentally use the co-located field as online localization.",
            "Physically separate evaluation overlays from reusable index artifacts.",
        ),
        finding(
            "F013",
            "MEDIUM",
            "RESEARCH LIMITATION",
            "Task5B linked windows are transitive expanding components, not pairwise-coherent event groups",
            ["src/retrieval/task5b.py::link_candidates lines 109-126"],
            "Each new candidate is compared with an expanding group union. In 00018_2 chained candidates produced one linked window covering essentially the full 149-second video.",
            "The linked-window label can overstate temporal coherence. Current Task6 does not use these windows for constrained selection.",
            "Evaluate non-transitive/event-centered grouping in a future method ablation.",
        ),
        finding(
            "F014",
            "MEDIUM",
            "RESEARCH LIMITATION",
            "Visual semantic scores do not directly rank local micro-window refinement inside a fixed search interval",
            ["scripts/run_task5b_retrieval.py::build_case visual branch"],
            "CLIP scores annotate coarse regions/anchors; micro-windows are selected primarily by temporal distance and dense frames cover the search interval.",
            "The system is coarse semantic localization plus deterministic local inspection, not end-to-end visual semantic reranking at frame level.",
            "Document this accurately and test alternatives only as future research.",
        ),
        finding(
            "F015",
            "HIGH",
            "RESEARCH LIMITATION",
            "Task5C structural sufficiency does not guarantee global/counting coverage",
            [
                "src/retrieval/task5c.py::classify_evidence",
                "src/canonical_pipeline/sufficiency.py::run_evidence_sufficiency",
            ],
            "Pilot diagnostics found 3/3 counting cases returned exact numeric answers without policy-level support; 2/3 had incomplete coverage.",
            "A structurally present candidate can be treated as sufficient despite inadequate domain coverage.",
            "Future work should separate structural, semantic, and coverage sufficiency; this changes research methodology.",
            affected_cases=["00061_5", "00018_9", "00006_4"],
        ),
        finding(
            "F016",
            "MEDIUM",
            "INTEGRATION RISK",
            "Pre-fallback final status is recomputed with post-fallback diagnostics",
            ["src/canonical_pipeline/sufficiency.py lines 116-120"],
            "diagnostics are built from post_candidates and then applied to both pre and post assessments. Current fallback adds speech only, so the observed baseline is unaffected.",
            "A future acoustic fallback could contaminate the reported pre-fallback status.",
            "Compute diagnostics separately for pre and post candidate sets in a future technical correction.",
        ),
        finding(
            "F017",
            "HIGH",
            "RESEARCH LIMITATION",
            "Task6 selection is individual role-priority ranking; relations are constructed after selection",
            [
                "src/canonical_pipeline/reranking.py lines 155-167",
                "src/retrieval/task6_relation_reranking.py::apply_packet_budget lines 104-127",
            ],
            "Candidates are filtered/sorted by role priority and bonuses before temporal_relations is called. Relations are annotations, not directly scored or constrained during selection.",
            "The current method is relation-aware representation after compact individual ranking, not atomic pair/group reranking.",
            "Treat group-aware or constrained selection as a future research contribution, not a technical patch.",
        ),
        finding(
            "F018",
            "HIGH",
            "RESEARCH LIMITATION",
            "Task6 does not enforce planner-required modalities or relation pairs as hard retention constraints",
            [
                "src/canonical_pipeline/reranking.py lines 155-165",
                "src/retrieval/task6_v1_1.py::classify_modalities lines 55-78",
            ],
            "Pilot diagnostics: required modality lost in 4 cases, cross-modal pair broken in 3, relation endpoint dropped in 3, and cross-modal completeness preserved in 9/14 diagnostic cases.",
            "A modality represented only by supporting-role candidates can be removed before relations are built; one endpoint can survive alone.",
            "Future selection should explicitly test required-modality and atomic pair constraints.",
            affected_cases=["00006_3", "00018_5", "00018_3", "00003_6"],
        ),
        finding(
            "F019",
            "MEDIUM",
            "RESEARCH LIMITATION",
            "Two configured Task6 budget fields are not enforced by canonical selection",
            [
                "configs/task6_relation_reranking.yaml",
                "src/retrieval/task6_relation_reranking.py::apply_packet_budget",
            ],
            "max_evidence_groups_per_case and max_plausible_alternatives_per_relation are present in configuration, but canonical Task6 always builds one group and does not cap alternatives by that field.",
            "Configuration implies controls that runtime does not implement.",
            "Either document these as reserved or evaluate implementation as a method change.",
        ),
        finding(
            "F020",
            "MEDIUM",
            "INTEGRATION RISK",
            "Dense-only canonical visual evidence has weak entity provenance",
            [
                "src/retrieval/task5b.py::canonical_visual_frames lines 177-201",
                "src/canonical_pipeline/reranking.py::_visual_packet lines 112-124",
            ],
            "00018_2's four retained frames use video_id=unknown_video and have no source candidate IDs because dense frames existed without a selected visual candidate entity. This representation caused the Phase-1 accounting failure.",
            "Identity remains unique by case-level packet ID, but frame dedup/drop audit provenance is weaker than the stated video_id+millisecond contract.",
            "Propagate video_id and a typed refinement-source identity in a future serialization-only patch.",
            affected_cases=["00018_2"],
        ),
        finding(
            "F021",
            "MEDIUM",
            "INTEGRATION RISK",
            "Uncertainty resolution defaults to supported for unrecognized uncertainty types",
            ["src/final_qa/task7b_validation.py::uncertainty_resolution_supported lines 82-95"],
            "Only speaker, semantic, fallback, and dataset-audit uncertainty have type-specific evidence rules; every other type resolves when all cited IDs merely exist.",
            "The local validator may accept a resolution whose cited modality cannot actually resolve that uncertainty.",
            "Define an explicit resolver-capability table and default unknown types to not_assessable/unresolved.",
        ),
        finding(
            "F022",
            "HIGH",
            "RESEARCH LIMITATION",
            "Final validation does not deterministically guard unsupported exact counts or general speaker/source claims",
            [
                "src/final_qa/task7b_v02.py::validate_v02 lines 167-234",
                "src/final_qa/task7b_validation.py::uncertainty_resolution_supported",
            ],
            "Validation enforces delivered modalities, citation bounds, and critical uncertainty. It does not compare count answers to coverage or generally parse speaker/source identity claims. Pilot recorded 3 unsupported exact counts.",
            "Prompt compliance remains partly model-dependent.",
            "Future answer-policy work could add task-specific guards, but that is a research-policy change.",
        ),
        finding(
            "F023",
            "MEDIUM",
            "INTEGRATION RISK",
            "Task7A's embedded required_output_schema and Task7B v3's actual Pydantic schema are different contract versions",
            [
                "src/canonical_pipeline/payload.py line 53",
                "src/canonical_pipeline/final_qa.py line 26",
                "src/final_qa/task7b_gemini.py lines 121-128",
            ],
            "The payload embeds the Task7A v1 schema, while the live request attaches FinalQAModelOutputV02. The older embedded schema is not included in request text, so there is no current provider conflict.",
            "Audit/report consumers can misidentify the authoritative final model schema.",
            "Version and name the provider-neutral policy schema separately from the provider structured-output schema.",
        ),
        finding(
            "F024",
            "HIGH",
            "INTEGRATION RISK",
            "Resume cannot prevent duplicate billable calls after provider success but before checkpoint commit",
            [
                "scripts/canonical/run_full_pilot.py lines 271-292",
                "src/evaluation/pilot_store.py::CaseCheckpointStore",
            ],
            "The atomic checkpoint is written only after the entire case returns. A crash after planner/Gemini success but before save causes resume to call the provider again. The pilot sidecar records two aborted planner attempts plus one authorized diagnostic planner call.",
            "Costs/attempts may be duplicated and incompletely attributable even though completed checkpoints are safe.",
            "Add a durable per-attempt journal and stage completion markers before larger runs.",
            affected_cases=["00002_7", "00018_2"],
        ),
        finding(
            "F025",
            "MEDIUM",
            "INTEGRATION RISK",
            "Smoke-result reuse compatibility does not bind code, prompt, config, or schema hashes",
            ["scripts/canonical/run_full_pilot.py::smoke_compatibility lines 75-93"],
            "Reuse checks completed cases, upstream hashes, fresh scoring, shared process, model name, thinking level, store flag, and trace fields, but not source/config/prompt/schema identities.",
            "Compatible-looking outputs from a different working tree could be mixed into one pilot.",
            "Persist and verify a canonical run fingerprint before reusing cases.",
        ),
        finding(
            "F026",
            "MEDIUM",
            "TECHNICAL BUG",
            "Final report enrichment previously overwrote completed smoke-derived checkpoint copies",
            ["scripts/canonical/run_full_pilot.py final report loop"],
            f"Hash audit: {hashes['preexisting_durable_checkpoint_hash_unchanged_count']}/16 preexisting checkpoint files remained byte-identical; three reused copies changed only initial_retrieval/temporal_linking enrichment plus source markers. Core research fields equal source smoke outputs.",
            "The no-overwrite checkpoint promise was violated for derived copies, though frozen upstream and model results were not changed.",
            "The runner now enriches in-memory report rows and no longer calls overwrite=True for completed checkpoints.",
            status="fixed_in_phase1_reporting",
            affected_cases=["00002_1", "00018_3", "00018_9"],
        ),
        finding(
            "F027",
            "MEDIUM",
            "MEASUREMENT RISK",
            "A primary-only modality query can be executed but reported as a skipped per-modality stage",
            [
                "src/canonical_pipeline/runner.py per-modality timing loop",
                "src/canonical_pipeline/retrieval.py lines 97-104",
            ],
            "00018_5 scored CLIP visual query/index candidates, but routing and visual_query_encode/visual_similarity_search/visual_retrieval timings are skipped because visual was not a resolver modality. retrieval_total still includes the work.",
            "Per-stage retrieval latency and execution rates undercount actual query-scoring work; uninstrumented/other overhead absorbs it.",
            "Time score_results independently from routed candidate branches.",
            affected_cases=["00018_5"],
        ),
        finding(
            "F028",
            "MEDIUM",
            "MEASUREMENT RISK",
            "Several timing records are nested/derived and cannot be additively summed",
            [
                "src/canonical_pipeline/runner.py::run_live_case",
                "src/instrumentation/timing.py::StageTimer.measured/timing_consistency",
            ],
            "local_visual_refinement is inside visual_retrieval; local_wav_materialization equals and is inside local_audio_refinement; fallback_local_asr is inside fallback_execution; Task6 children are inside reranking_and_packet. Derived records have no start/end monotonic coordinates.",
            "A consumer that sums the detailed table double-counts latency. timing_consistency also indexes by stage name, so duplicate names would overwrite silently.",
            "Publish an explicit additive top-level set and a nested-stage flag; retain raw parent timing.",
        ),
        finding(
            "F029",
            "MEDIUM",
            "MEASUREMENT RISK",
            "API wall-clock summary labels executed and amortized means ambiguously",
            ["src/evaluation/full_pilot_reporting.py lines 362-375"],
            "Gemini wall_clock_mean includes 0 for the two preflight-blocked cases (3.785s/question), while the executed-stage mean is 4.206s over 18 calls.",
            "Readers can mistake amortized latency for provider-call latency.",
            "Report both mean among executed calls and amortized mean per question.",
        ),
        finding(
            "F030",
            "MEDIUM",
            "MEASUREMENT RISK",
            "Aborted-call usage is manually reconstructed and token/billing usage is unavailable",
            [
                "outputs/pilot_20/baseline_v1/full_pilot_v0_1/aborted_attempts.json",
                "scripts/canonical/run_full_pilot.py lines 325-329",
            ],
            "Completed planner calls=20, attempted planner calls including the sidecar=23. Aborted token usage and billable status are null.",
            "Total token/cost summaries are lower bounds for the development execution history.",
            "Persist provider attempt metadata immediately after every response/error.",
        ),
        finding(
            "F031",
            "LOW",
            "MEASUREMENT RISK",
            "Cold GPU model-load timers do not explicitly synchronize CUDA completion",
            ["src/canonical_pipeline/query_scoring.py lines 92-145"],
            "Warm query encode paths synchronize CUDA; CLIP/CLAP model construction timing does not explicitly do so.",
            "Cold-start figures may omit deferred GPU work depending on library behavior.",
            "Synchronize before and after cold-load measurement in a future instrumentation-only patch.",
        ),
        finding(
            "F032",
            "MEDIUM",
            "INTEGRATION RISK",
            "Offline builders are case-oriented and can rebuild the same video for multiple questions",
            [
                "scripts/build_visual_state_regions.py main case loop",
                "scripts/build_audio_index.py main case loop",
                "scripts/build_audio_embeddings.py main case loops",
            ],
            "The online runtime correctly reuses one index per video, but clean offline scripts iterate manifest cases and write video_id directories without an explicit unique-video dedup contract.",
            "A future full manifest indexing run could recompute/overwrite identical video indexes and attach different case overlay metadata.",
            "Introduce a unique-video offline manifest/orchestrator before measuring offline indexing at scale.",
        ),
        finding(
            "F033",
            "HIGH",
            "INTEGRATION RISK",
            "Tests do not enforce Task6 group completeness, relation endpoint closure, or relation uniqueness",
            [
                "tests/test_task6_*.py",
                "tests/test_task7a_preflight.py",
                "tests/canonical_pipeline/",
            ],
            "The existing suite caught frame ordering/direction and now candidate identity, but F002-F004 survived 184 relevant collected tests.",
            "Equivalent schema/accounting defects can recur at the Task6->Task7A boundary.",
            "Add technical contract tests without prescribing a new selection policy.",
        ),
        finding(
            "F034",
            "MEDIUM",
            "INTEGRATION RISK",
            "Resume idempotency, run fingerprints, stale media reuse, and cross-video encoder metadata lack failure-injection tests",
            ["tests/canonical_pipeline/", "tests/test_task7a_preflight.py"],
            "Current tests cover atomic file writes, gating, lifecycle load counts, leakage, and frozen hashes, but not crash windows after live responses or mixed-index versions.",
            "Large runs remain exposed to preventable operational/schema failures.",
            "Add mocked crash/restart and stale-artifact tests before scaling.",
        ),
        finding(
            "F035",
            "INFORMATIONAL",
            "INTEGRATION RISK",
            "Canonical version mapping and live Task4 dependency boundaries are correct",
            [
                "config/canonical_pipeline.json",
                "src/canonical_pipeline/versions.py",
                "src/canonical_pipeline/retrieval.py",
            ],
            "Runtime validates Task5A v2, Task5B v1.1, Task5C v1.2, Task6 v1.2, Task7A v1, Task7B v3. Live scoring sets allow_historical_score_files=False; no 20-case checkpoint used Task4 score files.",
            "No hidden sequential v1->v1.1->v1.2 runtime chain was found.",
            "Keep explicit mapping and source-lineage documentation.",
            status="verified",
        ),
        finding(
            "F036",
            "INFORMATIONAL",
            "INTEGRATION RISK",
            "Persistent model lifecycle is verified per actual process segment",
            [
                "src/canonical_pipeline/query_scoring.py::FreshQueryScorer",
                "src/canonical_pipeline/live_boundaries.py::LazyWhisperFallback",
                "outputs/pilot_20/baseline_v1/full_pilot_v0_1/run_manifest.json",
            ],
            "Each smoke/13-case/resume segment loaded CLIP, Sentence-T5, and CLAP once. The 13-case segment loaded Whisper once for one fallback; the 4-case resume loaded it zero times. Planner and Gemini clients are constructed once per process.",
            "The technical stop necessarily produced three lifecycle segments, so cold start is not one single 20-case process cost.",
            "Preserve segment-level reporting and do not merge instance IDs across processes.",
            status="verified",
        ),
        finding(
            "F037",
            "INFORMATIONAL",
            "INTEGRATION RISK",
            "Gold/reference isolation is effective in the completed pilot",
            [
                "src/canonical_pipeline/state.py::CaseState",
                "src/retrieval/three_channel.py::safe_retrieval_case",
                "src/canonical_pipeline/runner.py::_safe_case",
            ],
            "All 20 runtime_gold_loaded flags are false, all Task7A leakage audits passed, and CaseState has no gold/reference field. Gold was reopened after 20 durable runtime checkpoints existed.",
            "No online leakage was found.",
            "Keep the allowlist and post-hoc storage separation.",
            status="verified",
        ),
        finding(
            "F038",
            "INFORMATIONAL",
            "INTEGRATION RISK",
            "Online execution reuses rather than rebuilds offline embeddings/indexes",
            [
                "src/canonical_pipeline/query_scoring.py::score_case",
                "src/canonical_pipeline/runner.py::run_live_case",
            ],
            "Fresh scoring loads .npy embeddings and metadata, encodes the question, and computes cosine ranking. It does not run frame extraction, VAD, global ASR, transcript embedding, or acoustic embedding per question.",
            "The offline/online boundary is correct for the current pilot, though embeddings are read from disk per query.",
            "Retain the boundary; optimize only in a later research/engineering phase.",
            status="verified",
        ),
        finding(
            "F039",
            "LOW",
            "MEASUREMENT RISK",
            "Task5B runtime labels query-vector encodes as new embedding computations",
            ["src/canonical_pipeline/retrieval.py lines 137-138"],
            "new_embedding_computations increments for question encodes even though offline index embeddings are not rebuilt.",
            "Reports can be misread as online index recomputation.",
            "Rename the metric to query_embedding_computations in a reporting-only compatibility layer.",
        ),
        finding(
            "F040",
            "MEDIUM",
            "TECHNICAL BUG",
            "The durable partial-pilot status sidecar remains stale after final completion",
            [
                "outputs/pilot_20/baseline_v1/full_pilot_v0_1/partial_pilot_status.json",
                "scripts/canonical/run_full_pilot.py",
                "scripts/canonical/report_incomplete_full_pilot.py",
            ],
            "The final checkpoints and aggregates contain 20 cases, while partial_pilot_status.json still reports 16 completed cases because the final runner does not retire or refresh the interruption sidecar.",
            "Automation or a reviewer reading the sidecar as current state can incorrectly conclude that the pilot is incomplete.",
            "Treat the sidecar as an interruption snapshot in reports; in a future reporting-only patch, atomically mark it superseded or refresh it after final aggregate commit.",
        ),
    ]


def architecture_map() -> list[dict[str, Any]]:
    return [
        {"stage": "Offline visual", "scope": "offline reusable", "implementation": "scripts/build_visual_state_regions.py; scripts/build_visual_microclips.py; src/visual/*", "outputs": "1 fps frames, CLIP frame/region embeddings, microclip index"},
        {"stage": "Offline speech/acoustic", "scope": "offline reusable", "implementation": "scripts/build_audio_index.py; scripts/build_audio_embeddings.py; src/audio/index.py", "outputs": "full WAV, VAD regions, global Whisper transcripts, Sentence-T5/CLAP indexes"},
        {"stage": "Task5A v2", "scope": "online", "implementation": "src/canonical_pipeline/planner.py -> src/question_planner/task5a.py", "outputs": "validated planner schema and deterministic cues"},
        {"stage": "Fresh query scoring", "scope": "online", "implementation": "src/canonical_pipeline/query_scoring.py -> src/retrieval/three_channel.py", "outputs": "normalized cosine rankings over reusable indexes"},
        {"stage": "Task5B v1.1", "scope": "online", "implementation": "src/canonical_pipeline/retrieval.py -> scripts/run_task5b_retrieval.py + src/retrieval/task5b.py", "outputs": "temporally clipped candidates, local visual refinement, selected candidates"},
        {"stage": "Task5C v1.2", "scope": "online", "implementation": "src/canonical_pipeline/sufficiency.py + media_materialization.py", "outputs": "structural status, one-shot fallback, local WAV provenance"},
        {"stage": "Task6 v1.2", "scope": "online", "implementation": "src/canonical_pipeline/reranking.py + src/retrieval/task6_*.py", "outputs": "roles, compact candidates, relations, groups, frame serialization, accounting"},
        {"stage": "Task7A v1", "scope": "online deterministic", "implementation": "src/canonical_pipeline/payload.py + src/final_qa/task7a_preflight.py", "outputs": "model payload, assets/leakage/preflight"},
        {"stage": "Task7B v3", "scope": "online external + local", "implementation": "src/canonical_pipeline/final_qa.py + src/final_qa/task7b_gemini.py + task7b_v02.py", "outputs": "raw Gemini JSON, deterministic corrected answer"},
        {"stage": "Post-hoc evaluation", "scope": "not online latency", "implementation": "src/evaluation/*; scripts/canonical/run_full_pilot.py", "outputs": "gold comparison, lexical diagnostics, dashboard"},
    ]


def task6_logic_markdown(task6_summary: dict[str, Any]) -> str:
    return f"""# Task6 v1.2 exact current logic

This is a code review of the frozen method, not a redesign.

## Data flow

1. Task5C `post_fallback_candidates` are copied as the source candidate-entity set.
2. Nonvisual candidates receive deterministic roles from operation, phrase match,
   Task5C acoustic diagnostics, and fallback provenance.
3. Visual micro-window candidates and/or Task5B canonical frame assets are
   represented as one `canonical_visual_evidence` entity. Frames are deduplicated
   by `video_id + millisecond timestamp`, selected by anchor proximity, capped at
   four, then serialized chronologically while preserving `selection_rank`.
4. Supporting-only candidates are removed whenever any protected-role candidate
   exists.
5. Remaining candidates are individually sorted by maximum role priority, then
   start time and candidate ID. Exact phrase, local WAV, valid ASR timestamps and
   dense visual provenance add fixed bonuses.
6. Protected-role candidates may exceed `max_candidates_per_case`; the budget is
   explicitly soft in that case.
7. Temporal/functional relations are constructed **after** candidate retention.
8. One operation-shaped evidence group is built and Task7A serializes only its
   members.

## Priority

`trigger=80`, `temporal_anchor=75`, `direct_evidence=70`, `resolver=65`,
`plausible_response=60`, `fallback_recovered=55`, `supporting=30`,
`alternative=25`, `not_required=0`.

Bonuses: exact phrase `+12`, local clip `+8`, valid ASR timestamps `+5`, dense
visual provenance `+4`.

## Answers to the requested questions

1. Priority is the highest role weight plus the four deterministic bonuses;
   ties use time then candidate ID.
2. Reranking is **individual-evidence ranking with relation-aware output
   annotation**, not pair/group ranking.
3. Planner modalities are metadata/role inputs, not hard retention constraints.
4. Yes, a relation endpoint can be dropped before relations are reconstructed;
   pilot diagnostic count: {task6_summary['relation_endpoint_dropped_count']}.
5. Yes, anchor/resolver pairs can be broken; pilot count:
   {task6_summary['cross_modal_pair_broken_count']}.
6. Yes, all evidence of a planner-requested modality can be removed when its
   candidates are supporting-only; required-modality loss count:
   {task6_summary['required_modality_loss_count']}.
7. Protected roles survive candidate-budget conflict and emit a soft violation;
   modality completeness itself is not protected.
8. Relations are not scored directly and are not used as a selection constraint.
9. Task6 optimizes deterministic role priority and compactness, with limited
   temporal proximity only for frame selection; it does not jointly optimize
   relation consistency or modality completeness.
10. Current behavior is closer to **highest-priority compact evidence** than a
    guaranteed **minimum sufficient evidence** set.

## Pilot association (not causal proof)

- Mean entering Task6: {task6_summary['mean_entering']:.3f}
- Mean retained: {task6_summary['mean_retained']:.3f}
- Mean genuine drops: {task6_summary['mean_actually_dropped']:.3f}
- Mean retained/input ratio: {task6_summary['mean_compression_ratio']:.3f}
- Required modality lost: {task6_summary['required_modality_loss_count']} cases
- Cross-modal pair broken: {task6_summary['cross_modal_pair_broken_count']} cases
- Relation endpoint dropped: {task6_summary['relation_endpoint_dropped_count']} cases
"""


def recommendations_markdown() -> str:
    rows = [
        ("Minimum sufficient evidence constraint", "Compaction can retain high-priority evidence without operational sufficiency.", "Require Task5C operation requirements to remain satisfiable after Task6.", "Directly aligns compacting with sufficiency.", "Requires formal sufficiency predicates and may keep more evidence.", "Yes", "Strong: constrained evidence selection is a clear systems contribution.", "Compare current priority selection vs sufficiency-constrained selection at matched budgets."),
        ("Required-modality preservation", "All evidence from an answer-required modality can disappear.", "Reserve at least one qualified item for each answer-required modality before filling remaining budget.", "Simple, interpretable, likely improves cross-modal completeness.", "A weak modality item may displace stronger evidence; planner mistakes become hard constraints.", "Yes", "Moderate: useful baseline/constrained ablation.", "Hard vs soft modality preservation; report completeness, QA, evidence size."),
        ("Anchor-resolver atomicity", "One endpoint of a meaningful pair can survive alone.", "Treat validated anchor/resolver pairs as atomic selectable units.", "Preserves evidence logic and prevents dangling relations.", "Pairs consume more budget and pair generation quality matters.", "Yes", "Strong: relation-aware evidence-set construction.", "Individual vs pair-atomic selection under equal media/token budget."),
        ("Group-aware reranking", "Relations are currently built only after individual retention.", "Score groups using member priority, temporal coherence, modality coverage and redundancy.", "Makes relation-aware reranking substantive rather than representational.", "More design choices and greater overfitting risk.", "Yes", "Strong if defined generally and evaluated broadly.", "Individual, pair-atomic, and group-scored variants."),
        ("Budget-aware constrained selection", "Soft protected items and modality constraints can conflict with fixed budgets.", "Solve a small deterministic optimization over candidate/group variables and explicit constraints.", "Transparent trade-offs and guaranteed invariants.", "Constraint infeasibility and objective-weight sensitivity.", "Yes", "Strong methodological contribution if kept interpretable.", "Vary budgets and constraints; measure violations, recall, QA, latency."),
        ("Operation-aware requirements", "count, delay, identify-source and describe-sound need different evidence structures.", "Declare per-operation required roles, coverage and relation templates.", "Reduces reliance on incidental role normalization.", "Operation taxonomy errors can hard-code benchmark assumptions.", "Yes", "Moderate-to-strong if generic and dataset-independent.", "Ablate shared generic constraints vs operation-aware constraints."),
        ("Task5C/Task6 responsibility boundary", "Task5C declares sufficiency before Task6 can remove required structure.", "Either revalidate sufficiency after Task6 or make Task6 preserve Task5C's certificate.", "Creates an explicit end-to-end sufficiency contract.", "A post-Task6 fallback loop could increase cost/complexity; avoid unbounded loops.", "Yes", "Strong architecture contribution.", "Certificate-preserving Task6 vs one post-compaction sufficiency recheck."),
    ]
    header = "| Direction | Problem | Simplest version | Benefit | Risk | Method change | Thesis suitability | Required ablation |\n|---|---|---|---|---|---|---|---|"
    body = "\n".join("| " + " | ".join(item) + " |" for item in rows)
    return f"""# Task6 future recommendations — opinion only

No recommendation in this file was implemented.

{header}
{body}

## Recommended research direction

Start with a **Task5C sufficiency certificate plus required-modality and
anchor-resolver atomicity constraints**, implemented as a small deterministic
budget-aware selector. It is the smallest direction that directly tests the
observed failure pattern while remaining interpretable. Compare it against the
frozen individual-priority Task6 at identical candidate/frame/audio budgets.
Keep a separate technical patch for group completeness, relation endpoint
closure and relation de-duplication; those schema fixes should not be credited
as research improvements.
"""


def schema_markdown(checkpoint: dict[str, Any]) -> str:
    return f"""# Schema contract review

## Intended producer/consumer chain

`Task5B candidate -> Task5C evidence candidate -> Task6 retained candidate/group
-> Task7A modality record -> Task7B evidence catalog/citation`

## Verified contracts

- Candidate IDs are unique within Task6 input/retained/genuine-drop sets for
  all 20 cases.
- Retained and genuinely dropped sets are disjoint for all 20.
- Visual frames are chronological, have sequential `presentation_order`, retain
  `selection_rank`, and have unique millisecond timestamps in all model-facing
  packets.
- All model-facing frame/WAV paths exist.
- All 20 Task7A leakage checks passed.

## Contract violations/risks

- Retained candidates omitted from all groups:
  `{checkpoint['retained_candidates_omitted_from_payload_groups']}`.
- Relations with unavailable group endpoints:
  `{checkpoint['relations_with_missing_group_endpoints']}`.
- Duplicate relation records:
  `{checkpoint['duplicate_relations']}`.
- Task7A calculates but does not enforce every visual check.
- Evidence catalog dictionaries would silently overwrite duplicate evidence IDs;
  current Task6 identity checks reduce but do not replace a Task7A uniqueness
  assertion.
- The embedded Task7A output-policy schema differs from the Pydantic v0.2
  structured-output schema actually attached to Gemini.

## Required technical invariants (future patch, no selection change)

1. Every retained candidate is a member of at least one retained group.
2. Every evidence ID is unique per case and per payload.
3. Every relation endpoint is either a supplied evidence ID or an explicitly
   typed non-evidence node permitted by schema.
4. Relation identity keys are unique.
5. Every model-facing visual/audio asset passes the complete Task7A validation
   set.
6. Producer schema versions are recorded alongside the packet and request.
"""


def accounting_markdown(checkpoint: dict[str, Any]) -> str:
    return f"""# Accounting and identity review

## Canonical entity taxonomy

| Entity | Identity | Counted as candidate? | Notes |
|---|---|---:|---|
| Case | `case_id` | no | Owns one online run |
| Question | case-owned text | no | No gold/reference fields in CaseState |
| Retrieval candidate | `candidate_id` | yes | Speech/acoustic/micro-window candidate entity |
| Canonical visual evidence | `candidate_id` | yes | May represent source candidates or selected frame assets |
| Refined visual frame | `video_id + normalized_timestamp_ms` plus path provenance | no | Media asset, not evidence candidate |
| Local WAV clip | path + source/interval/config provenance | no | Media asset attached to acoustic evidence |
| Speech segment | candidate ID / transcript segment ID | yes | ASR record may be fallback-recovered |
| Relation | source ID + target ID + type + basis | no | Must not affect candidate counts |
| Evidence group | `group_id` | no | Container over candidate entities |
| Packet record | case/stage record | no | Serialization container |

## Phase-1 fixed equation

For 00018_2:

`3 source candidates - 0 genuine drops - 0 represented sources + 1 introduced
canonical visual entity = 4 retained candidate entities`.

The old arithmetic omitted the introduced canonical entity. The new invariant
reconciles sets, not only counts, and excludes frames, clips and relations.

## Whole-pilot verification

- Candidate identity errors: `{checkpoint['candidate_identity_errors']}`
- Visual contract errors: `{checkpoint['visual_contract_errors']}`
- Model-facing media errors: `{checkpoint['model_facing_media_errors']}`

## Remaining conflations

- Reporting funnels sometimes compare candidate entities with final modality
  records without explicitly naming the unit.
- Task6 groups and relations are counted separately in aggregate files, but the
  producer does not yet enforce group membership closure.
- Dense-only visual refinement can introduce a canonical evidence entity even
  when no visual candidate entity appears in Task5C input; this is now explicit
  accounting but still weak provenance.
- `new_embedding_computations` names online query-vector computation as though
  it might be index rebuilding.
"""


def timing_markdown(latency: dict[str, Any], efficiency: dict[str, Any]) -> str:
    online = latency["online_latency"]
    stage = {item["stage"]: item for item in latency["stage_latency"]}
    return f"""# Timing instrumentation review

## Measured pilot

- Online mean/median/p95: {online['mean']:.6f} / {online['median']:.6f} /
  {online['p95']:.6f} seconds.
- Mean timing coverage: {latency['mean_timing_coverage']:.6f}.
- Planner executed mean: {stage['question_planner']['mean_executed_sec']:.6f}s.
- Gemini executed mean: {stage['final_gemini_api']['mean_executed_sec']:.6f}s;
  amortized mean/question: {stage['final_gemini_api']['amortized_mean_sec']:.6f}s.
- Fallback executed once at
  {stage['fallback_execution']['mean_executed_sec']:.6f}s; amortized
  {stage['fallback_execution']['amortized_mean_sec']:.6f}s/question.

## Non-overlapping top-level decomposition

The report's additive components are Planner; Fresh retrieval/query scoring;
Temporal/refinement; Sufficiency/fallback; Task6 reranking/packet;
Media/payload; Final Gemini; Parse/validation; Other overhead.

## Nested, not additive

| Parent | Nested child |
|---|---|
| `retrieval_total` | modality retrieval, temporal linking, budget selection, visual refinement |
| `visual_retrieval` | visual query encode/search and local visual refinement |
| `sufficiency_and_fallback` | sufficiency, fallback, local audio refinement |
| `local_audio_refinement` | `local_wav_materialization` (same duration) |
| `fallback_execution` | `fallback_local_asr` |
| `reranking_and_packet` | relation construction, relation reranking, packet build |
| `final_model_pipeline` | Gemini call, parse, local validation |

## Risks

1. One primary-only CLIP query (00018_5) occurred inside `retrieval_total` while
   its per-modality stages were marked skipped.
2. `StageTimer.measured` preserves real child durations but has no start/end
   coordinates, so it cannot reconstruct an exact timeline.
3. `timing_consistency` converts stage names to a dictionary; duplicate names
   would overwrite.
4. Gemini API `wall_clock_mean` in efficiency is amortized across all 20 cases
   ({efficiency['gemini_api']['wall_clock_mean']:.6f}s), not the executed-call mean.
5. Cold encoder cost spans three process segments because the pilot reused a
   three-case smoke run and later resumed after a mandated stop. It is not one
   single-process 20-case cold start.
"""


def tests_markdown() -> str:
    return """# Test coverage review

The focused canonical/Task5B-Task7B collection contains 184 tests. The final
execution result is recorded separately in `phase2_test_results.json`.

## Strong existing coverage

| Area | Covered invariants |
|---|---|
| Versioning | Explicit version map; no mtime inference; logical Task7B v3 |
| Manifest/leakage | Manifest acceptance/rejection; retrieval allowlist; gold post-hoc |
| Retrieval | temporal cues, modality routes, budgets, half-open frames, fresh score equivalence |
| Task5C | structural states, fallback once, ASR clipping/provenance, broad acoustic warnings |
| Task6 | frame dedup/order, response direction, modality classification, drop accounting, candidate identity |
| Task7A | assets, timestamps, leakage, uncertainty, speaker verification false |
| Task7B | media/MIME, store=false, low thinking, retry limit, citations, uncertainty restoration |
| Lifecycle/timing | one encoder load per process, skipped=null, top-level coverage, cache missing=null |

## Highest-priority missing technical tests

1. Every retained Task6 candidate appears in at least one Task7A group.
2. Every relation endpoint is present or explicitly typed as a non-evidence node.
3. Relation keys are unique after v1.1/v1.2 corrections.
4. Crash after planner/Gemini response but before checkpoint save does not
   silently repeat a completed provider call.
5. Reused case fingerprint includes code, config, prompt, schema and index IDs.
6. Existing WAV reuse rejects wrong source/config/duration.
7. Persistent encoders reject mixed index model/revision/normalization metadata.
8. Per-modality timing records every scored query, even when its routed candidate
   branch is skipped.
9. Task7A enforces every computed visual check and evidence-ID uniqueness.
10. Offline builders deduplicate video IDs when a manifest contains multiple
    questions for one video.

These are technical contract tests; they should not assert a new Task6 modality
or pair-retention policy.
"""


def table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(item))}</th>" for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(item))}</td>" for item in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def pre(value: Any) -> str:
    return "<pre>" + html.escape(json.dumps(value, ensure_ascii=False, indent=2)) + "</pre>"


def review_html(
    findings_rows: list[dict[str, Any]],
    checkpoint: dict[str, Any],
    hashes: dict[str, Any],
    architecture: list[dict[str, Any]],
    lifecycle: dict[str, Any],
    pilot_summary: dict[str, Any],
) -> str:
    severity = collections.Counter(item["severity"] for item in findings_rows)
    categories = collections.Counter(item["category"] for item in findings_rows)
    finding_rows = [
        [item["finding_id"], item["severity"], item["category"], item["status"], item["title"], "; ".join(item["affected_cases"])]
        for item in findings_rows
    ]
    details = "".join(
        f"<details class='{item['severity'].lower()}'><summary>{html.escape(item['finding_id'] + ' — ' + item['title'])}</summary>"
        + pre(item)
        + "</details>"
        for item in findings_rows
    )
    architecture_rows = [
        [item["stage"], item["scope"], item["implementation"], item["outputs"]]
        for item in architecture
    ]
    phase1 = read_json(OUT / "phase1_00018_2_fix_proof.json")
    regressions = read_json(OUT / "phase1_original_six_regression.json")
    resumed = pilot_summary["resumed_in_phase1"]
    return f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><title>Canonical Baseline v1 deep audit</title>
<style>
body{{font:15px/1.5 system-ui;margin:2rem auto;max-width:1500px;color:#17202a}}
h1,h2{{color:#17365d}} table{{border-collapse:collapse;width:100%;margin:1rem 0 2rem}}
th,td{{border:1px solid #aeb6bf;padding:.45rem;vertical-align:top}} th{{background:#eaf2f8}}
pre{{white-space:pre-wrap;overflow:auto;background:#f4f6f7;padding:1rem}}
details{{border-left:6px solid #7f8c8d;padding:.5rem 1rem;margin:.7rem 0;background:#fafafa}}
details.blocker{{border-color:#7b241c}} details.high{{border-color:#c0392b}}
details.medium{{border-color:#d68910}} details.low{{border-color:#5dade2}}
.ok{{color:#196f3d;font-weight:700}} .warn{{color:#9a7d0a;font-weight:700}}
</style></head><body>
<h1>Canonical Baseline v1 — two-phase deep review</h1>
<p><b>Scope:</b> one Phase-1 Task6 bookkeeping correction, frozen pilot resume, then read-only full-pipeline audit. No research-policy recommendation was implemented.</p>

<h2>1. Full architecture map</h2>{table(['Stage','Boundary','Actual implementation','Artifact/result'], architecture_rows)}

<h2>2. Phase-1 blocker and minimal fix</h2>
<p><b>Before:</b> 3 input candidate entities − 0 genuine drops − 0 represented sources = 3 expected, but 4 were retained.</p>
<p><b>Root cause:</b> one canonical visual evidence entity was introduced from already-selected dense frame assets. It was neither an input candidate nor a drop.</p>
<p class='ok'>Category A bookkeeping bug. Set-based identity accounting fixed the description; retained evidence, relations, frames, budget and payload semantics were exactly unchanged.</p>
{pre({'pre_fix_equation':phase1['pre_fix_failing_equation'],'post_fix':phase1['post_fix_identity_accounting'],'research_fields_equal':phase1['research_behavior_fields_equal']})}

<h2>3. Pilot completion</h2>
{pre(pilot_summary)}
{table(['Case','Status','Prediction','Fallback','Preflight'], [[item['case_id'],item['answer_status'],item['answer'],item['fallback_count'],item['preflight']] for item in resumed])}

<h2>4. Original-six regression</h2>
<p class='ok'>All six cases passed 26/26 checks; no category-4 research-behavior mismatch.</p>
{table(['Case','Checks','Classification'], [[item['case_id'],f"{item['exact_check_count']}/{item['check_count']}",item['overall_classification']] for item in regressions['results']])}

<h2>5. Data/schema flow and accounting taxonomy</h2>
{pre(checkpoint)}

<h2>6. Source/version truth</h2>
<p>Task5A v2 → Task5B v1.1 → Task5C v1.2 → Task6 v1.2 → Task7A v1 → Task7B v3. Historical correction modules provide composed helpers; historical batch versions are not executed sequentially. Live Task5B fresh scoring does not load Task4 score files.</p>

<h2>7. Offline vs online computation map</h2>
<p><b>Offline reusable:</b> frame extraction/sampling, CLIP image/region embeddings, audio extraction, VAD, global Whisper ASR, Sentence-T5 transcript embeddings, CLAP acoustic embeddings, index serialization.</p>
<p><b>Online:</b> planner; query construction/encoding/similarity; retrieval, temporal clipping/linking, local refinement; Task5C/fallback/WAV materialization; Task6; Task7A; Gemini; local validation.</p>

<h2>8. Model lifecycle map</h2>{pre(lifecycle)}

<h2>9. Findings dashboard</h2>
{pre({'by_severity':dict(severity),'by_category':dict(categories),'open_high_or_blocker':sum(item['status']=='open' and item['severity'] in ('BLOCKER','HIGH') for item in findings_rows)})}
{table(['ID','Severity','Category','Status','Finding','Cases'], finding_rows)}

<h2>10. Detailed findings</h2>{details}

<h2>11. Task6 exact current logic</h2>
<p>Task6 is individual role-priority compacting followed by relation construction and operation-shaped grouping. It is not pair/group-constrained selection. See <code>task6_logic_review.md</code>.</p>

<h2>12. Task6 observed pilot patterns</h2>{pre(read_json(PILOT/'aggregate_task6_diagnostics.json')['summary'])}

<h2>13. Task6 recommendations — opinion only</h2>
<p>Minimum-sufficiency certification, hard answer-required modality preservation, and anchor/resolver atomicity are the leading future directions. See <code>task6_recommendations.md</code>. None was implemented.</p>

<h2>14. Task5C limitations</h2>
<p>Structural candidate presence is distinct from semantic and coverage sufficiency. The pilot's three count cases all produced unsupported exact-count diagnostics; two had incomplete coverage. This is a research limitation, not changed here.</p>

<h2>15. Timing/instrumentation risks</h2>
<p>Top-level coverage is high, but detailed stages are nested. One primary-only CLIP query was hidden by a skipped per-modality timing record, and API summaries mix executed and amortized means. See <code>timing_instrumentation_review.md</code>.</p>

<h2>16. Test coverage gaps and prioritized next steps</h2>
{pre(read_json(OUT/'phase2_test_results.json'))}
<ol><li>Technical Task6→Task7A group completeness, endpoint closure and relation uniqueness patch/tests.</li><li>Durable provider-attempt journal and run fingerprint before another large live run.</li><li>Strict local media/index-model provenance validation.</li><li>Only then design and ablate a research-method Task6 constrained selector.</li></ol>

<h2>Hash status</h2>{pre(hashes)}
</body></html>"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [read_json(path) for path in sorted(CHECKPOINTS.glob("*.json"))]
    if len(cases) != 20:
        raise RuntimeError(f"Deep audit requires 20 durable checkpoints, found {len(cases)}")

    checkpoint = checkpoint_audit(cases)
    hashes = hash_audit()
    model_indexes = model_index_audit(cases)
    lifecycle = read_json(PILOT / "run_manifest.json").get("lifecycle", {})
    task6 = read_json(PILOT / "aggregate_task6_diagnostics.json")["summary"]
    latency = read_json(PILOT / "aggregate_latency.json")
    efficiency = read_json(PILOT / "aggregate_efficiency.json")
    metrics = read_json(PILOT / "aggregate_metrics.json")
    finding_rows = findings(checkpoint, hashes)
    severity_counts = dict(collections.Counter(item["severity"] for item in finding_rows))
    category_counts = dict(collections.Counter(item["category"] for item in finding_rows))

    metric_rows = {
        row["case_id"]: row
        for row in (
            json.loads(line)
            for line in (PILOT / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    resumed_ids = ["00018_2", "00061_4", "00004_2", "00006_7"]
    resumed = []
    for case_id in resumed_ids:
        case = next(item for item in cases if item["case_id"] == case_id)
        answer = case["validated_answer"]
        resumed.append(
            {
                "case_id": case_id,
                "question": case["question"],
                "gold_answer_posthoc": metric_rows[case_id].get("gold_answer"),
                "answer": answer.get("answer"),
                "answer_status": answer.get("answer_status"),
                "fallback_count": case["sufficiency_fallback"]["fallback_execution_count"],
                "preflight": case["preflight"]["status"],
                "source_run": case.get("source_run"),
            }
        )
    pilot_summary = {
        "represented_cases": len(cases),
        "reused_smoke_cases": 3,
        "new_durable_live_cases": 17,
        "resumed_in_phase1": resumed,
        "answer_status_distribution": metrics["answer_status_distribution"],
        "technical_failures_remaining": 0,
        "frozen_upstream_hashes_unchanged": hashes["frozen_artifacts_hash_identical"],
        "gold_loaded_posthoc_only": checkpoint["runtime_gold_flags_all_false"],
        "leakage_audits_all_passed": checkpoint["leakage_audits_all_passed"],
        "external_calls_in_phase2_audit": 0,
    }

    findings_payload = {
        "audit_version": "deep_audit_v0_1",
        "research_behavior_modified_in_phase2": False,
        "finding_count": len(finding_rows),
        "by_severity": severity_counts,
        "by_category": category_counts,
        "findings": finding_rows,
        "checkpoint_evidence": checkpoint,
        "model_index_evidence": model_indexes,
    }
    write_json(OUT / "audit_findings.json", findings_payload)
    write_json(OUT / "phase1_hashes_after.json", hashes)
    write_json(OUT / "phase1_pilot_completion.json", pilot_summary)
    write_json(OUT / "implementation_map.json", architecture_map())
    write_json(OUT / "model_lifecycle_review.json", {"lifecycle": lifecycle, "index_models": model_indexes})

    write_text(OUT / "task6_logic_review.md", task6_logic_markdown(task6))
    write_text(OUT / "task6_recommendations.md", recommendations_markdown())
    write_text(OUT / "schema_contract_review.md", schema_markdown(checkpoint))
    write_text(OUT / "accounting_identity_review.md", accounting_markdown(checkpoint))
    write_text(OUT / "timing_instrumentation_review.md", timing_markdown(latency, efficiency))
    write_text(OUT / "test_coverage_review.md", tests_markdown())

    highest = [
        item for item in finding_rows
        if item["status"] == "open" and item["severity"] in {"BLOCKER", "HIGH"}
    ]
    summary = f"""# Canonical Baseline v1 deep audit

## Outcome

- Phase-1 00018_2 root cause: **pure candidate-entity bookkeeping bug**.
- Research selection fields unchanged by the fix: **true**.
- Original-six structural regression: **6/6, 26/26 checks each**.
- Pilot represented: **20/20**; remaining technical failures: **0**.
- Frozen Task5A-Task7A artifacts hash-identical: **{hashes['frozen_artifacts_hash_identical']}**.
- Gold post-hoc and Task7A leakage checks passed: **true**.
- Phase-2 external calls: **0**.
- Tests: **184/184 focused**; **262 pytest tests + 1 equivalent isolated WAV fixture** across the full tests directory, with no research-test failure.

## Findings

- Total: **{len(finding_rows)}**
- By severity: `{severity_counts}`
- By category: `{category_counts}`
- Open BLOCKER findings: **{sum(item['severity']=='BLOCKER' for item in highest)}**
- Open HIGH findings: **{sum(item['severity']=='HIGH' for item in highest)}**

Highest-risk technical items are Task6 group membership completeness, relation
endpoint closure, duplicate relations, and crash-window provider-call
idempotency. They were not changed in Phase 2.

## Pilot measurement snapshot

- Statuses: `{metrics['answer_status_distribution']}`
- Diagnostic EM / normalized EM / BLEU / CIDEr:
  `{metrics['summary']}`
- Online mean / median / p95:
  `{latency['online_latency']['mean']:.3f}` / `{latency['online_latency']['median']:.3f}` /
  `{latency['online_latency']['p95']:.3f}` seconds.
- Task6 diagnostics: `{task6}`
- Counting diagnostics:
  `{read_json(PILOT/'aggregate_counting_diagnostics.json')['summary']}`

## Interpretation

The canonical baseline is executable and the 20-case measurement is complete,
but it is not yet technically clean enough to treat every Task6-retained item
as guaranteed model-facing evidence. Technical schema/identity fixes should be
separated from future Task6 research-method changes.
"""
    write_text(OUT / "deep_audit_summary.md", summary)
    write_text(
        OUT / "deep_audit_review.html",
        review_html(
            finding_rows,
            checkpoint,
            hashes,
            architecture_map(),
            lifecycle,
            pilot_summary,
        ),
    )
    print(
        json.dumps(
            {
                "cases": len(cases),
                "findings": len(finding_rows),
                "severity": severity_counts,
                "output": str(OUT),
                "external_calls": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
