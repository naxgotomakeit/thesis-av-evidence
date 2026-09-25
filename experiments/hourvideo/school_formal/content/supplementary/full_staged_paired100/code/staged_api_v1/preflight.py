"""No-model/no-API population and protocol validation."""
from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path
from typing import Any

from .config import StagedConfig, canonical_sha, sha256_file
from .contracts import PROMPTS, final_schema, fine_schema, shared_schema


PROHIBITED_KEYS = {"answer", "correct_answer", "gold", "gold_answer", "correct_option", "previous_prediction"}
LEGACY_ENTRY_MODULE = "experiments.hourvideo_v6_6_2_shared_coarse_contract_v1.core"
RUNTIME_DISTRIBUTIONS = ("anthropic", "numpy", "torch", "transformers", "scikit-learn", "Pillow")


def _resolve_local_module(module: str, roots: list[Path]) -> Path | None:
    relative = Path(*module.split("."))
    for root in roots:
        for candidate in (root / relative.with_suffix(".py"), root / relative / "__init__.py"):
            if candidate.is_file():
                return candidate.resolve()
    return None


def legacy_dependency_files(cfg: StagedConfig) -> list[Path]:
    """Statically close the frozen V6.6.2 local Python import graph.

    V6.6.2 is imported lazily at live time.  Hashing only this workspace would
    therefore miss changes to its scheduling, validators and retrieval code.
    """
    legacy_root = Path(cfg.legacy_runtime_root).resolve()
    roots = [legacy_root, legacy_root / "src"]
    entry = _resolve_local_module(LEGACY_ENTRY_MODULE, roots)
    if entry is None:
        raise FileNotFoundError(f"legacy entry module unavailable: {LEGACY_ENTRY_MODULE}")
    pending, seen = [entry], set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module)
                modules.update(f"{node.module}.{alias.name}" for alias in node.names if alias.name != "*")
        for module in modules:
            dependency = _resolve_local_module(module, roots)
            if dependency is not None and dependency not in seen:
                pending.append(dependency)
    return sorted(seen, key=str)


def live_dependency_versions(cfg: StagedConfig) -> dict[str, str]:
    # Keep missing optional distributions explicit instead of importing them.
    safer_script = """
import importlib.metadata, json, pathlib, platform, sys
names = %r
versions = {}
for name in names:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = "NOT_INSTALLED"
print(json.dumps({"python": platform.python_version(), "implementation": platform.python_implementation(),
                  "executable": str(pathlib.Path(sys.executable).resolve()), "distributions": versions}, sort_keys=True))
""" % (list(RUNTIME_DISTRIBUTIONS),)
    completed = subprocess.run([cfg.live_python, "-c", safer_script], capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"cannot fingerprint live dependencies: {completed.stderr.strip()}")
    return json.loads(completed.stdout)


def runtime_fingerprint_payload(*, config_path: Path, workspace_root: Path,
                                workspace_files: list[str], legacy_files: list[Path],
                                dependency_versions: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_sha256": sha256_file(config_path),
        "workspace_files": {name: sha256_file(workspace_root / name) for name in workspace_files},
        "legacy_dynamic_import_files": {str(path): sha256_file(path) for path in legacy_files},
        "live_dependency_versions": dependency_versions,
    }


def _reject_gold(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        found = PROHIBITED_KEYS.intersection(value)
        if found:
            raise ValueError(f"prohibited fields at {path}: {sorted(found)}")
        for key, child in value.items():
            _reject_gold(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_gold(child, f"{path}[{index}]")


def population(cfg: StagedConfig) -> list[str]:
    manifest = json.loads(Path(cfg.eval300_manifest).read_text(encoding="utf-8"))
    ids = [str(row["question_id"]) for row in manifest["questions"]]
    ordered = [line.strip() for line in Path(cfg.eval300_uid_order).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(ids) != 300 or len(set(ids)) != 300 or ids != ordered:
        raise ValueError("frozen Planner population is not canonical Eval300 order")
    return ids


def _five_question_pilot(ids: list[str]) -> list[str]:
    result, videos = [], set()
    for qid in ids:
        video = qid.rsplit("_", 2)[0]
        if video not in videos:
            result.append(qid); videos.add(video)
        if len(result) == 5:
            break
    if len(result) != 5:
        raise ValueError("cannot select five distinct-video pilot identities")
    return result


def validate(cfg: StagedConfig, *, workspace_root: Path) -> dict[str, Any]:
    ids = population(cfg)
    planner_rows, video_assets = [], {}
    for index, qid in enumerate(ids):
        question_path = Path(cfg.source_experiment) / "cases" / qid / "question_input.json"
        case_config_path = Path(cfg.source_experiment) / "case_configs" / f"{qid}.json"
        planner_path = Path(cfg.planner_source_experiment) / "cases" / qid / "r3_2" / "planner.json"
        if not all(path.is_file() for path in (question_path, case_config_path, planner_path)):
            raise FileNotFoundError(qid)
        question = json.loads(question_path.read_text(encoding="utf-8")); case = json.loads(case_config_path.read_text(encoding="utf-8")); planner = json.loads(planner_path.read_text(encoding="utf-8"))
        _reject_gold(question); _reject_gold(case); _reject_gold(planner)
        video = str(question["video_uid"])
        if question["question_id"] != qid or case["question_id"] != qid or case["video_uid"] != video:
            raise ValueError(f"question/case identity mismatch: {qid}")
        if planner.get("question_id") != qid or planner.get("video_uid") != video or planner.get("route") != "r3_2":
            raise ValueError(f"Planner identity mismatch: {qid}")
        if planner.get("planner_model") != cfg.model or planner.get("provider") != "anthropic":
            raise ValueError(f"Planner provider/model mismatch: {qid}")
        map_path = Path(case["asset_paths"]["r3_2_navigation_map.json"])
        if sha256_file(map_path) != case["asset_sha256"]["r3_2_navigation_map.json"] or sha256_file(map_path) != planner["input_asset_sha256"]:
            raise ValueError(f"R3 map lineage mismatch: {qid}")
        if sha256_file(question_path) != case["question_input_sha256"]:
            raise ValueError(f"question SHA mismatch: {qid}")
        if [row["option_id"] for row in question["answer_options"]] != list("ABCDE"):
            raise ValueError(f"option order mismatch: {qid}")
        if video not in video_assets:
            assets = {}
            for name in ("shared_hierarchy.json", "medium_siglip.float32.npy", "r3_medium_captions.json", "r3_2_navigation_map.json"):
                path = Path(case["asset_paths"][name])
                if not path.is_file() or sha256_file(path) != case["asset_sha256"][name]:
                    raise ValueError(f"asset missing/SHA mismatch: {video}:{name}")
                assets[name] = {"path": str(path), "sha256": case["asset_sha256"][name], "bytes": path.stat().st_size}
            fine_path = Path(case["siglip_npz"])
            if not fine_path.is_file() or sha256_file(fine_path) != case["siglip_npz_sha256"]:
                raise ValueError(f"Fine index mismatch: {video}")
            assets["fine_siglip.npz"] = {"path": str(fine_path), "sha256": case["siglip_npz_sha256"], "bytes": fine_path.stat().st_size}
            hierarchy = json.loads(Path(case["asset_paths"]["shared_hierarchy.json"]).read_text(encoding="utf-8"))
            for fine in hierarchy["fine_nodes"]:
                frame = Path(fine["source_frame_path"])
                if not frame.is_file(): raise FileNotFoundError(frame)
            assets["fine_frame_count"] = len(hierarchy["fine_nodes"])
            video_assets[video] = assets
        planner_rows.append({"index": index, "question_id": qid, "video_uid": video, "planner_path": str(planner_path), "planner_sha256": sha256_file(planner_path), "r3_map_sha256": planner["input_asset_sha256"]})

    sample_q = json.loads((Path(cfg.source_experiment) / "cases" / ids[0] / "question_input.json").read_text(encoding="utf-8"))
    schema_examples = {
        "shared": shared_schema(sample_q, ["E0"], ["C01"]),
        "fine": fine_schema(["F001"], [f"{ids[0]}::shared"]),
        "final": final_schema(sample_q, 1, "grounded"),
    }
    code_paths = [workspace_root / "src/staged_api_v1" / name for name in ("contracts.py", "provider.py", "store.py", "config.py", "legacy_bridge.py", "preflight.py", "runtime.py")]
    code_paths.append(workspace_root / "scripts/run_full_staged_api_r3_eval300_v1.py")
    legacy_paths = legacy_dependency_files(cfg)
    dependency_versions = live_dependency_versions(cfg)
    return {
        "status": "PASS", "experiment_id": cfg.experiment_id, "gold_loaded": False,
        "api_calls": 0, "model_calls": 0, "question_count": len(ids), "video_count": len(video_assets),
        "route_count": len(ids), "route": "r3_2", "ordered_question_ids": ids,
        "planner_reuse": {"count": len(planner_rows), "all_provider_anthropic_haiku45": True, "rows": planner_rows},
        "assets": video_assets, "pilot_proposal": _five_question_pilot(ids),
        "frozen_population_sources": {
            "eval300_manifest": {"path": cfg.eval300_manifest, "sha256": sha256_file(Path(cfg.eval300_manifest))},
            "eval300_uid_order": {"path": cfg.eval300_uid_order, "sha256": sha256_file(Path(cfg.eval300_uid_order))},
        },
        "prompt_sha256": {stage: canonical_sha(text) for stage, text in PROMPTS.items()},
        "schema_example_sha256": {stage: canonical_sha(schema) for stage, schema in schema_examples.items()},
        "schema_examples": schema_examples,
        "config": {"model": cfg.model, "temperature": cfg.temperature, "timeout_sec": cfg.timeout_sec,
            "max_tokens_by_stage": cfg.max_tokens_by_stage, "max_transport_retries": cfg.max_transport_retries,
            "max_validation_retries": cfg.max_validation_retries, "max_claim_rounds": cfg.max_claim_rounds,
            "max_fine_images_per_batch": cfg.max_fine_images_per_batch, "cache_policy": cfg.cache_policy,
            "pricing": cfg.pricing, "hard_api_budget_usd": cfg.hard_api_budget_usd,
            "request_admission_reserve_usd": cfg.per_request_budget_reserve_usd,
            "request_reserve_semantics": "conservative admission floor, not a verified provider-cost upper bound",
            "unknown_provider_outcome_policy": "retain reservation; do not fabricate zero cost",
            "live_python": cfg.live_python},
        "code_sha256": {str(path): sha256_file(path) for path in code_paths if path.is_file()},
        "legacy_dynamic_import_sha256": {str(path): sha256_file(path) for path in legacy_paths},
        "legacy_dynamic_import_file_count": len(legacy_paths),
        "live_dependency_versions": dependency_versions,
    }


def legacy_downstream_config(cfg: StagedConfig, *, preflight_root: Path) -> dict[str, Any]:
    return {
        "experiment": cfg.experiment_id,
        "source_experiment": cfg.source_experiment,
        "planner_source_experiment": cfg.planner_source_experiment,
        "output_root": str(Path(cfg.output_root) / "legacy_live"),
        "case_identity": "question_id", "planner_identity": "question_id", "execution_sides": ["r3_2"],
        "max_validation_retries": cfg.max_validation_retries, "max_claim_rounds": cfg.max_claim_rounds,
        "ranking": {"visual_weight": 0.6, "lexical_weight": 0.3, "max_fine_per_medium": cfg.max_fine_per_medium,
            "minimum_fine_gap_sec": cfg.minimum_fine_gap_sec, "max_total_fine_evidence_per_claim": cfg.max_total_fine_evidence_per_claim,
            "max_total_fine_evidence_per_batch": cfg.max_fine_images_per_batch},
        "siglip_text": cfg.siglip_text,
        "anthropic": {"provider": "anthropic", "model": cfg.model, "timeout_sec": cfg.timeout_sec, "temperature": cfg.temperature,
            "claim_max_tokens": cfg.max_tokens_by_stage["shared"], "planner_max_tokens": 8192},
        "gemini": {"provider": "anthropic_vision", "model": cfg.model, "timeout_sec": cfg.timeout_sec, "temperature": cfg.temperature,
            "max_output_tokens": cfg.max_tokens_by_stage["fine"]},
        "final": {"provider": "anthropic", "model": cfg.model, "timeout_sec": cfg.timeout_sec, "temperature": cfg.temperature,
            "max_output_tokens": cfg.max_tokens_by_stage["final"]},
        "eval300_ordered_uid_path": cfg.eval300_uid_order,
        "pilot10_question_ids": [], "annotation_path": "PROHIBITED_DURING_LIVE_EXECUTION",
        "preflight_root": str(preflight_root),
    }
