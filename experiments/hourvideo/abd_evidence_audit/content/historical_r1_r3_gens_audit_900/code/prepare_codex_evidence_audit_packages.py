#!/usr/bin/env python3
"""Prepare blinded/no-gold evidence packages for the 900-route Codex audit."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gens_haiku_eval300.runtime import atomic_json, canonical_sha, sha256_file  # noqa: E402

DIRECT = ROOT / "outputs/direct_v1_formal/direct_v1_2_3x16_r1_r3_eval300_formal_v1"
GENS = ROOT / "outputs/gens_haiku_structured_v3/formal_v3_direct_parser_aligned_eval300"
GENS_MANIFEST = ROOT / "outputs/gens_haiku_structured_v3/preflight_v3_direct_parser_aligned/formal_manifest.json"
OUT = ROOT / "audit/codex_evidence_audit_r1_r3_gens_v3_v1"


CRITERIA = """# Codex辅助证据审计：冻结标准 v1

本审计由当前 Codex 分批执行，不是人工复核，也不调用独立评审模型。审阅阶段隐藏方法标签、gold、正确性和旧版本结果；Direct map 的结构可能暴露方法类型，因此不宣称完全盲法。

## A：最终选项的证据支持

- `supported`：实际输入中存在能区分候选项的明确证据。
- `partially_supported`：存在相关线索，但关键判断仍缺证据。
- `unsupported`：实际输入无法支持所选项。
- `contradicted`：实际输入明确反驳所选项。
- `unreviewable`：图片无法查看、证据包损坏或无法完成核验。
- `no_final_answer`：路线没有最终答案。

## B：最终理由的依据（可多选）

- `corresponds_to_input`
- `reasonable_unverified_inference`
- `common_sense_option_wording_or_guess`
- `unsupported_factual_assertion`
- `contradicts_input`
- `input_evidence_cannot_verify_claim`

必须核对实际发送图片；不能仅凭模型理由、文件名或统计报告判断。Direct 还须核对实际 map、截至最终回答已观察的图片与工具反馈，并把 evidence source 标为 `map`、`images`、`map_and_images` 或 `none`。map 支持但图片未确认时不能称为独立视觉支持。

持续时间、次数、先后、全程不存在等问题要求相应时间覆盖。更多选中帧不等于持续更久；选中帧未显示不等于视频中不存在。理由提到输入未显示内容时，只标为输入证据无法核验，不推断现实中不存在。

每条必须记录简短说明、可定位证据引用、缺少证据、置信度（high/medium/low）和是否需进一步复核。审计标签不改预测、不改正式评分。
"""


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def options_from_question(question: dict) -> dict[str, str]:
    return {row["option_id"]: row["text"] for row in question["answer_options"]}


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"audit package already exists; refuse overwrite: {OUT}")
    direct_manifest = load_json(DIRECT / "formal_manifest.json")
    direct_routes = {row["route_id"]: row for row in direct_manifest["routes"]}
    direct_inputs = {p.stem: load_json(p) for p in (DIRECT / "inputs").glob("*.json")}
    gens_manifest = load_json(GENS_MANIFEST)
    selector = {
        json.loads(line)["qa_uid"]: json.loads(line)
        for line in Path(gens_manifest["config"]["selector_path"]).read_text(encoding="utf-8").splitlines()
        if line
    }
    identities = []
    source_rows = []
    for artifact_path in sorted((DIRECT / "route_artifacts").glob("*.json")):
        artifact = load_json(artifact_path)
        source_id = f"{artifact['method']}:{artifact['question_id']}"
        route = direct_routes[source_id]
        question = direct_inputs[artifact["question_id"]]
        seen = []
        feedback = []
        for turn in artifact.get("turns", []):
            feedback.append({
                "turn_index": turn["turn_index"],
                "action_type": turn["action_type"],
                "requested_timestamps_sec": turn.get("requested_timestamps_sec", []),
                "images_transmitted": turn.get("images_transmitted", 0),
                "duplicate_requests": turn.get("duplicate_requests", 0),
                "provider_status": turn.get("provider_status"),
            })
            for frame in turn.get("resolved_frames", []):
                if not frame.get("duplicate_of_seen_frame") and not frame.get("duplicate_of_turn_frame"):
                    seen.append({
                        "requested_timestamp_sec": frame["requested_timestamp_sec"],
                        "resolved_timestamp_sec": frame["resolved_timestamp_sec"],
                        "frame_path": frame["frame_path"],
                        "observed_sha256": frame["observed_sha256"],
                    })
        source_rows.append({
            "source_identity": source_id,
            "source_group": artifact["method"],
            "question_id": artifact["question_id"],
            "question": question["question_text"],
            "options": options_from_question(question),
            "prediction": artifact.get("final_prediction"),
            "reason": next((t.get("action_reason") for t in reversed(artifact.get("turns", [])) if t.get("action_type") == "final_answer"), None),
            "terminal_status": artifact["terminal_status"],
            "map_path": route["map_path"],
            "map_sha256": route["map_sha256"],
            "images": seen,
            "tool_feedback": feedback,
        })
    for qid in gens_manifest["question_ids"]:
        artifact = load_json(GENS / "routes" / f"{qid}.json")
        row = selector[qid]
        source_rows.append({
            "source_identity": f"GENS:{qid}",
            "source_group": "GENS",
            "question_id": qid,
            "question": row["question"],
            "options": row["options"],
            "prediction": artifact.get("prediction"),
            "reason": artifact.get("reason"),
            "terminal_status": artifact["terminal_status"],
            "map_path": None,
            "map_sha256": None,
            "images": [{
                "requested_timestamp_sec": None,
                "resolved_timestamp_sec": frame["timestamp_sec"],
                "frame_path": frame["path"],
                "observed_sha256": frame["observed_sha256"],
            } for frame in artifact["selected_frames"]],
            "tool_feedback": [],
        })

    # Deterministic method-obscuring order; mapping is stored separately.
    source_rows.sort(key=lambda row: hashlib.sha256(("codex-audit-v1\0" + row["source_identity"]).encode()).hexdigest())
    package_dir = OUT / "review_packages"
    map_dir = OUT / "review_maps"
    package_dir.mkdir(parents=True)
    map_dir.mkdir(parents=True)
    public_rows = []
    for index, row in enumerate(source_rows, 1):
        neutral_id = f"E{index:04d}"
        map_review_path = None
        if row["map_path"]:
            raw = Path(row["map_path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != row["map_sha256"]:
                raise RuntimeError(f"map SHA mismatch: {row['source_identity']}")
            map_review_path = map_dir / f"{neutral_id}.json"
            map_review_path.write_bytes(raw)
        for frame in row["images"]:
            path = Path(frame["frame_path"])
            if not path.is_file() or sha256_file(path) != frame["observed_sha256"]:
                raise RuntimeError(f"frame unavailable/SHA mismatch: {row['source_identity']} {path}")
        public = {
            "neutral_id": neutral_id,
            "question": row["question"],
            "options": row["options"],
            "prediction": row["prediction"],
            "reason": row["reason"],
            "terminal_status": row["terminal_status"],
            "map_review_path": str(map_review_path) if map_review_path else None,
            "map_sha256": row["map_sha256"],
            "images": row["images"],
            "tool_feedback": row["tool_feedback"],
        }
        atomic_json(package_dir / f"{neutral_id}.json", public)
        public_rows.append(public)
        identities.append({
            "neutral_id": neutral_id,
            "source_identity": row["source_identity"],
            "source_group": row["source_group"],
            "question_id": row["question_id"],
        })

    # At most 10 routes, and reduce batches when the image burden is high.
    batches, current, image_total = [], [], 0
    for row in public_rows:
        count = len(row["images"])
        if current and (len(current) >= 10 or image_total + count > 50):
            batches.append(current); current=[]; image_total=0
        current.append(row["neutral_id"]); image_total += count
    if current:
        batches.append(current)
    batch_manifest = [{"batch_id": f"B{i:03d}", "neutral_ids": ids,
                       "route_count": len(ids),
                       "image_count": sum(len(public_rows[int(x[1:])-1]["images"]) for x in ids)}
                      for i, ids in enumerate(batches, 1)]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "AUDIT_CRITERIA_FROZEN.md").write_text(CRITERIA, encoding="utf-8")
    atomic_json(OUT / "identity_mapping.json", {"warning": "contains method identities; keep separate during review", "rows": identities})
    atomic_json(OUT / "batch_manifest.json", {"batches": batch_manifest})
    atomic_json(OUT / "package_freeze.json", {
        "audit_name": "Codex辅助证据审计",
        "gold_loaded": False,
        "route_count": len(public_rows),
        "direct_routes": 600,
        "gens_routes": 300,
        "batch_count": len(batch_manifest),
        "criteria_sha256": sha256_file(OUT / "AUDIT_CRITERIA_FROZEN.md"),
        "public_packages_sha256": canonical_sha(public_rows),
        "identity_mapping_sha256": sha256_file(OUT / "identity_mapping.json"),
        "direct_raw_namespace": str(DIRECT),
        "gens_raw_namespace": str(GENS),
    })
    print(json.dumps(load_json(OUT / "package_freeze.json"), indent=2))


if __name__ == "__main__":
    main()
