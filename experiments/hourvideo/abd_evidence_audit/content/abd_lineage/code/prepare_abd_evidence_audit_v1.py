#!/usr/bin/env python3
"""Prepare a fresh, no-gold ABD evidence-audit package.

This deliberately does not read annotations or historical audit labels.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "outputs/abd_eval300_formal_v1"
INPUTS = ROOT / "drafts/abd_direct_eval300_v1/inputs"
OUT = ROOT / "outputs/abd_eval300_evidence_audit_v1"
MANIFEST = ROOT / "drafts/abd_direct_eval300_v1/formal_candidate/formal_candidate_manifest.json"
RAW_CLOSURE = ROOT / "outputs/abd_eval300_analysis_v1/structural_closure_report.json"
ARCHIVE = ROOT / "outputs/abd_eval300_analysis_v1/ABD_RESULTS_PACKAGE.tar.gz"
SCIENTIFIC_SHA = "24e6305d798378c352099ee963b00efb54d5b40b545c34221949dfba36cd36dc"
RAW_SHA = "b16175b602bd62e82dd42e44ec5a4249654acf0d75a90a2fda623ad81057649d"
ARCHIVE_SHA = "d9f1273d1463b1a68e6a2a60455895072d27bc05f157dabfd89ae7f73af45c31"

CRITERIA = (ROOT / "audit/codex_evidence_audit_r1_r3_gens_v3_v1/AUDIT_CRITERIA_FROZEN.md").read_text(encoding="utf-8")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refuse overwrite: {OUT}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if len(manifest["tasks"]) != 900 or {x["variant"] for x in manifest["tasks"]} != {"A", "B", "D"}:
        raise SystemExit("candidate manifest population mismatch")
    rows = {}
    for arm in "ABD":
        for line in (INPUTS / f"{arm}.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rows[f"{arm}:{row['question_id']}"] = row
    starts = {}
    for line in (FORMAL / "journals/request_starts.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if d["task_id"] in starts:
            raise SystemExit("duplicate request start")
        starts[d["task_id"]] = d
    artifacts = {}
    for p in (FORMAL / "task_artifacts").glob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        artifacts[d["task_id"]] = d
    if set(starts) != set(artifacts) or len(starts) != 900:
        raise SystemExit("request/artifact population mismatch")
    source = []
    for task in manifest["tasks"]:
        tid = task["task_id"]
        row = rows[tid]
        start, artifact = starts[tid], artifacts[tid]
        if start["variant"] != task["variant"] or artifact["result_class"] != "valid_answer":
            raise SystemExit(f"status/identity mismatch: {tid}")
        imgs = []
        for image in row.get("images") or []:
            p = Path(image["path"])
            if not p.is_file() or sha(p) != image["sha256"]:
                raise SystemExit(f"image SHA mismatch: {tid} {p}")
            imgs.append({
                "frame_path": str(p),
                "frame_sha256": image["sha256"],
                "resolved_timestamp_sec": image["resolved_timestamp_sec"],
            })
        source.append({
            "task_id": tid,
            "question_id": row["question_id"],
            "question": row["question"],
            "options": row["options"],
            "prediction": artifact["parsed_prediction"],
            "reason": artifact["reason"],
            "terminal_status": artifact["result_class"],
            "map": row.get("map"),
            "images": imgs,
            "request": {"model": start["model"], "max_tokens": start["max_tokens"], "image_count": start["image_count"], "map_present": start["map_present"]},
        })
    # Deterministic identity-obscuring order, exactly 192 batches as in the prior protocol.
    source.sort(key=lambda x: hashlib.sha256(("abd-codex-audit-v1\\0" + x["task_id"]).encode()).hexdigest())
    ids = []
    mapping = []
    public = []
    for i, item in enumerate(source, 1):
        aid = f"E{i:04d}"
        pub = dict(item)
        pub["audit_id"] = aid
        pub.pop("task_id", None)
        pub["map"] = item["map"]
        pub["images"] = item["images"]
        (OUT / "review_packages").mkdir(parents=True, exist_ok=True)
        if item["map"]:
            mp = Path(item["map"]["path"])
            if sha(mp) != item["map"]["sha256"]:
                raise SystemExit(f"map SHA mismatch: {item['task_id']}")
            target = OUT / "review_packages" / f"{aid}.map.json"
            target.write_bytes(mp.read_bytes())
            pub["map_locator"] = str(target)
        else:
            pub["map_locator"] = None
        write_json(OUT / "review_packages" / f"{aid}.json", pub)
        public.append(pub)
        ids.append(aid)
        mapping.append({"audit_id": aid, "task_id": item["task_id"], "variant": item["task_id"].split(":", 1)[0], "question_id": item["question_id"]})
    # Force 192 batches (the historical protocol's batch count) with balanced contiguous slices.
    batches = []
    for i in range(192):
        lo = round(i * len(ids) / 192)
        hi = round((i + 1) * len(ids) / 192)
        batches.append({"batch_id": f"B{i+1:03d}", "audit_ids": ids[lo:hi], "route_count": hi - lo, "image_count": sum(len(public[j]["images"]) for j in range(lo, hi))})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "AUDIT_CRITERIA_FROZEN.md").write_text(CRITERIA, encoding="utf-8")
    write_json(OUT / "identity_mapping.json", {"warning": "contains identity; separate from blinded review packages", "rows": mapping})
    write_json(OUT / "batch_manifest.json", {"batches": batches, "batch_count": 192})
    write_json(OUT / "package_freeze_manifest.json", {
        "schema_version": "abd_evidence_audit_package_v1",
        "gold_loaded": False,
        "route_count": 900,
        "arms": {a: 300 for a in "ABD"},
        "scientific_manifest_sha256": SCIENTIFIC_SHA,
        "raw_closure_sha256": RAW_SHA,
        "archive_sha256": ARCHIVE_SHA,
        "source_candidate_manifest_sha256": sha(MANIFEST),
        "raw_closure_file_sha256": sha(RAW_CLOSURE),
        "archive_file_sha256": sha(ARCHIVE),
        "public_package_count": len(public),
        "batch_count": len(batches),
        "historical_audit_labels_loaded": False,
    })
    print(json.dumps({"out": str(OUT), "routes": len(public), "batches": len(batches), "gold_loaded": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
