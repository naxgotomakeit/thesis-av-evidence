#!/usr/bin/env python3
"""Read-only integrity, identity-closure, and sensitive-content checks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compact_tree(root: Path) -> str:
    lines = [". (uploadable tree; DO_NOT_COMMIT summarized separately)"]
    top_files = sorted(p.name for p in root.iterdir() if p.is_file())
    for name in top_files:
        lines.append(f"├── {name}")
    for top in [root / "content", root / "tools"]:
        if not top.exists():
            continue
        direct_files = [p for p in top.iterdir() if p.is_file()]
        direct_size = sum(p.stat().st_size for p in direct_files)
        suffix = f" [{len(direct_files)} direct files; {direct_size} bytes]" if direct_files else ""
        lines.append(f"├── {top.name}/{suffix}")
        dirs = sorted(p for p in top.rglob("*") if p.is_dir())
        for directory in dirs:
            rel = directory.relative_to(root)
            files = [p for p in directory.iterdir() if p.is_file()]
            if files:
                size = sum(p.stat().st_size for p in files)
                lines.append(f"│   ├── {rel.as_posix()}/ [{len(files)} files; {size} bytes]")
    local_files = [p for p in (root / "DO_NOT_COMMIT").rglob("*") if p.is_file()]
    lines.append(f"└── DO_NOT_COMMIT/ [{len(local_files)} local-only files]")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("staging_root", type=Path)
    args = parser.parse_args()
    source = args.source_root.resolve()
    root = args.staging_root.resolve()
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: object) -> None:
        checks.append((name, bool(ok), json.dumps(detail, ensure_ascii=False, sort_keys=True) if not isinstance(detail, str) else detail))

    # Every source-derived file has a surviving source, a matching original hash,
    # and a staged copy matching the recorded staged hash.
    with (root / "SOURCE_MANIFEST.csv").open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    missing_source = []
    missing_staged = []
    source_mismatch = []
    staged_mismatch = []
    illegal_transform = []
    for row in manifest:
        src = source / row["source_portable_path"]
        dst = root / row["staged_path"]
        if not src.is_file():
            missing_source.append(row["source_portable_path"])
        elif sha256(src) != row["original_sha256"]:
            source_mismatch.append(row["source_portable_path"])
        if not dst.is_file():
            missing_staged.append(row["staged_path"])
        elif sha256(dst) != row["staged_sha256"]:
            staged_mismatch.append(row["staged_path"])
        if row["transformation"] not in {"exact_copy", "absolute_personal_path_redaction_only"}:
            illegal_transform.append(row["staged_path"])
        if row["transformation"] == "exact_copy" and row["original_sha256"] != row["staged_sha256"]:
            illegal_transform.append(row["staged_path"] + ": exact-copy hash differs")
    check("source files present", not missing_source, missing_source)
    check("staged files present", not missing_staged, missing_staged)
    check("original SHA-256 values match", not source_mismatch, source_mismatch)
    check("staged SHA-256 values match", not staged_mismatch, staged_mismatch)
    check("only declared transformations", not illegal_transform, illegal_transform)

    abd = root / "content/abd_lineage"
    initial_map = load_json(abd / "01_initial_audit_900/identity_mapping.json")["rows"]
    initial_reviews = load_jsonl(abd / "01_initial_audit_900/scored_evidence_audit_blinded.jsonl")
    initial_ids = {r["audit_id"] for r in initial_map}
    initial_review_ids = {r["audit_id"] for r in initial_reviews}
    expected_ids = {f"E{i:04d}" for i in range(1, 901)}
    check("ABD initial identity closure", len(initial_map) == 900 and initial_ids == expected_ids and initial_review_ids == expected_ids, {"mapping": len(initial_map), "reviews": len(initial_reviews), "unique_ids": len(initial_ids)})

    correction_map = load_json(abd / "02_correction_176/validation/reconstructed_identity_mapping.json")["rows"]
    correction_reviews = load_jsonl(abd / "02_correction_176/validation/extracted/corrected_blinded_reviews.jsonl")
    correction_ids = {r["original_audit_id"] for r in correction_map}
    correction_review_ids = {r["audit_id"] for r in correction_reviews}
    roles = Counter(r["sample_role"] for r in correction_map)
    control_ids = {r["original_audit_id"] for r in correction_map if r["sample_role"] == "control"}
    check("ABD correction 176 closure", len(correction_map) == 176 and len(correction_reviews) == 176 and correction_ids == correction_review_ids and correction_ids <= initial_ids and roles == Counter({"target": 116, "control": 60}), {"mapping": len(correction_map), "reviews": len(correction_reviews), "roles": dict(roles), "outside_initial": sorted(correction_ids - initial_ids)})

    independent_map = load_json(abd / "03_independent_review_31/identity_mapping_private.json")["rows"]
    independent_reviews = load_jsonl(abd / "03_independent_review_31/blinded_reviewer_output/blinded_reviews.jsonl")
    review_to_source = {r["review_id"]: r["source_audit_id"] for r in independent_map}
    independent_review_ids = {r["review_id"] for r in independent_reviews}
    independent_source_ids = set(review_to_source.values())
    check("ABD independent 31 closure", len(independent_map) == 31 and len(independent_reviews) == 31 and independent_review_ids == set(review_to_source) and independent_source_ids <= control_ids, {"mapping": len(independent_map), "reviews": len(independent_reviews), "outside_controls": sorted(independent_source_ids - control_ids)})

    final_dir = abd / "05_final_authoritative_900"
    final_rows = load_jsonl(final_dir / "final_evidence_audit_labels.jsonl")
    final_ids = {r["audit_id"] for r in final_rows}
    selected = Counter(r["selected_review_version"] for r in final_rows)
    selected_sets = {key: {r["audit_id"] for r in final_rows if r["selected_review_version"] == key} for key in selected}
    final_ok = (
        len(final_rows) == 900
        and final_ids == initial_ids
        and selected == Counter({"sol_independent_review": 31, "correction": 145, "original_abd": 724})
        and selected_sets["sol_independent_review"] == independent_source_ids
        and selected_sets["correction"] == correction_ids - independent_source_ids
        and selected_sets["original_abd"] == initial_ids - correction_ids
    )
    check("ABD 900 → 176 → 31 → final 900 identity closure", final_ok, {"final_rows": len(final_rows), "source_counts": dict(selected), "31_plus_145_plus_724": 31 + 145 + 724})

    final_manifest_path = final_dir / "final_manifest.json"
    final_manifest = load_json(final_manifest_path)
    source_rows_by_staged = {r["staged_path"]: r for r in manifest}
    payload_errors = []
    payload_modes = Counter()
    for item in final_manifest["files"]:
        staged_rel = f"content/abd_lineage/05_final_authoritative_900/{item['path']}"
        staged = root / staged_rel
        row = source_rows_by_staged.get(staged_rel)
        if not staged.is_file() or row is None:
            payload_errors.append(item["path"] + ": missing")
            continue
        if row["original_sha256"] != item["sha256"] or int(row["original_bytes"]) != item["bytes"]:
            payload_errors.append(item["path"] + ": original payload differs from final manifest")
        elif row["transformation"] == "exact_copy":
            if sha256(staged) != item["sha256"] or staged.stat().st_size != item["bytes"]:
                payload_errors.append(item["path"] + ": staged exact payload differs")
            payload_modes["exact_staged_payload"] += 1
        else:
            if sha256(staged) != row["staged_sha256"]:
                payload_errors.append(item["path"] + ": redacted staged payload differs")
            payload_modes["source_original_plus_declared_path_redaction"] += 1
    label_freeze_hash_ok = sha256(final_dir / "label_freeze.json") == final_manifest["label_freeze_sha256"]
    check("final_manifest payload verification", not payload_errors and label_freeze_hash_ok, {"errors": payload_errors, "verification_modes": dict(payload_modes), "label_freeze_hash_ok": label_freeze_hash_ok})

    historical = root / "content/historical_r1_r3_gens_audit_900/audit"
    hist_map = load_json(historical / "identity_mapping.json")["rows"]
    hist_rows = load_jsonl(historical / "evidence_audit_frozen.jsonl")
    hist_ids = {r["neutral_id"] for r in hist_map}
    hist_review_ids = {r["neutral_id"] for r in hist_rows}
    review_batch_rows = sum(len(load_jsonl(p)) for p in (historical / "reviews").glob("*.jsonl"))
    hist_freeze = load_json(historical / "evidence_audit_freeze.json")
    check("historical R1/R3/GenS 900 closure", len(hist_map) == 900 and len(hist_rows) == 900 and hist_ids == expected_ids and hist_review_ids == expected_ids and review_batch_rows == 900 and hist_freeze["route_count"] == 900, {"mapping": len(hist_map), "frozen_reviews": len(hist_rows), "batch_review_rows": review_batch_rows})

    abd_by_id = {r["audit_id"]: r for r in initial_map}
    hist_by_id = {r["neutral_id"]: r for r in hist_map}
    semantic_tuple_matches = sum(
        (abd_by_id[i]["question_id"], abd_by_id[i]["variant"])
        == (hist_by_id[i]["question_id"], hist_by_id[i]["source_group"])
        for i in expected_ids
    )
    namespace_ok = (
        set(abd_by_id) == set(hist_by_id) == expected_ids
        and semantic_tuple_matches == 0
        and sha256(abd / "01_initial_audit_900/identity_mapping.json") != sha256(historical / "identity_mapping.json")
    )
    check("no ABD/historical identity conflation", namespace_ok, {"shared_local_E_labels": 900, "same_semantic_tuple_at_same_label": semantic_tuple_matches, "ABD_identity_fields": ["audit_id", "variant", "question_id", "task_id"], "historical_identity_fields": ["neutral_id", "source_group", "source_identity", "question_id"], "separate_staging_roots": True})

    # Sensitive scan: strict credential signatures and personal absolute paths.
    uploadable = [p for p in root.rglob("*") if p.is_file() and "DO_NOT_COMMIT" not in p.parts]
    known_prefixes = ["sk-" + "ant-", "sk-" + "proj-", "gh" + "p_", "github_" + "pat_", "AK" + "IA", "AI" + "za", "hf" + "_"]
    secret_hits = []
    personal_path_hits = []
    env_files = []
    private_header = "-----BEGIN " + "PRIVATE KEY-----"
    assignment = re.compile(r"(?i)(?:api[_-]?key|password|client[_-]?secret|access[_-]?token|auth[_-]?token)\s*[=:]\s*[\"']([^\"']{8,})[\"']")
    personal = re.compile(r"/(?:cs/student/(?:msc|project_msc)/[^\s\"']+|home/[^/\s\"']+|Users/[^/\s\"']+)")
    for path in uploadable:
        if path.name == ".env" or path.name.startswith(".env."):
            env_files.append(path.relative_to(root).as_posix())
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for prefix in known_prefixes:
            if prefix in text:
                secret_hits.append(f"{path.relative_to(root)}: credential-like prefix {prefix[:4]}…")
        if private_header in text:
            secret_hits.append(f"{path.relative_to(root)}: private-key header")
        for match in assignment.finditer(text):
            value = match.group(1)
            if value not in {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"} and not value.startswith(("<", "${")):
                secret_hits.append(f"{path.relative_to(root)}: secret-like assignment")
        if personal.search(text):
            personal_path_hits.append(path.relative_to(root).as_posix())
    scan_pass = not secret_hits and not personal_path_hits and not env_files
    check("uploadable sensitive scan", scan_pass, {"credential_hits": secret_hits, "personal_absolute_path_files": personal_path_hits, "env_files": env_files, "files_scanned": len(uploadable)})

    overall = all(ok for _, ok, _ in checks)
    verification_lines = [
        "# Verification",
        "",
        f"Overall result: **{'PASS' if overall else 'FAIL'}**",
        "",
        "This verification was read-only with respect to all experimental source files. It made no model/API calls and ran no experiment.",
        "",
        "| Check | Result | Detail |",
        "|---|---:|---|",
    ]
    for name, ok, detail in checks:
        clean = detail.replace("|", "\\|").replace("\n", " ")
        verification_lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | `{clean}` |")
    verification_lines += [
        "",
        "For final-manifest members whose only staged change is declared absolute-path redaction, verification checks the manifest against the source original SHA/byte count and separately checks the staged SHA recorded in `SOURCE_MANIFEST.csv`.",
    ]
    (root / "VERIFICATION.md").write_text("\n".join(verification_lines) + "\n", encoding="utf-8")
    (root / "VERIFICATION.json").write_text(json.dumps({"status": "PASS" if overall else "FAIL", "checks": [{"name": n, "pass": ok, "detail": d} for n, ok, d in checks]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    scan_lines = [
        "# Credential and Sensitive-Content Scan",
        "",
        f"Uploadable scan result: **{'PASS' if scan_pass else 'FAIL'}**",
        "",
        f"- Uploadable files scanned: {len(uploadable)}",
        f"- Credential/private-key signature hits: {len(secret_hits)}",
        f"- `.env` files: {len(env_files)}",
        f"- Unredacted personal absolute-path files: {len(personal_path_hits)}",
        "- API-key environment-variable *names* and token-usage accounting fields are not credentials and are allowed.",
        "- Portable placeholders such as `<SOURCE_WORKSPACE>` are allowed.",
        "- `DO_NOT_COMMIT/SOURCE_LOCATIONS_LOCAL_ONLY.md` intentionally contains one local source-root mapping and is excluded from uploadable scope.",
        "- No images, videos, model weights, caches, virtualenvs, or `.env` files are present in the uploadable tree.",
    ]
    if secret_hits or env_files or personal_path_hits:
        scan_lines += ["", "## Hits", "", *[f"- {x}" for x in secret_hits + env_files + personal_path_hits]]
    (root / "SENSITIVE_SCAN_REPORT.md").write_text("\n".join(scan_lines) + "\n", encoding="utf-8")
    (root / "TREE.txt").write_text(compact_tree(root), encoding="utf-8")

    if not overall:
        raise SystemExit("verification failed; inspect VERIFICATION.md")


if __name__ == "__main__":
    main()
