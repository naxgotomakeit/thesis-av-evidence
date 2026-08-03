from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


ReviewStatus = Literal["confirmed", "probable", "uncertain", "not_supported"]
RequirementEffect = Literal["supports_requirement", "inconclusive"]


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_scope(scope: str) -> str:
    value = re.sub(r"[^a-z0-9_]+", "_", scope.strip().lower()).strip("_")
    if not value or value != scope:
        raise ValueError("scope must already be canonical lower_snake_case")
    return value


@dataclass(frozen=True)
class ReviewRecord:
    record_version: str
    review_contract_version: str
    source_type: str
    fine_id: str
    image_sha256: str
    image_size_bytes: int
    timestamp_sec: float
    review_scope: str
    status: ReviewStatus
    requirement_effect: RequirementEffect
    direct_visual_support: bool
    finding: str
    confidence: Literal["high", "medium", "low", "none"]
    supporting_image_ids: tuple[str, ...]
    model_id: str
    model_config_hash: str
    synthetic_smoke_record: bool = False

    def validate(self) -> None:
        if self.record_version != "reviewed_visual_evidence_cache_record_v1":
            raise ValueError("invalid record version")
        if self.source_type != "reviewed_visual_frame":
            raise ValueError("invalid source type")
        normalize_scope(self.review_scope)
        if not re.fullmatch(r"[0-9a-f]{64}", self.image_sha256):
            raise ValueError("invalid image sha256")
        if self.image_size_bytes <= 0 or self.timestamp_sec < 0:
            raise ValueError("invalid image metadata")
        if not self.finding.strip():
            raise ValueError("empty finding")
        if self.fine_id not in self.supporting_image_ids:
            raise ValueError("fine_id absent from supporting_image_ids")
        if self.requirement_effect == "supports_requirement" and not self.direct_visual_support:
            raise ValueError("supporting result lacks direct visual support")
        if self.status == "not_supported" and self.requirement_effect == "supports_requirement":
            raise ValueError("not_supported cannot positively support a requirement")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value["supporting_image_ids"] = list(self.supporting_image_ids)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReviewRecord":
        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            raise ValueError(f"record fields mismatch: {sorted(set(value) ^ expected)}")
        copied = dict(value)
        copied["supporting_image_ids"] = tuple(copied["supporting_image_ids"])
        record = cls(**copied)
        record.validate()
        return record


class ImmutableVisualReviewCache:
    def __init__(self, root: Path):
        self.root = root

    def record_path(self, image_sha256: str, contract_version: str, scope: str) -> Path:
        normalize_scope(scope)
        contract_hash = hashlib.sha256(contract_version.encode("utf-8")).hexdigest()[:16]
        return self.root / "records" / image_sha256 / contract_hash / f"{scope}.json"

    def lookup(self, image_path: Path, contract_version: str, scopes: list[str]) -> dict[str, Any]:
        image_sha = sha256_file(image_path)
        records, missing = {}, []
        for scope in scopes:
            path = self.record_path(image_sha, contract_version, scope)
            if not path.is_file():
                missing.append(scope)
                continue
            record = ReviewRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if record.image_sha256 != image_sha or record.review_contract_version != contract_version or record.review_scope != scope:
                raise ValueError("cache provenance mismatch")
            records[scope] = record.to_dict()
        status = "hit" if not missing else "miss" if not records else "partial_hit"
        return {"status": status, "image_sha256": image_sha, "records": records, "missing_scopes": missing}

    def store(self, record: ReviewRecord) -> dict[str, Any]:
        payload = record.to_dict()
        path = self.record_path(record.image_sha256, record.review_contract_version, record.review_scope)
        data = canonical_bytes(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            disposition = "created"
        except FileExistsError:
            existing = path.read_bytes()
            if existing != data:
                raise FileExistsError("immutable cache collision: existing record differs")
            disposition = "reused_identical"
        return {"disposition": disposition, "path": str(path), "record_sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


class LayeredVisualReviewCache:
    """Read immutable records from seed layers and write only to a new layer."""
    def __init__(self, readable_roots: list[Path], writable_root: Path):
        self.readers = [ImmutableVisualReviewCache(path) for path in readable_roots]
        self.writer = ImmutableVisualReviewCache(writable_root)

    def lookup(self, image_path: Path, contract_version: str, scopes: list[str]) -> dict[str, Any]:
        image_sha = sha256_file(image_path); records: dict[str, Any] = {}; sources = {}; missing = []
        for scope in scopes:
            found = []
            for reader in [self.writer, *self.readers]:
                result = reader.lookup(image_path, contract_version, [scope])
                if result["status"] == "hit": found.append(result["records"][scope])
            if not found:
                missing.append(scope); continue
            canonical = {canonical_bytes(item) for item in found}
            if len(canonical) != 1:
                raise ValueError("conflicting immutable cache layers")
            records[scope] = found[0]; sources[scope] = "writable_or_seed_layer"
        status = "hit" if not missing else "miss" if not records else "partial_hit"
        return {"status": status, "image_sha256": image_sha, "records": records, "record_sources": sources, "missing_scopes": missing}

    def store(self, record: ReviewRecord) -> dict[str, Any]:
        return self.writer.store(record)


def update_requirement(requirements: list[dict[str, Any]], target_requirement_id: str,
                       record: ReviewRecord) -> list[dict[str, Any]]:
    record.validate()
    ids = [row["requirement_id"] for row in requirements]
    if ids.count(target_requirement_id) != 1:
        raise ValueError("target requirement must exist exactly once")
    output = []
    for row in requirements:
        if row["requirement_id"] != target_requirement_id:
            output.append(dict(row))
            continue
        updated = dict(row)
        if record.requirement_effect == "supports_requirement":
            updated.update({
                "status": "supported", "direct_support": True,
                "supporting_evidence_ids": [f"reviewed_visual::{record.fine_id}::{record.review_scope}"],
                "finding": record.finding, "evidence_type": "reviewed_visual_frame",
                "review_scope": record.review_scope,
            })
        else:
            updated.update({
                "status": "uncertain", "direct_support": False,
                "supporting_evidence_ids": [f"reviewed_visual::{record.fine_id}::{record.review_scope}"],
                "finding": record.finding, "evidence_type": "reviewed_visual_frame",
                "review_scope": record.review_scope,
            })
        output.append(updated)
    return output


def recompute_gate_status(requirements: list[dict[str, Any]]) -> str:
    critical = [row for row in requirements if row.get("answer_critical", True)]
    if any(row.get("status") == "conflicted" for row in critical):
        return "conflicted"
    supported = [row for row in critical if row.get("status") == "supported" and row.get("direct_support")]
    if critical and len(supported) == len(critical):
        return "answer_ready"
    if supported:
        return "provisional"
    return "unresolved"
