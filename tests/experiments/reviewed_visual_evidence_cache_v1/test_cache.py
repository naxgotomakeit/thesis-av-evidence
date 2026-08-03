from pathlib import Path

import pytest

from experiments.reviewed_visual_evidence_cache_v1.cache import (
    ImmutableVisualReviewCache, ReviewRecord, recompute_gate_status,
    sha256_file, update_requirement,
)


def record(image: Path, scope: str = "weapon_visibility", finding: str = "visible") -> ReviewRecord:
    return ReviewRecord(
        "reviewed_visual_evidence_cache_record_v1", "contract-v1", "reviewed_visual_frame",
        "F01", sha256_file(image), image.stat().st_size, 1.0, scope, "confirmed",
        "supports_requirement", True, finding, "high", ("F01",), "mock", "0" * 64, True,
    )


def test_miss_store_hit_partial_and_scope_separation(tmp_path):
    image = tmp_path / "x.jpg"; image.write_bytes(b"image")
    cache = ImmutableVisualReviewCache(tmp_path / "cache")
    assert cache.lookup(image, "contract-v1", ["weapon_visibility"])["status"] == "miss"
    cache.store(record(image))
    assert cache.lookup(image, "contract-v1", ["weapon_visibility"])["status"] == "hit"
    assert cache.lookup(image, "contract-v1", ["weapon_visibility", "visible_injury"])["status"] == "partial_hit"


def test_immutable_collision_rejected(tmp_path):
    image = tmp_path / "x.jpg"; image.write_bytes(b"image")
    cache = ImmutableVisualReviewCache(tmp_path / "cache")
    cache.store(record(image))
    assert cache.store(record(image))["disposition"] == "reused_identical"
    with pytest.raises(FileExistsError): cache.store(record(image, finding="different"))


def test_contract_version_and_image_bytes_invalidate(tmp_path):
    image = tmp_path / "x.jpg"; image.write_bytes(b"image")
    cache = ImmutableVisualReviewCache(tmp_path / "cache"); cache.store(record(image))
    assert cache.lookup(image, "contract-v2", ["weapon_visibility"])["status"] == "miss"
    image.write_bytes(b"changed")
    assert cache.lookup(image, "contract-v1", ["weapon_visibility"])["status"] == "miss"


def test_only_target_requirement_updates(tmp_path):
    image = tmp_path / "x.jpg"; image.write_bytes(b"image")
    before = [
        {"requirement_id": "r1", "answer_critical": True, "status": "uncertain", "direct_support": False},
        {"requirement_id": "r2", "answer_critical": False, "status": "not_found", "direct_support": False},
    ]
    after = update_requirement(before, "r1", record(image))
    assert after[1] == before[1]
    assert recompute_gate_status(before) == "unresolved"
    assert recompute_gate_status(after) == "answer_ready"
