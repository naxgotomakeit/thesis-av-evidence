from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ConfigurationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_profile(path: str | Path) -> dict[str, Any]:
    profile_path = Path(path).absolute()
    raw = profile_path.read_bytes()
    profile = json.loads(raw)
    if profile.get("profile") not in {
        "gens_hybrid_cap16",
        "gens_hybrid_cap16_cache_reuse_v1",
        "gens_hybrid_symmetric_mcq_cap16_v1",
    }:
        raise ConfigurationError("unrecognized GenS frozen profile")
    profile["_profile_path"] = str(profile_path)
    profile["_profile_sha256"] = hashlib.sha256(raw).hexdigest()
    profile["_config_dir"] = str(profile_path.parent)
    return profile


def resolve_profile_path(profile: dict[str, Any], value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(profile["_config_dir"]) / path
    return path.absolute()


def verify_frozen_artifacts(profile: dict[str, Any], full_manifests: bool = False) -> dict[str, str]:
    results: dict[str, str] = {}
    for section_name in ("source", "clip", "gens"):
        section = profile[section_name]
        manifest = resolve_profile_path(profile, section["sha256_manifest"])
        actual = sha256_file(manifest)
        expected = section["sha256_manifest_sha256"]
        if actual != expected:
            raise ConfigurationError(
                f"{section_name} manifest hash drift: expected {expected}, got {actual}"
            )
        results[f"{section_name}_manifest_sha256"] = actual

        if full_manifests:
            base = (
                resolve_profile_path(profile, section["local_path"])
                if "local_path" in section
                else resolve_profile_path(profile, "../source/GenS")
            )
            verify_sha256_manifest(manifest, base)

    template_path = resolve_profile_path(profile, profile["query"]["template_path"])
    actual_template = sha256_file(template_path)
    expected_template = profile["query"]["template_sha256"]
    if actual_template != expected_template:
        raise ConfigurationError(
            f"query template hash drift: expected {expected_template}, got {actual_template}"
        )
    results["query_template_sha256"] = actual_template

    if "candidate_cache" in profile:
        cache = profile["candidate_cache"]
        for key in ("manifest", "summary", "source_verification"):
            path = resolve_profile_path(profile, cache[key])
            actual = sha256_file(path)
            expected = cache[f"{key}_sha256"]
            if actual != expected:
                raise ConfigurationError(
                    f"candidate cache {key} hash drift: expected {expected}, got {actual}"
                )
            results[f"candidate_cache_{key}_sha256"] = actual
    return results


def verify_sha256_manifest(manifest: Path, base: Path) -> None:
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            expected, rel = line.split("  ", 1)
        except ValueError as exc:
            raise ConfigurationError(f"invalid manifest line {line_number}: {line!r}") from exc
        rel_path = rel[2:] if rel.startswith("./") else rel
        target = base / rel_path
        actual = sha256_file(target)
        if actual != expected:
            raise ConfigurationError(
                f"artifact hash drift for {target}: expected {expected}, got {actual}"
            )
