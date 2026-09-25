from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class AccessViolation(RuntimeError):
    pass


def lexical_path(path: str | Path) -> str:
    """Normalize without resolving symlinks or touching the filesystem."""
    return os.path.abspath(os.path.normpath(os.fspath(path)))


class AccessPolicy:
    def __init__(
        self,
        selector_path: str,
        video_paths: list[str],
        model_roots: list[str],
        cache_roots: list[str],
        output_roots: list[str],
        forbidden_prefix: str,
    ) -> None:
        self.selector_path = lexical_path(selector_path)
        self.video_paths = {lexical_path(path) for path in video_paths}
        self.model_roots = tuple(lexical_path(path) for path in model_roots)
        self.cache_roots = tuple(lexical_path(path) for path in cache_roots)
        self.output_roots = tuple(lexical_path(path) for path in output_roots)
        self.forbidden_prefix = lexical_path(forbidden_prefix)

    @classmethod
    def from_allowlist(
        cls,
        allowlist_path: str | Path,
        model_roots: list[str],
        output_roots: list[str],
        cache_roots: list[str] | None = None,
    ) -> "AccessPolicy":
        data = json.loads(Path(allowlist_path).read_text(encoding="utf-8"))
        return cls(
            selector_path=data["selector"]["path"],
            video_paths=[item["path"] for item in data["videos"]],
            model_roots=model_roots,
            cache_roots=cache_roots or [],
            output_roots=output_roots,
            forbidden_prefix=data["forbidden_prefix"],
        )

    def _under(self, path: str, root: str) -> bool:
        return path == root or path.startswith(root + os.sep)

    def assert_read_allowed(self, path: str | Path) -> str:
        candidate = lexical_path(path)
        if self._under(candidate, self.forbidden_prefix):
            raise AccessViolation("private evaluation path is forbidden")
        if candidate == self.selector_path or candidate in self.video_paths:
            return candidate
        if any(self._under(candidate, root) for root in self.model_roots):
            return candidate
        if any(self._under(candidate, root) for root in self.cache_roots):
            return candidate
        if any(self._under(candidate, root) for root in self.output_roots):
            return candidate
        raise AccessViolation(f"read path is outside the positive allowlist: {candidate}")

    def assert_write_allowed(self, path: str | Path) -> str:
        candidate = lexical_path(path)
        if any(self._under(candidate, root) for root in self.output_roots):
            return candidate
        raise AccessViolation(f"write path is outside experiment outputs: {candidate}")

    def video_path_for_id(self, allowlist_path: str | Path, video_id: str) -> str:
        data = json.loads(Path(allowlist_path).read_text(encoding="utf-8"))
        matches = [item["path"] for item in data["videos"] if item["video_id"] == video_id]
        if len(matches) != 1:
            raise AccessViolation(f"video_id must map to exactly one allowlisted video: {video_id}")
        return self.assert_read_allowed(matches[0])


def reject_forbidden_keys(value: Any, forbidden: set[str], location: str = "record") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in forbidden:
                raise AccessViolation(f"forbidden field {key!r} at {location}")
            reject_forbidden_keys(child, forbidden, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_forbidden_keys(child, forbidden, f"{location}[{index}]")
