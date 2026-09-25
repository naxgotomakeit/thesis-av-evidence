from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .access import AccessPolicy, AccessViolation, reject_forbidden_keys
from .config import sha256_file


OPTION_KEYS = ("A", "B", "C", "D", "E")


def normalize_field(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("question and option values must be strings")
    return re.sub(r"\s+", " ", value, flags=re.UNICODE).strip()


def build_retrieval_query(record: dict[str, Any], template: str) -> str:
    question = normalize_field(record["question"])
    options = record["options"]
    if not isinstance(options, dict) or tuple(options.keys()) != OPTION_KEYS:
        raise ValueError("options must contain exactly A, B, C, D, E in order")
    values = {f"option_{key}": normalize_field(options[key]) for key in OPTION_KEYS}
    return template.format(question=question, **values)


def query_sha256(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def uid_order_sha256(records: list[dict[str, Any]]) -> str:
    payload = "".join(f"{record['qa_uid']}\n" for record in records).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_selector(profile: dict[str, Any], policy: AccessPolicy) -> list[dict[str, Any]]:
    config = profile["inputs"]
    selector_path = Path(policy.assert_read_allowed(config["selector"]))
    actual_sha = sha256_file(selector_path)
    if actual_sha != config["selector_sha256"]:
        raise AccessViolation(
            f"selector hash drift: expected {config['selector_sha256']}, got {actual_sha}"
        )

    allowed_fields = set(config["allowed_selector_fields"])
    forbidden = {field.lower() for field in config["forbidden_fields"]}
    records: list[dict[str, Any]] = []
    with selector_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            reject_forbidden_keys(record, forbidden, f"line {line_number}")
            if set(record) != allowed_fields:
                raise AccessViolation(
                    f"line {line_number} fields differ from frozen public selector schema"
                )
            if not isinstance(record["options"], dict) or tuple(record["options"].keys()) != OPTION_KEYS:
                raise AccessViolation(f"line {line_number} options must be ordered A-E")
            records.append(record)

    if len(records) != config["uid_count"]:
        raise AccessViolation(f"selector row count is {len(records)}, expected {config['uid_count']}")
    uids = [record["qa_uid"] for record in records]
    if len(set(uids)) != len(uids):
        raise AccessViolation("selector contains duplicate qa_uid values")
    actual_order_hash = uid_order_sha256(records)
    if actual_order_hash != config["uid_order_sha256"]:
        raise AccessViolation(
            f"selector UID order drift: expected {config['uid_order_sha256']}, got {actual_order_hash}"
        )
    return records


def load_template(profile: dict[str, Any]) -> str:
    path = Path(profile["_config_dir"]) / profile["query"]["template_path"]
    actual = sha256_file(path)
    expected = profile["query"]["template_sha256"]
    if actual != expected:
        raise ValueError(f"query template hash drift: expected {expected}, got {actual}")
    return path.read_text(encoding="utf-8")

