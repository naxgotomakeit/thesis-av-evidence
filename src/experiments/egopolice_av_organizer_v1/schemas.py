from __future__ import annotations

import json
from typing import Any

from .config import SCHEMA_PATH


def frozen_schemas() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
