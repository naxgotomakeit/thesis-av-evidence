from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
FLAT_TRAJECTORY = Path(
    "/home/naxucl/data/HourVideo/videoseal_original/runs_dgx_eval300_v1/"
    "115774b6-534d-444f-b7aa-d1b834eb0ee7/20260814T024123Z-ftil/trajectory.json"
)
EXPECTED_SHA256 = "71346effd5acd8d567e16921316e0f6cb0e94b422f42c4d55edb24a86d4a1863"

os.environ["RETRIEVAL_BACKEND"] = "dense_semantic_beam_b"
os.environ["DENSE_BEAM_B"] = "15"

from videoseal.tools.tool_map import build_tool_map  # noqa: E402


flat_text = json.loads(FLAT_TRAJECTORY.read_text())["tools_schema"]
tool_map = build_tool_map()
# The retrieve schema property has no instance state. Bypass provider initialization
# so this parity check cannot initialize or call a model/API.
retrieve_cls = tool_map["visual_retrieve"]
h_schema = [retrieve_cls.__new__(retrieve_cls).json, tool_map["visual_inspect"]().json]
h_text = json.dumps(h_schema, ensure_ascii=False)
flat_schema = json.loads(flat_text)

result = {
    "status": "PASS" if flat_text == h_text else "FAIL",
    "flat_trajectory": str(FLAT_TRAJECTORY),
    "flat_schema_sha256": hashlib.sha256(flat_text.encode()).hexdigest(),
    "h_schema_sha256": hashlib.sha256(h_text.encode()).hexdigest(),
    "expected_schema_sha256": EXPECTED_SHA256,
    "field_equal": flat_schema == h_schema,
    "byte_equal": flat_text == h_text,
    "backend": os.environ["RETRIEVAL_BACKEND"],
    "b": int(os.environ["DENSE_BEAM_B"]),
    "flat_schema": flat_schema,
    "h_schema": h_schema,
}
(HERE / "schema_parity_report.json").write_text(json.dumps(result, indent=2) + "\n")
if not (result["field_equal"] and result["byte_equal"]):
    raise SystemExit("schema parity failed")
if result["h_schema_sha256"] != EXPECTED_SHA256:
    raise SystemExit("schema SHA mismatch")
print(json.dumps({k: result[k] for k in ("status", "flat_schema_sha256", "h_schema_sha256", "field_equal", "byte_equal", "backend", "b")}, indent=2))
