from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime_overlay_v2"
sys.path.insert(0, str(RUNTIME))

from videoseal.tools.visual_tools import env_int_first
from videoseal.utils.video.tooling import select_time_diverse_windows

flat_env = (ROOT / "flat15.env").read_text(encoding="utf-8")
smoke = (RUNTIME / "scripts/dgx/smoke_one.sh").read_text(encoding="utf-8")
visual = (RUNTIME / "videoseal/tools/visual_tools.py").read_text(encoding="utf-8")
embed = (RUNTIME / "videoseal/utils/RAG/rag_query_embed.py").read_text(encoding="utf-8")

assert re.search(r"^VISUAL_RETRIEVE_TOPK=15$", flat_env, re.M)
assignments = re.findall(r"VISUAL_RETRIEVE_TOPK=([^\s]+)", smoke)
assert assignments == ["15"], assignments
assert 'env_int_first(("VISUAL_RETRIEVE_TOPK", "SEMANTIC_RETRIEVE_TOPK"), 100)' in visual
assert "hits = query_embed(idx_dir, query, topk=k)" in visual
assert "select_time_diverse_windows(windows, k=k, min_gap_sec=min_gap)" in visual
assert "np.argpartition(-sim, k - 1)" in embed and "np.argsort(-sim[idx])" in embed

def resolve(v: str) -> int:
    os.environ["VISUAL_RETRIEVE_TOPK"] = v
    os.environ["SEMANTIC_RETRIEVE_TOPK"] = "40"
    return env_int_first(("VISUAL_RETRIEVE_TOPK", "SEMANTIC_RETRIEVE_TOPK"), 100)

assert resolve("30") == 30
assert resolve("15") == 15
windows = [(float(i * 16), float((i + 1) * 16)) for i in range(40)]
flat30 = select_time_diverse_windows(windows, k=30, min_gap_sec=15.0)
flat15 = select_time_diverse_windows(windows, k=15, min_gap_sec=15.0)
assert flat15 == flat30[:15]
assert len(flat30) == 30 and len(flat15) == 15
print("PASS: active Flat-15 trace is 15 -> k=15 -> query_embed(topk=15) -> temporal selection k=15")
