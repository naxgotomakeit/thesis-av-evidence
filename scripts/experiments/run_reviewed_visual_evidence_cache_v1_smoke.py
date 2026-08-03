from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from experiments.reviewed_visual_evidence_cache_v1.smoke import run


if __name__ == "__main__":
    print(run(ROOT, ROOT / "configs/experiments/reviewed_visual_evidence_cache_v1_smoke.json"))
