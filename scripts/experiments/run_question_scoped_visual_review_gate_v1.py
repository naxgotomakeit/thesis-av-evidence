from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from experiments.question_scoped_visual_review_gate_v1 import run


if __name__ == "__main__":
    result = run(ROOT, ROOT / "configs/experiments/question_scoped_visual_review_gate_v1.json")
    print(result)
