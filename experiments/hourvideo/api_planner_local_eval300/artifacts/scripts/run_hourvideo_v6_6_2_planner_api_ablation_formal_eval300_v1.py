from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from experiments.hourvideo_v6_6_2_planner_api_ablation_eval300_v1.formal_eval300 import launch
print(json.dumps(launch(ROOT,ROOT/'configs/experiments/hourvideo_v6_6_2_planner_api_ablation_formal_eval300_v1.json'),indent=2))
