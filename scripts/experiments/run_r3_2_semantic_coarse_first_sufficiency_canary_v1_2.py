import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from experiments.r3_2_semantic_coarse_first_sufficiency_canary_v1_2.core import run
if __name__=='__main__':print(json.dumps(run(ROOT,ROOT/'configs/experiments/r3_2_semantic_coarse_first_sufficiency_canary_v1_2.json'),ensure_ascii=False,indent=2))

