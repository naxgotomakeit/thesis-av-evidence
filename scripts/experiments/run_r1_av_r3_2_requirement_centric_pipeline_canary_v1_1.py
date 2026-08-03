from __future__ import annotations
import json,sys
from pathlib import Path
from dotenv import load_dotenv
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from experiments.r1_av_r3_2_requirement_centric_pipeline_canary_v1_1 import run
if __name__=='__main__':load_dotenv(ROOT/'.env',override=False);print(json.dumps(run(ROOT,ROOT/'configs/experiments/r1_av_r3_2_requirement_centric_pipeline_canary_v1_1.json'),ensure_ascii=False,indent=2))
