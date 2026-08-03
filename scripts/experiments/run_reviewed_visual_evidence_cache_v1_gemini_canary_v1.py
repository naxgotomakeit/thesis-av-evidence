import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from experiments.reviewed_visual_evidence_cache_v1_gemini_canary import recompute_existing_cost, run

if __name__ == "__main__":
    load_dotenv(ROOT / ".env", override=False)
    parser=argparse.ArgumentParser(); parser.add_argument("--allow-api-calls",action="store_true"); parser.add_argument("--recompute-cost",action="store_true"); args=parser.parse_args()
    config=ROOT/"configs/experiments/reviewed_visual_evidence_cache_v1_gemini_canary_v1.json"
    result=recompute_existing_cost(ROOT,config) if args.recompute_cost else run(ROOT,config,args.allow_api_calls)
    print(json.dumps(result,ensure_ascii=False,indent=2))
