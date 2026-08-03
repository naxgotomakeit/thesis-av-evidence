import argparse,json,sys
from pathlib import Path
from dotenv import load_dotenv
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"src"))
from experiments.r1_av_r3_2_review_cache_integration_canary_v1 import run
if __name__=="__main__":
 load_dotenv(ROOT/".env",override=False);p=argparse.ArgumentParser();p.add_argument("--allow-api-calls",action="store_true");a=p.parse_args();print(json.dumps(run(ROOT,ROOT/"configs/experiments/r1_av_r3_2_review_cache_integration_canary_v1.json",a.allow_api_calls),ensure_ascii=False,indent=2))
