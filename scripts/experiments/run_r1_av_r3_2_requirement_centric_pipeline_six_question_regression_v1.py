from pathlib import Path

from dotenv import load_dotenv

from experiments.r1_av_r3_2_requirement_centric_pipeline_six_question_regression_v1.core import run


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env", override=False)
    result = run(root, root / "configs/experiments/r1_av_r3_2_requirement_centric_pipeline_six_question_regression_v1.json")
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))
