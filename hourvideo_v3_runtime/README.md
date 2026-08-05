# HourVideo V3 runtime

A source-identical runtime snapshot of the HourVideo ten-video V3 baseline.
It contains the complete code path from the no-API question projection and
ten-video selection through hierarchy construction, ASR, detector/caption
stages, R1_AV/R3_2 map-assisted retrieval, and V3 question-symmetric visual
review.

It deliberately excludes V4/V5 experiments, generated outputs, datasets,
model weights and credentials. Configure the four JSON files under
`configs/experiments/` for the school filesystem before running.

## Stages

```powershell
python scripts/experiments/run_hourvideo_r1_av_r3_2_no_api_interface_validation_v1_1.py
python scripts/experiments/run_hourvideo_ten_video_pilot_selection_v1.py
python scripts/experiments/run_hourvideo_r1_av_r3_2_ten_video_pilot_v1.py --stage prepare
python scripts/experiments/run_hourvideo_r1_av_r3_2_ten_video_pilot_v1.py --stage audio
python scripts/experiments/run_hourvideo_r1_av_r3_2_ten_video_pilot_v1.py --stage detector
python scripts/experiments/run_hourvideo_r1_av_r3_2_ten_video_pilot_v1.py --stage captions
python scripts/experiments/run_hourvideo_r1_av_r3_2_ten_video_pilot_v1.py --stage preflight
python scripts/experiments/run_hourvideo_r1_av_r3_2_ten_video_pilot_v1.py --stage live
python scripts/experiments/run_hourvideo_r1_av_r3_2_question_symmetric_visual_review_v3.py --stage preflight
python scripts/experiments/run_hourvideo_r1_av_r3_2_question_symmetric_visual_review_v3.py --stage live
```

`runtime_manifest.json` records SHA-256 identities for every copied source
file, so this snapshot can be checked against the original V3 source.
