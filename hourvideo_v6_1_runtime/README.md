# HourVideo V6.1 runtime

A source-identical runtime snapshot of the frozen V6.1 coarse-locked
claim/audit loop, built from git blobs (not the working tree) at two
pinned revisions -- see `runtime_manifest.json` for exactly which
revision each file came from and its SHA-256.

It deliberately excludes V6.2 (still under test, not frozen), V4/V5
experiments, generated outputs, datasets, model weights and credentials.

## Before running anything

Every JSON file under `configs/experiments/` has machine-specific paths
baked in (D:/Thesisdata/HourVideo/..., this machine's venv/model-cache
paths, etc.) -- rewrite them for the school filesystem first.

`configs/experiments/hourvideo_ten_video_pilot_selection_v1.json` also
has a `selected_cases` list from the original 10-video pilot on this
machine. Selection was intentionally NOT run as part of building this
snapshot, and no video list is authoritative here -- replace
`selected_cases` (and `hourvideo_root`, `frame_audit_path`,
`video_audit_path`) with your own choices, validated against whatever
HourVideo video files actually exist on the school machine, before
running the selection stage.

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
python scripts/experiments/run_hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6.py --config configs/experiments/hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6_1.json --stage preflight
python scripts/experiments/run_hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6.py --config configs/experiments/hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6_1.json --stage live
python scripts/experiments/run_hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6.py --config configs/experiments/hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6_1.json --stage evaluate
```

`runtime_manifest.json` records the source revision and SHA-256 for every
copied file, so this snapshot can be checked against the original source.
