# HourVideo frame embeddings 3-way v1 resume

## Current status

- Overall status: `blocked_before_siglip_offline_smoke`
- Frozen frame manifest: 50 videos / 141,550 frames
- DINOv2: `completed_preserved`
- SigLIP: full encoding not started
- C-RADIOv4: full encoding not started

No frame extraction, retrieval, captioning, QA, staging, or Git commit was run.

## DINOv2 preservation

All 50 DINOv2 NPZ shards and their 50 metadata JSON files were reloaded and
validated against the frozen frame manifest. The validation checked exact frame
IDs and order, 384-dimensional float16 embeddings, finite/non-zero values, and
the recorded SHA-256 values. The frozen completion manifest is:

`dino_v2/frozen_completion_manifest.json`

Only the stale `dino_v2/progress.json` status metadata was finalized. No DINOv2
model was initialized and no DINOv2 shard was rewritten.

## SigLIP cache blocker

The configured cache is:

`D:\ThesisData\HourVideo\model_cache`

It currently contains the C-RADIO torch hub material but no complete
`google/siglip-base-patch16-224` Hugging Face snapshot. In particular,
`preprocessor_config.json` is absent. A strict `local_files_only=True` check
therefore fails clearly and does not fall back online.

This execution environment cannot write to that external data root. Run the
following once in the `thesis_av` environment:

```powershell
C:\Users\72977\miniforge3\envs\thesis_av\python.exe `
  scripts\data\run_hourvideo_frame_embeddings_3way_v1.py `
  --models siglip `
  --prepare-siglip-cache `
  --offline-siglip-check-only
```

After it passes, inspect resume state without loading a model:

```powershell
C:\Users\72977\miniforge3\envs\thesis_av\python.exe `
  scripts\data\run_hourvideo_frame_embeddings_3way_v1.py `
  --models siglip c_radio_v4 `
  --dry-run
```

Then start the sequential background run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  scripts\data\run_hourvideo_frame_embeddings_3way_v1_background.ps1 `
  -Models siglip,c_radio_v4
```

The parent runner launches one child process for SigLIP and, only after it exits
successfully, a new child process for C-RADIOv4. DINOv2 is not initialized.

## Resume implementation

- `--models` accepts any supported model subset.
- `--dry-run` reports completed and pending shards.
- Complete shards are verified and skipped.
- Corrupt or incomplete shards are rejected and may be recomputed only for the
  selected model.
- NPZ publication is atomic.
- Model progress, summary, failed-frame log, and completion manifest are
  separate.
- Background success and failure both write a non-empty exit code and structured
  status.
- C-RADIOv4 uses batch size 1, 512x512 inputs, no spatial outputs, the actual
  `dino_v3_7b` adaptor, and the stable `dino_v3_summary` alias.
- After two new C-RADIO shards, `early_runtime_estimate.json` records measured
  decode, preprocessing, forward, total throughput, VRAM, and remaining ETA.

## Tests

The focused resume suite reports `35 passed`.

Final three-way validation has not run because SigLIP and C-RADIO full shards do
not yet exist. The experiment must not be described as `completed_validated`.
