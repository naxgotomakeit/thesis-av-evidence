# Linux / School GPU Setup

This setup supports the frozen visual-only v1 pipeline and the deliberately
simple EgoPolice B0 baseline. It does not place data, frames, caches, outputs,
or model weights inside Git.

## Recommended layout

```text
~/thesis/
  thesis-av-evidence/
  models/
    Qwen2.5-VL-3B-Instruct/
  data/
    EgoPolice_1.0.0/
      mcq_1s.json
      mcq_10s.json
      mcq_60s.json
      video.txt
      videos/
  outputs/
```

Python 3.10 is recommended because that is the validated Windows environment.

## Clone and create the environment

```bash
mkdir -p ~/thesis && cd ~/thesis
git clone https://github.com/naxgotomakeit/thesis-av-evidence.git
cd thesis-av-evidence
git switch dev/thesis-av
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
nvidia-smi
```

Do not assume the server CUDA/driver stack matches Windows. PyTorch wheels
bundle a CUDA runtime but still require a sufficiently new NVIDIA driver. If
`nvidia-smi` shows a driver compatible with CUDA 12.1 or newer, reproduce the
validated environment first:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

If the server driver is older, or the school requires a different module,
select the matching command from the official PyTorch installation selector
instead of forcing `cu121`. Verify the install before the project dependencies:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
pip install -r requirements.txt
sudo apt-get install ffmpeg  # or use the school's module/package manager
```

`ffmpeg` and `ffprobe` must both be on `PATH`, or their exact paths must be
provided through environment variables.

## Download the model outside Git

After any required Hugging Face authentication, download directly to the model
directory:

```bash
hf download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir ~/thesis/models/Qwen2.5-VL-3B-Instruct
```

The checkpoint must never be copied into the repository.

## Materialize only the frozen EgoPolice pool

The supplied metadata maps the frozen videos to Vimeo or Google Drive sources.
The downloader is a dry run unless `--execute` is passed and never requests a
video outside the frozen manifest:

```bash
python scripts/data/download_egopolice_50videos.py \
  --manifest config/data/egopolice_50videos.json \
  --data-root ~/thesis/data/EgoPolice_1.0.0 \
  --execute
```

If the source requires an authenticated browser session, add (for example)
`--cookies-from-browser chrome`. Do not store cookies in the repository. Refresh
the ffprobe availability audit without changing the frozen selection:

```bash
python scripts/data/prepare_egopolice_50videos.py \
  --data-root ~/thesis/data/EgoPolice_1.0.0 \
  --manifest config/data/egopolice_50videos.json \
  --summary-csv outputs/data_audit/egopolice_50videos_summary.csv \
  --missing-json outputs/data_audit/egopolice_50videos_missing.json
```

## Configure portable paths

```bash
export DATA_ROOT=~/thesis/data/EgoPolice_1.0.0
export MODEL_PATH=~/thesis/models/Qwen2.5-VL-3B-Instruct
export OUTPUT_ROOT=~/thesis/outputs
export FFMPEG_PATH=ffmpeg
export FFPROBE_PATH=ffprobe

python scripts/check_environment.py
```

Alternatively set `MODEL_ROOT=~/thesis/models`; B0 will append the frozen
checkpoint directory name. CLI arguments override environment variables.

## B0 smoke and future staged evaluation

Run one question whose source video has been downloaded:

```bash
python scripts/baselines/smoke_egopolice_b0.py \
  --case-id <QUESTION_ID> \
  --output "$OUTPUT_ROOT/baselines/egopolice_b0/smoke.jsonl"
```

After that succeeds, the current safety-gated runner can process at most five
questions per invocation:

```bash
python scripts/baselines/run_egopolice_b0.py --limit 5 \
  --subset-manifest config/data/egopolice_50videos.json \
  --output "$OUTPUT_ROOT/baselines/egopolice_b0/dev_5.jsonl"
```

The frozen 50-video pool is defined by
`config/data/egopolice_50videos.json`. Full-pool execution remains intentionally
disabled in this checkpoint; it must not be used for threshold tuning.

## Windows equivalents

```powershell
$env:DATA_ROOT = 'D:\ThesisData\EgoPolice_1.0.0'
$env:MODEL_PATH = 'C:\Users\72977\msc_thesis\models\Qwen2.5-VL-3B-Instruct'
$env:OUTPUT_ROOT = 'C:\Users\72977\msc_thesis\outputs'
$env:FFMPEG_PATH = '<PATH-TO-ffmpeg.exe>'
$env:FFPROBE_PATH = '<PATH-TO-ffprobe.exe>'

C:\Users\72977\miniforge3\envs\thesis_av\python.exe scripts\check_environment.py
C:\Users\72977\miniforge3\envs\thesis_av\python.exe scripts\baselines\smoke_egopolice_b0.py --case-id 60s_386
```

The Windows 6 GB RTX 3060 requires the configured bitsandbytes NF4 path for the
approximately 7.51 GB Qwen checkpoint. The runner fails preflight if
bitsandbytes or Accelerate is unavailable rather than silently changing the
baseline.
