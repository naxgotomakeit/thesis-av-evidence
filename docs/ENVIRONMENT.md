# Portable demo environment

## Install

Recommended: **Python 3.11** (tested: 3.11.11 on macOS ARM64).
Install system `ffmpeg` and `ffprobe` on PATH (tested: 8.0.1).
From the repository root, preferably in a new virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python3.11 -m pip install -r requirements.txt
python3.11 -m pip check
python3.11 -B -m unittest discover -s demo -p 'test_*.py'
```

The activation command above is for POSIX shells. No editable-package installation
or PYTHONPATH setting is needed: invoke the scripts from the repository root.
The four pinned Python packages cover optional Whisper AV preparation; they are
not required for either replay. No GPU is required. The current Whisper device
policy is CUDA when available, otherwise CPU; it does not select Apple MPS.

`ANTHROPIC_API_KEY` is required only for paid/live API execution, through the
environment. Plans need no key and do not call models. Whisper-small weights
may download on first approved audio execution into the default Whisper cache
(`~/.cache/whisper`, or beneath `XDG_CACHE_HOME` when set). They are not installed
by pip, bundled with this repository, or downloaded by the tests.

## Dependency audit

| Package | Tested pin | Why it is present |
|---|---|---|
| `torch` | 2.9.1 | Local Whisper model loading/inference and CPU/CUDA selection |
| `numpy` | 2.2.6 | Whisper numerical arrays and soundfile waveform decoding |
| `openai-whisper` | 20250625 | `whisper.load_model` and timestamped English transcription |
| `soundfile` | 0.14.0 | Decode extracted mono 16 kHz WAV to float32 waveform |

`soundfile` requires libsndfile (tested: 1.2.2). Supported wheels may bundle it;
source/platform-specific installations may need a system libsndfile installation.

Audited entrypoints: `demo/run_demo.py`, `prepare.py`, `ask.py`, `audio.py`,
`open_direct.py`, and `run_live_example.py`; shared `portable.py` and
`replay_direct.py`; and the imported preserved `direct_api_v1` / `direct_api_prep`
module closure. The pure preserved map-conversion function is loaded separately
through AST extraction: unrelated historical module imports are not executed.

- Replay: Python standard library only; no ffmpeg, media, model or API needed.
- Visual-only PREPARE: standard library plus ffmpeg/ffprobe; captions and Organizer
  use `urllib.request` for API requests after approval.
- MCQ/open ASK: standard library and preserved local Direct modules. The injected
  client bypasses the preserved provider's optional Anthropic SDK import. Images
  are read as existing JPEG bytes; no Pillow decoder is used.
- Whisper PREPARE: additionally uses the four packages above. Required transitives
  include numba/llvmlite, tiktoken, more-itertools, tqdm and cffi. The installed
  tiktoken package requires `requests` and `regex`; requests is therefore an
  indirect dependency even though portable API transport does not use it.
- Public runner: standard-library download/verification and subprocess delegation;
  it uses the same environment as its child PREPARE/ASK commands.

Do not add `anthropic`, `httpx`, Pillow, pydantic or transformers for these paths.
They are not required by the portable implementation. Tests use `unittest` and
mock model/provider execution; pytest is not required.

## Reproducibility boundaries

The requirements file pins tested top-level versions, not every transitive
dependency or platform wheel. No entire conda environment was copied. The tested
environment also had numba 0.67.0, llvmlite 0.49.0 and tiktoken 0.14.0; these remain
pip-resolved transitives, not a claimed universal lock. Record a platform-specific
resolved environment for deployment if byte-identical dependency resolution is
needed. No fresh full-package installation on Linux or Windows is claimed here.

On Linux x86_64, default Torch 2.9.1 package metadata includes NVIDIA CUDA runtime
dependencies, and Whisper declares Triton. These are upstream distribution
dependencies, not a requirement for thesis-faithful GPU reproduction. For a
CPU-only deployment, select a compatible official CPU Torch distribution before
installing the remaining requirements; do not silently replace the tested version.
Platform wheels and native libraries need validation on the target host.
For GPU deployments, install a PyTorch build compatible with the target CUDA/runtime rather than assuming the pinned torch wheel is universally appropriate.

## Thesis-faithful deployments: not fully packaged

This environment supports the current `PORTABLE_API` profile, optional
`PORTABLE_AV_EXTENSION`, and `PORTABLE_OPEN_ENDED_EXTENSION`. It does not reproduce
the historical Qwen caption model, complete embedding/index construction or
School/DGX/Myriad deployments. Their CUDA, model-server, weights and dependency
contracts require a separate audit. No thesis-faithful requirements file is
provided. See [usage](USAGE.md) and [reproduction boundaries](REPRODUCTION.md).
