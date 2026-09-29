# Public Live example

A reproducible public Live example using a [CC BY 4.0 Wikimedia Commons video](ATTRIBUTION.md).
The runner downloads and verifies the original video into ignored `demo_runs/`;
the video itself is not stored in Git. No smoke-run results or answer keys are included.

```bash
python3 demo/run_live_example.py --question audio
python3 demo/run_live_example.py --question visual
```

The two MCQs target spoken information and a visible spatial detail, respectively.
Frame inspection remains the model's decision; it is not forced.

PREPARE runs once; the READY workspace supports multiple independent ASK questions.
Before READY, the commands above generate a PREPARE plan only. After READY, they
generate an ASK plan only. Review each plan and its costs before separately authorizing:

```bash
python3 demo/run_live_example.py --question audio --stage prepare --execute --approve-plan <PREPARE_SHA>
python3 demo/run_live_example.py --question audio
python3 demo/run_live_example.py --question audio --stage ask --execute --approve-plan <ASK_SHA>
```

Repeat the ASK planning/approval steps with `--question visual` for the other MCQ.
Each approval is stage- and input-specific. `ANTHROPIC_API_KEY` is required through
the environment for paid execution; no key flag is supported. The wrapper uses
the invoking Python interpreter and delegates to the existing scripts without
changing their safeguards or retry policy.

Install the [optional Whisper dependencies](../README.md#optional-speech-during-preparation)
first. Plan-only may download the video and extract frames, but never loads
Whisper or calls an API. Approved preparation may download missing Whisper weights.
This uses `PORTABLE_API` / `PORTABLE_AV_EXTENSION`, not exact thesis preprocessing.
ASK reads ASR text in the map; it does not listen to raw audio. See the
[usage guide](../../docs/USAGE.md) for dependencies, fidelity and cost boundaries.
