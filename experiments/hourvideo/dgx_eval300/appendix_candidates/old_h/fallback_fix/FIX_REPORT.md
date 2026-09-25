# Eval300 full-video fallback fix

## Scope

This is a code-and-test-only correction. No model, GPU, API, retry, or formal evaluation was started, and no Eval300 artifact was modified.

## Root cause and local correction

The forced fallback previously emitted `[00:00:00, ceil(duration)]`. For a fractional duration this endpoint exceeded the decoder-reported duration, so the shared ordinary-window normalizer treated the request as an out-of-bounds Inspector window and replaced it with the final 15 seconds. The 64-frame budget then yielded one frame.

Only `videoseal/agents/tool_agent.py` was changed. The fallback now formats the real duration to non-overshooting microsecond precision and derives its sampling FPS from that exact serialized interval. A one-ULP upward adjustment prevents floating-point truncation from turning 64 into 63. The shared `normalize_window_min_width` implementation was not changed.

## Verification

- Integer duration `1800.0`: start 0, full interval retained, 64 uniform timestamps.
- Fractional duration `4396.83`: start 0, endpoint equals the real duration at microsecond precision, 64 uniform timestamps.
- Ordinary out-of-bounds Inspector window: still clamps to the final 15 seconds (`85.25` to `100.25`).
- Syntax/import compilation passed in the frozen runner environment.
- Unit tests: 3 passed.

## Affected attempts

The accompanying TSV contains 13 affected hierarchical attempts: H-6=8, H-15=3, H-30=2. They are a future targeted bug-correction rerun list only. Original attempts must remain immutable, and no rerun was performed in this task.
