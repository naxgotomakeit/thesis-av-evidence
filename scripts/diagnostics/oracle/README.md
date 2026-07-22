# EgoPolice short-clip Oracle diagnostic

This runner is isolated from B0. It sends each metadata-defined ground-truth
interval (`[start, end)`) to Qwen as a native video input; it does not call the
long-video uniform midpoint sampler.

Run the fixed five-question diagnostic with one BF16, unquantized model load:

```bash
python scripts/diagnostics/oracle/run_egopolice_oracle.py \
  --data-root /cs/student/project_msc/2025/rai/xinanx01/msc_thesis/data/EgoPolice_1.0.0 \
  --model-path /cs/student/project_msc/2025/rai/xinanx01/msc_thesis/models/Qwen2.5-VL-7B-Instruct
```

The default duration plan is `1s, 10s, 10s, 60s, 1s`. Within each duration
class the runner takes the first metadata-ordered question whose exact interval
can be materialized from an available source video. Selection never consults
the answer or model output. Use `--preflight-only` to print the resolved cases
without importing or loading the model.

The JSON result is written to
`outputs/diagnostics/oracle/egopolice_oracle_5.json` by default.
