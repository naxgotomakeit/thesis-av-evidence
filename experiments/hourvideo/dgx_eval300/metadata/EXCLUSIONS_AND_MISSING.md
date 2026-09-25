# Exclusions and missing items

## Deliberately excluded

- All videos and extracted frames.
- Model weights, checkpoints, tokenizer/model caches, and vector arrays/shards.
- Runtime-generated cache directories, `__pycache__`, `.pyc`, process IDs, and unrelated third-party source trees.
- Credential files, especially the referenced `.env.embedding`; no credential file was opened or copied.
- Planner/Inspector/memory service logs that are not needed for per-question result verification.
- Full first-pass raw trajectory trees. The staging retains each condition's final 300-row per-question result, retry raw attempts, runner logs, and summaries.
- The 40 MB Old-H corrected `final_selected_attempts.jsonl`; Old H is retained only as an explicitly excluded appendix candidate with configs, manifests, summaries, and reports.

## Not available as a self-contained upload asset

- Video media and frame evidence required to re-run the experiments.
- Local model weights for Qwen3-8B, Qwen2.5-VL-7B-Instruct, and SigLIP.
- Dense Coarse/Medium embedding arrays and Fine frame-vector shards; their build reports, code, source lineage, model manifests, and validation summaries are retained.
- A formal local R1/R3 paired-comparison bundle; it was not located in the DGX-only inventory and is outside the five accepted conditions.
- Flat caption-build source-to-output SHA binding remains historically unconfirmed; the accepted downstream runtime and materialized-result evidence are retained.

No school server was accessed. School-origin paths may remain as historical strings inside locally captured DGX provenance files; no staging file depends on a live school path for inspection.
