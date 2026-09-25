# GenS V2 Eval300 staging report

Status: **PASS**

- Absolute staging path: `/myriadfs/home/ucemxna/Scratch/workspace/staging/gens_eval300_v2_github_20260925`
- Regular files: 60
- Total bytes: 2907953
- Unique Eval300 UIDs: 300/300
- Lightweight downstream rows: 300/300
- Audit provenance rows: 300/300
- Lightweight vs formal per-question selection mismatches: 0
- Formal statuses: `{"ok": 300}`
- Finish reasons: `{"eos": 300}`
- Selected-frame distribution: `{"1": 207, "12": 1, "16": 83, "2": 7, "3": 2}`
- Selected-frame references / unique video-frame pairs: 1567 / 1219
- Sensitive scan: PASS (0 credential/private-key hits)
- Banned image/video/weight/embedding payloads: 0
- Redactions: none; prompts and selection results remain byte-identical to their staged sources.
- Missing required items: none.

`MANIFEST.sha256` covers every file except itself and this final aggregate report. `COPY_PROVENANCE.jsonl` records both source and copy SHA-256 for every copied file.
