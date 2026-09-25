# Directory structure

```text
.
├── README.md
├── INCLUDED_EXCLUDED.md
├── CANDIDATES.md
├── MISSING_FILES.md
├── VERIFICATION.md
├── CREDENTIAL_SCAN.md
├── content/
│   ├── common/legacy_runtime/                 # shared frozen dependency closure
│   ├── preprocessing_index/
│   │   ├── code/  config/  reports/
│   │   ├── freeze_manifests/
│   │   ├── r3/
│   │   └── r3_v74/
│   ├── direct_r1_r3_eval300/
│   │   ├── code/  config/  run_scripts/
│   │   ├── freeze/
│   │   ├── results/{route_status,journals,canonical_summary_v1}/
│   │   └── reports/
│   ├── gens_haiku_v3/
│   │   ├── code/  config/  run_and_score/
│   │   ├── selector_myriad_v2/  preflight/
│   │   └── results/{routes,route_status,journals,evaluation}/
│   ├── capacity_aware_eval300_paired150/
│   │   ├── code/  config/  prompts/  run_and_score/
│   │   ├── eligibility/  freeze/
│   │   ├── results/{post_planner_attempts,route_status,canonical_summary_v1}/
│   │   └── reports/
│   └── supplementary/full_staged_paired100/
│       ├── code/  config/  run_and_score/  preflight/
│       ├── results/{route_status,journals,analysis}/
│       └── reports/thesis_package/
├── provenance/
│   ├── SOURCE_MANIFEST.jsonl
│   ├── REDACTIONS.json
│   ├── CREDENTIAL_SCAN.json
│   └── CHECKSUMS.sha256
├── DO_NOT_COMMIT/                            # absolute-path local audit only
└── tools/                                    # one-off local assembly utilities
```

`DO_NOT_COMMIT/` and `tools/` are excluded by `.gitignore`. The latter contains
the local assembler with server paths and is not part of the GitHub import.

