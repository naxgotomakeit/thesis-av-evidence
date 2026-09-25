# Data-machine transfer and preflight

Suggested destination:

`/data/experiments/hourvideo_v7_4_variant_c_budgets_v1/`

Transfer the two archives and their `.manifest.json` and `.contents.txt`
sidecars with `rsync -avP`. Verify each archive with `sha256sum -c` using the
hash printed in its manifest, then extract into separate `runtime/` and
`index/` directories. Do not overlay an existing runtime or index.

Before an online run, execute the two bundled offline tests described in
`DATA_MACHINE_RUNBOOK.md`, verify the twelve case directories and set
`INDEX_ROOT` to the extracted `work_index`. The preflight must not load a model,
call an API, or read an Eval300 UID list.
