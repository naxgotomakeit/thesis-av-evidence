# R3 Full Staged vs R3 Direct paired100

- Full Staged: **24/100 = 24%**, predictions 97/100.
- R3 Direct: **32/100 = 32%**, predictions 99/100.
- Fixed denominator: 100 for both methods; failed routes count as incorrect.
- Paired outcomes (Staged/Direct): both correct 15, Staged-only correct 9, Direct-only correct 17, neither correct 59.

## Full Staged resources

- Logical calls: 478 ({'shared': 296, 'fine': 82, 'final': 100}).
- Physical requests: 662; validation retries 184; transport retries 0.
- Images: 1424 physical transmissions; 1312 unique reviewed fine images.
- Tokens: ordinary input 7621330; cache write 382270; cache read 209771; output 615100.
- API latency sum 6432.660s; observed formal wall clock 6602.582s.
- Failed routes: `4572b198-2c1c-4920-bcf0-95fcebe12261_4_3`, `819c8af7-851f-434f-ab32-318285bc54b1_7_17`, `db3f7933-dfa0-4678-9d4f-393b628ded45_11_2`.

## Cost scopes

- New Full Staged downstream cost: **$11.195645**.
- Historical selected Planner cost: **$3.299245** (recorded usage-based estimate).
- Reconstructed complete Full Staged cost: **$14.494890**.
- Direct R3 historical cost on the same 100: **$4.301863**.
- Raw closure SHA: `fec945bfcf788b4cdf3b709794e6797e2a5bf974c83005cb86ef58b2bcdc3a17`.

Planner and downstream were run at different times; their summed latency is not reported as a measured end-to-end wall time.
