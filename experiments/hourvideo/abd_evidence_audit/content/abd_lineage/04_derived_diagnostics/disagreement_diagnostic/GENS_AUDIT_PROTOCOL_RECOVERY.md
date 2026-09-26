# GenS evidence-audit protocol recovery and comparison

## Recovered original sources

The original pre-review instruction is preserved in local Codex history at entry 1306/1307,
session `01a06c43-6803-7301-9a6f-2703b6e6c179`. Its review phase required:

- current Codex, not a paid API or independent reviewer model;
- neutral IDs, with method, gold, correctness and prior versions hidden;
- batches of at most ten routes, smaller when image-heavy;
- actual inspection of transmitted images, maps and necessary tool feedback;
- no inference from reason wording or filenames alone;
- the six option-support labels and the independent multi-label reason dimension;
- evidence locators, missing evidence, confidence and further-review flag;
- special caution for duration, counts, ordering and full-video absence claims;
- classification freeze before any later association.

The turn context immediately preceding the audit request and the subsequent audit turns records model
`gpt-5.6-sol` (the longer-lived session had used `gpt-5.6-terra` in earlier, unrelated turns). The frozen criteria file SHA-256
is `7e9e25909a97ade16b86c896e5b086f5148b7d3f1bda8a8118db8c07f318f4e3`.
No separate item-level reviewer system prompt, decoding settings, or pre-review labelled exemplars
were preserved. `fixed_examples.json` was generated only later and is outside this no-association
diagnostic; it was not opened here.

## Labels recovered verbatim

- `supported`: actual input contains clear evidence capable of distinguishing the selected option.
- `partially_supported`: relevant clues exist, but decisive evidence is missing.
- `unsupported`: actual input cannot support the selected option.
- `contradicted`: actual input explicitly refutes the selected option.
- `unreviewable`: images cannot be viewed, package is damaged, or verification cannot be completed.
- `no_final_answer`: no final answer exists.

The preserved operational examples are rules rather than labelled item examples: duration, count,
ordering and global non-occurrence require corresponding temporal coverage; more selected frames do
not prove longer duration, and absence from selected frames does not prove absence from the video.

## Comparison

### Original GenS/R1/R3 audit versus old ABD audit

The old ABD `AUDIT_CRITERIA_FROZEN.md` is byte-identical to the original file and has the same SHA.
The ABD task instruction also explicitly required reusing the five applicable labels, judging only
model-visible evidence, no re-answering, explicit contradiction for `contradicted`, and actual map/image
inspection. Its valid second-pass ledger attests actual evidence viewing. It changed workflow execution,
however: the old ABD second pass used clean-context subagents (local session metadata records the
parent as `gpt-5.6-sol` and the three contemporaneous worker sessions as `gpt-5.6-luna`), whereas the
original audit was documented as current-Codex batched review under `gpt-5.6-sol`. Exact
per-item prompts and decoding settings for those ABD workers were not persisted, so full execution
equivalence cannot be confirmed.

### Original GenS/R1/R3 audit versus correction review

The portable correction input archive contains no criteria/prompt file. The returned report states a
compatible high-level rule—record-local supplied evidence, the five labels, no common-sense completion,
and support assessment rather than re-answering—but adds a specific operational rule that absent option
images are `unsupported`, not technical `unreviewable`. The correction reviewer/model identity, exact
prompt, decoding settings, per-item presentation and navigation sequence were not included in the
returned freeze. Therefore the correction round cannot be verified as protocol-identical to GenS.
