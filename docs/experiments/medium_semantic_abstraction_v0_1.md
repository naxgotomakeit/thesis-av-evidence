# Medium semantic abstraction v0.1

## Research question

Does a deterministic three-frame view (25/50/75%) materially improve local VLM descriptions of frozen Medium hierarchy nodes over one shared midpoint frame, and is any improvement worth the extra image processing cost?

## Frozen inputs and controls

- Exactly the 10-video Boundary-aware Safe Merge hierarchy from `fine_to_coarse_hierarchy_v0_1`.
- 82 Medium reference-cut nodes and 43 Coarse reference-cut nodes.
- Fine leaves are never captioned.
- The 1-frame midpoint is exactly the midpoint image in the 3-frame condition.
- Both conditions use the same local Qwen2-VL-2B-Instruct model, prompt, decoding settings, hierarchy, and node budget.
- Coarse summaries read ordered Medium text only and never inspect images.
- No question, options, gold, QA, retrieval, external API, or canonical-pipeline integration is present.

## Model

The runner resolves the existing read-only local snapshot at `D:/lolly_data/models/models--Qwen--Qwen2-VL-2B-Instruct/` into an output-local symlink view. It never downloads or changes the Lolly cache. One persistent model instance performs a one-image smoke test and is then reused for all Medium and Coarse generations.

## Timing

Model load is measured once. For each Medium condition, image decode, preprocessing, model inference, and total generation are recorded separately. Per-video and aggregate reports keep one-time model loading separate from steady-state semantic indexing. Coarse text-only aggregation is reported separately.

## Human review

`outputs/experiments/medium_semantic_abstraction_v0_1/comparison.html` is the primary artifact. It exposes every selected image, timestamp, raw Qwen output, timing, ordered Medium-to-Coarse inputs, Coarse outputs, and blank human-review fields.

## Limitation

This experiment provides cost measurements and review material; it does not automatically establish that one or three frames are semantically superior.

## Completed measurements

The one persistent local model loaded once in 15.94s on an NVIDIA GeForce RTX 3060 Laptop GPU (6GB physical VRAM; approximately 4.42GB allocated after load). It completed 250 local generations: 164 Medium descriptions and 86 text-only Coarse summaries.

- 1-frame: 82 images, 72.40s Medium VLM inference, 80.61s total Medium semantic indexing, 0.983s mean per Medium, and 46.88s Coarse aggregation.
- 3-frame: 246 images, 90.63s Medium VLM inference, 103.94s total Medium semantic indexing, 1.268s mean per Medium, and 41.61s Coarse aggregation.
- Three-frame Medium inference cost was 1.252× the one-frame condition while processing exactly 3× the images.

The Coarse timing difference is output-length/content dependent and must not be interpreted as an inherent advantage of either image condition. Semantic quality remains entirely for human review in `comparison.html`.
