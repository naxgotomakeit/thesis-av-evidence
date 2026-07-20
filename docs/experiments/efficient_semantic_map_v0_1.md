# Efficient semantic map v0.1

## Research question

Can one local-Qwen image caption per conservatively deduplicated Medium visual state, followed by text-only Coarse and whole-video aggregation, create a useful global semantic map over the already frozen EgoPolice long-video hierarchy?

This is an isolated semantic-layer experiment. It does not change Fine, Medium, or Coarse boundaries, parent-child membership, Safe Merge, canonical Ours-v0.1, retrieval, Planner, audio, or QA.

## Frozen structural input

The four hierarchies are read exactly from `outputs/experiments/long_video_hierarchy_stress_v0_2/safe_merge_hierarchies.jsonl`. The file identity and the v0.2 manifest/config identities are bound into the run fingerprint. The hierarchy hash is checked again after execution. The input contains 295 Fine leaves, 149 Medium reference-cut nodes, and 76 Coarse reference-cut nodes.

## Informative Medium keyframe

Candidate images are the already materialized representative frames attached to each frozen Medium node. Selection does not use a VLM or text. Each candidate receives cheap image-quality measurements (Laplacian sharpness, exposure/clipping, entropy) plus existing DINOv2 measurements (similarity to the Medium pooled representation, local temporal stability, and discriminativeness from other Medium centroids). Components are min-max normalized only within the Medium and combined with frozen weights. A stable score/time/path ordering chooses exactly one keyframe.

This is deliberately lightweight and is not claimed to solve keyframe selection. The HTML exposes every candidate and score for manual review.

## Conservative semantic deduplication

Within each video, selected keyframe DINOv2 embeddings are grouped using greedy complete-link similarity at cosine ≥0.995 with a maximum group size of three. Every member must clear the threshold against every existing member, limiting transitive chaining. The highest keyframe-selection score supplies the group representative. Groups are metadata only and never change structural parents.

One direct local-Qwen image caption is generated per group. Other members explicitly store `caption_source=propagated_from_group`, their source Medium, and similarity. Reduced calls are an efficiency measurement, not evidence of equal caption quality.

## Semantic chain

1. `Qwen/Qwen2-VL-2B-Instruct` reads one selected group keyframe and writes a short factual static visual description.
2. Each fixed Coarse node reads only its ordered timestamped Medium captions and generates a concise local storyline.
3. The whole-video stage reads only ordered timestamped Coarse captions and produces a 1–3 sentence overall story plus a numbered storyline.

The model is loaded once from the existing read-only local Lolly snapshot. Generation is deterministic (`do_sample=false`), local-files-only, and question independent. No external API, gold, answer option, original frame at Coarse/story time, or QA path is present.

## Profiling

Runtime separates feature-cache loading, candidate preparation, image-quality scoring, DINO representation scoring, pairwise similarity, grouping, Qwen image decode/preprocessing/inference, Coarse text preparation/inference, story generation, one-time model loading, steady-state indexing, and cold total. Counts distinguish 149 selected Medium keyframes from actual VLM image calls after conservative grouping.

## Human review

The primary artifact is `outputs/experiments/efficient_semantic_map_v0_1/semantic_map_review.html`. For each video it presents the story first, followed by ordered Coarse captions, Medium selected images and captions, all keyframe candidates/scores, propagation provenance, and a semantic-group audit. All subjective review controls are blank.

## Limitations

- A one-frame caption cannot establish temporal dynamics.
- DINO visual similarity may group same-looking frames with different semantic roles.
- Propagation quality requires human review.
- Text-only summaries can accumulate Medium-caption errors.
- Coarse/Medium cuts remain reference operating cuts rather than learned semantic levels.

## Executed results

All four frozen long videos completed with one persistent local Qwen instance. The 149 Medium nodes exposed 533 existing representative candidates, and one keyframe was selected for every Medium. At the prespecified complete-link cosine threshold of 0.995, no pair satisfied the conservative grouping contract: 149 Medium nodes formed 149 singleton groups. Consequently, actual image calls equalled the one-call-per-Medium baseline and call reduction was 0%. The threshold was not changed after observing this result.

The local model performed 149 Medium image calls, 76 Coarse text-only calls, and four story text-only calls: 229 local inference calls total. Qwen did not reliably follow the requested numbered-list formatting for the story stage. Raw output is preserved; the HTML therefore labels and displays a deterministic ordered-Coarse-caption fallback for the high-level list. Overall-story prose remains the raw model-derived text after whitespace/header parsing.

Steady-state semantic indexing took 464.09 seconds across the four videos; cold total including the single 14.21-second model load was 478.30 seconds. Peak allocated VRAM was 5.61 GB. No OOM or inference failure occurred. These measurements do not establish semantic correctness: the HTML is the required human-review artifact.
