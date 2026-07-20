# Visual pipeline v1

This is the selected, reusable visual-only implementation on `dev/thesis-av`.
It was ported selectively from the frozen experiment record, rather than by
merging the experiment branch.

## Architecture

`raw video -> 1 FPS grid -> DINOv2 -> CoMET-style Fine Events -> complete
Boundary-aware Safe-Merge tree -> original pure Fluid Loose Medium -> informative
keyframe -> local Qwen2-VL-2B Medium caption -> deterministic Semantic Coarse ->
Coarse summaries -> storyline`

Fine nodes are immutable evidence reserve. The tree is complete and reversible.
Medium is the only selected operating frontier: original pure Fluid Loose
(`q_rank >= 0.40`, `local_drop <= 0.30`). Semantic Coarse is adjacent-only and
uses Sentence-T5 caption continuity, a DINO visual-transition veto, posture/state
conflict, and group-centroid anti-chaining. It does not use the failed Qwen
MERGE/STOP boundary classifier.

## Frozen long4 validation

The real cached-input fidelity replay reproduces 295 Fine leaves, 104 Medium
nodes (`8/12/29/55`), 104/104 computed keyframes, and 65 Semantic Coarse nodes
(`8/12/27/18`). It compares complete Fine and Safe-Merge artifacts, exact
Medium IDs/lineage, pair decisions, anti-chaining, summaries, and stories. No
count-only fixture is used. Every Medium retains ordered Fine descendants and
maps exactly once to Semantic Coarse.

The known limitation remains: rapid viewpoint/shot changes can make pairwise
local similarity under-group one broader event. This does not discard evidence:
later retrieval can descend Coarse -> Medium -> Fine.

## Deliberately excluded

Reference 50/25 cuts, Elbow, Fluid Strict/Balanced/Guarded, Absolute thresholds,
Photometric Robust, Photo+Temporal, KTS, and the Qwen boundary classifier remain
experiment-only and are not selectable by v1.

The actually executed deterministic runner at commit `7c5c739` is authoritative.
It did not execute the later-declared photometric bypass, so v1 does not add one.
It also makes zero Qwen MERGE/STOP boundary calls.
