# Fine-only vs hierarchy-guided retrieval v0.1

## 1. Research question

Can a reusable bottom-up hierarchy over immutable CoMET-style Event leaves reduce Fine-level query scoring while preserving access to Fine evidence found by exhaustive Fine-only search?

## 2. Motivation

The earlier hierarchy proof-of-concept established reversible parent-child structure, but did not test online navigation. This experiment isolates navigation: both methods use the same questions, Fine leaves, CLIP frame embeddings, CLIP text encoder, and final Top-3 Fine budget.

## 3. Relation to the old hierarchical refinement experiment

`hierarchical_refinement_comparison_v0_1` used independent KTS units to gate where CoMET segmentation ran. Its Top-3 KTS zones covered 63.5% of video, produced 11.6 local Fine candidates versus 15.9 full-video candidates, and reduced Fine candidates by 27.7%, but it had no persistent Fine-derived lineage. Here, all Fine and parent nodes already exist offline; query-time search only follows stored parent-child links. Historical KTS→local-CoMET rows appear in the HTML as context, not as a primary method.

## 4. Frozen inputs

- Exact 10-video manifest: `data/manifests/coarse_segmentation_3way_10.json`.
- Fine leaves: `coarse_segmentation_3way_v0_1/comet_segments.jsonl`.
- Complete Safe Merge trees: `fine_to_coarse_hierarchy_v0_1/safe_merge_hierarchies.jsonl`.
- Questions: frozen question-only `full_comet_results.jsonl`; its leakage flags are false.
- Existing 1 FPS CLIP ViT-B/32 frame embeddings.
- Frozen question-scope annotations are displayed for only the three exact matches available; no relabelling was performed.

All source hashes and per-video CLIP embedding hashes are recorded in `frozen_config.json` and `runtime_metrics.json`.

## 5. Fine leaf definition

Fine leaves are the 159 unchanged CoMET-style DINOv2 Event segments. They are not KTS units, motion-refined Actions, fixed windows, or regenerated segments. The hierarchy retains all leaves with exact source IDs and intervals.

## 6. Boundary-aware adjacent bottom-up merge

The experiment reuses the exact prior `boundary_aware_safe_merge` trees and parameters. Only adjacent active nodes were merged. Merge priority used data-derived semantic difference, original boundary strength, merged variability increase, and size/duration pressure, with the frozen Fine→reference-Medium strong-boundary veto. No new formula or tuning was introduced.

## 7. Technical positioning

This is a boundary-aware temporally constrained hierarchical agglomeration experiment inspired at a high level by temporally constrained clustering, TW-FINCH-like temporal constraints, action-boundary signals, and adaptive region reasoning. It is not an exact reproduction of Action100M, TW-FINCH, ABD, or Felzenszwalb. Natural stopping is not implemented.

## 8. Hierarchy representation

The full binary tree is retained. The ≈50% and ≈25% cuts are **reference operating cuts**, not natural semantic levels. Each query-independent parent stores:

- ordered children and all descendant Fine IDs;
- exact interval, boundary and coherence metadata;
- normalized CLIP frame centroid;
- a descendant-Fine medoid prototype;
- representative thumbnails.

Parent scoring is `0.5 × cosine(query, centroid) + 0.5 × cosine(query, medoid)`. It does not score or inspect descendant Fine embeddings online. Two vector comparisons per operating-view node are counted explicitly.

## 9. Fine-only method

The question-only CLIP vector scores every Fine centroid exactly once, stably ranks all leaves, and returns Top-3 Fine evidence.

## 10. Hierarchy-guided method

The primary pre-specified beam is 3:

1. score every Coarse reference-view node;
2. retain the best three branches;
3. score only Medium reference nodes inside them;
4. retain the best three Medium branches;
5. score only descendant Fine leaves;
6. return Top-3 Fine evidence.

Beam 1/2/3 sensitivity is diagnostic and was not selected using QA correctness.

## 11. Fairness controls

The methods share frozen questions, Fine segmentation, Fine embeddings, CLIP encoder, stable ranking, and Top-3 final evidence budget. There is no Planner, LLM label, sufficiency mechanism, reranker, final VLM, answer option, gold answer, or QA correctness input.

## 12. Metrics

Search work distinguishes operating-view node scores, Fine scores, and actual vector comparisons. Fine-only Top-3 reach/overlap is a **diagnostic proxy only**, because temporal evidence ground truth is unavailable. Temporal duration is reported separately and is never equated with compute reduction.

## 13. Aggregate validated findings

- Fine-only: 15.9 Fine/vector comparisons per question.
- Primary hierarchy: 9.7 operating-view nodes, 7.6 Fine leaves, and 17.3 total node-score operations.
- Fine-level scoring falls 52.2%, but total node-score operations rise 8.8%.
- Because every parent uses two prototypes, vector comparisons rise from 15.9 to 27.0 (+69.8%).
- Fine-only Top-1 remains in a visited branch for 9/10 cases.
- Mean Fine-only Top-3 branch reach and final-ID overlap are both 90.0%.
- Mean final selected duration is similar: 38.6s versus 39.8s.

Sensitivity exposes the efficiency/coverage trade-off:

- Beam 1: 8.0 node operations, 14.1 vector comparisons, 60% Top-1 and 40% Top-3 proxy reach.
- Beam 2: 12.5 node operations, 20.4 vector comparisons, 90% Top-1 and 73.3% Top-3 proxy reach.
- Beam 3: 17.3 node operations, 27.0 vector comparisons, 90% Top-1 and 90% Top-3 proxy reach.

## 14. Per-case observations

These are deterministic structural observations, not semantic human judgments:

- `1dace116-...`: the 54–62s Fine-only Top-1 is pruned; the selected Coarse branches cover 0–54s and 174–180s. This is the clearest parent-representation/branch-policy failure.
- `4bd1cbed-...`: Fine-only Top-1 remains reachable, but only one of the Fine-only Top-3 survives branch gating.
- Eight other cases preserve all Fine-only Top-3 IDs; one additional case preserves two of three.
- Seven cases show a narrow Fine Top-5 score spread under the frozen structural redundancy proxy.
- The two frozen Global annotations and one Multi-event annotation all preserve their Fine-only Top-3 proxies, but three annotations are insufficient to assess question-type suitability.

## 15. Runtime and cost

Parent CLIP prototypes take about 0.0022s/video to build from cached frame embeddings. The hierarchy itself is reused with zero reconstruction. Query encoding uses one local cached CLIP load; API calls are zero. Microbenchmark retrieval latency is approximately 0.00004s/question Fine-only versus 0.00012s hierarchy-guided, so control overhead dominates at this small index size. Serialized hierarchy metadata/prototypes are about 8 MB.

## 16. Failure cases

The experiment explicitly records: useful pruning, Fine-only proxy branches pruned, large-parent abstraction risk when it co-occurs with lost proxy leaves, Fine score competition, and broad/multi-event coverage risk where a frozen scope exists. These are structural diagnostics, not causal or correctness labels.

## 17. Limitations

- Fine-only Top-3 is not temporal ground truth.
- Ten videos and three frozen scope labels cannot establish question-type effects.
- Reference cuts are forced compression views and can contain long, mixed parents.
- The two-prototype parent representation is expensive relative to a 16-node Fine index.
- Fixed Top-3 evidence does not solve Global QA coverage.
- No downstream answer accuracy was measured.

## 18. Structural conclusion

The reusable hierarchy clearly reduces Fine-level scoring, but the current centroid+medoid, beam-3 navigation does **not** reduce total retrieval work at this scale and loses the Fine-only Top-1 proxy once. The result supports continued investigation of parent representation and cheaper branch scoring more strongly than it validates the present reference cuts as useful natural levels.

## 19. Future work

**Hypotheses / future ideas only:** natural stopping; cheaper or learned parent prototypes; query-scope-aware beam breadth; score caching; broad-coverage search for Global questions; and evidence budgets that are not fixed Top-3. None is implemented here.

## 20. Outputs

- Main report: `outputs/experiments/fine_only_vs_hierarchy_guided_v0_1/comparison.html`
- Metrics and traces: the same output directory.
- Frozen configuration/provenance: `frozen_config.json` and `run_manifest.json`.

The HTML contains blank manual-review controls. No human observations are auto-filled.
