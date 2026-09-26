# ABD Cross Analysis Report

## Accuracy reproduction and paired comparisons

- A: 98/300 = 32.67%; B: 103/300 = 34.33%; D: 103/300 = 34.33%.
- D vs B (primary): 69 both correct, 34 D-only, 34 B-only, 163 both wrong; exact two-sided McNemar p = 1; answers changed on 120 questions.
- D vs A (supplementary): 79 both correct, 24 D-only, 19 A-only, 178 both wrong; exact p = 0.542384; answers changed on 78 questions.
- A vs B: 57 both correct, 41 A-only, 46 B-only, 156 both wrong; exact p = 0.668285; answers changed on 160 questions.

## D vs B gain and loss routes

The 34 D-only gains and 34 B-only losses confirm that adding the map changed decisions but produced zero net accuracy gain. Among gain routes, the frozen support label improved on 10, stayed level on 13, and declined on 11. Among loss routes the corresponding counts were 14, 9, and 11. Thus correctness changes cannot be equated with grounding changes.

For D routes with semantic relationship coding, gains most often involved `consistent` evidence (13), while losses included consistent (8), complementary (4), and explicitly conflicting (2) map/frame cases. Detailed transitions and question IDs are in the JSON/CSV outputs. Early valid batches contain 117 D rows with the legacy source descriptor `map_and_images`; these were preserved rather than post-hoc recoded after gold exposure.

Duration, frequency, factual-recall, sequence-recall and spatial questions account for most gain/loss routes. This is descriptive, not a causal attribution: the audit can identify support, complementarity, conflict or insufficiency in the supplied evidence, but cannot reveal the model's internal reason for changing its answer.

## Interpretation boundaries

- A is not language-only: it receives a video-derived semantic map.
- B is not an independent frame-selection method: its images came from the historical Direct R3 navigation/inspection process.
- D vs B is the primary ablation: the answerer, images, timestamps and other visible input are matched; D adds the map.
- Equal B/D accuracy means no net accuracy gain, not no effect. Each arm has 34 unique correct answers relative to the other.
- Reliability or groundedness conclusions come from the frozen evidence audit, not accuracy alone.
- This Codex-assisted audit is a strict exploratory evidence-support judgment, not objective truth and not access to hidden reasoning.
- Per-question-type and per-video results are exploratory, especially for small cells.
