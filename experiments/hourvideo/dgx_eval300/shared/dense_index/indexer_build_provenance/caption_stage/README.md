# R3 three-frame action-preserving index build

This isolated directory builds the visual-only hierarchical index for the 12
videos named by the frozen 300-UID list. The UID list is used only to extract
the UUID prefix. No question, option, answer, ASR, subtitle, Planner, Inspector,
or Eval300 execution is part of this build.

The Fine registry, Fine SigLIP embeddings and Medium visual embeddings are
copied byte-for-byte from the existing question-independent assets and verified
by SHA-256. Medium captions are regenerated with the frozen C-minimal prompt
from three chronological frames. Haiku 4.5 then organizes the complete ordered
Medium-caption timeline into a visual-only Coarse navigation map.

All generated artifacts, logs, accounting and copied index assets remain below
this directory. The pipeline is resumable and refuses to overwrite a completed
artifact with a different prompt, model, source hash or configuration.

Stages:

```bash
python pipeline.py preflight
python pipeline.py prepare
CUDA_VISIBLE_DEVICES=<free_gpu> python pipeline.py caption
python pipeline.py organize
python pipeline.py validate
```

`organize` is the only stage that calls an external API. `caption` uses one
persistent local Qwen2.5-VL-7B model instance. Neither stage reads benchmark
question content.
