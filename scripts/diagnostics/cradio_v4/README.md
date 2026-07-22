# C-RADIOv4 representation smoke

This isolated diagnostic embeds the full `pasadena/YKI08` source video on a
deterministic 1 FPS grid with the official NVIDIA C-RADIOv4-SO400M checkpoint.
It stores only timestamps, source frame indices, and compact float16
`siglip2-g` visual-summary embeddings. It then retrieves timestamps using only
the frozen question text and evaluates annotated GT intervals post hoc.

It does not call Qwen, define B1, or modify B0 or the frozen visual pipeline.

Install the two additional official TorchHub dependencies and run:

```bash
python -m pip install -r scripts/diagnostics/cradio_v4/requirements.txt
python scripts/diagnostics/cradio_v4/run_yki08_representation_smoke.py
```

The model implementation and checkpoint revisions are pinned in the runner.
Artifacts are written to `outputs/diagnostics/cradio_v4/`.

