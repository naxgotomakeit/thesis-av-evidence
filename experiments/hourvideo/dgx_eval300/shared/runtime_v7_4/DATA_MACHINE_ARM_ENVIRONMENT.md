# ARM runner environment

Source smoke used Python 3.10 on x86_64. Create a fresh ARM/aarch64 environment; do not copy this environment or wheels. Match Python 3.10 and the locked Python package versions where ARM-compatible builds exist. Install the DGX/CUDA-compatible torch and vLLM builds first, then transformers, tokenizers, numpy, pyarrow, OpenCV, OpenAI and requests. Map the existing SigLIP cache for `google/siglip-base-patch16-224`; no model files are included.
