from __future__ import annotations

from typing import Any

from src.diagnostics.cradio_v4.runner import (
    INPUT_RESOLUTION,
    RADIO_ADAPTOR,
    RADIO_CHECKPOINT_FILENAME,
    RADIO_GITHUB_REVISION,
    RADIO_HF_REPO,
    RADIO_HF_REVISION,
    SIGLIP2_TEXT_MODEL,
)


B1_BASELINE_ID = "egopolice_b1_raw_question_cradio_1fps_top8_qwen7b_v1"
CRADIO_INDEX_SCHEMA_VERSION = "egopolice-cradio-v4-so400m-siglip2g-1fps-index-v1"
CRADIO_RETRIEVAL_SCHEMA_VERSION = "egopolice-b1-raw-question-retrieval-v1"
B1_RESULT_SCHEMA_VERSION = "egopolice-b1-formal-question-result-v1"
B1_RUN_SCHEMA_VERSION = "egopolice-b1-formal-run-v1"
INDEX_BATCH_SIZE = 4
INDEX_EMBEDDING_DIMENSION = 1536
INDEX_STORAGE_DTYPE = "float16"
INDEX_SAMPLING_FPS = 1.0
INDEX_MINIMUM_FREE_VRAM_BYTES = 12 * 1024**3
QWEN_MINIMUM_FREE_VRAM_BYTES = 20 * 1024**3


def cradio_frozen_configuration() -> dict[str, Any]:
    """Return the exact configuration validated by the YKI08 diagnostic."""
    return {
        "official_model_id": RADIO_HF_REPO,
        "checkpoint_filename": RADIO_CHECKPOINT_FILENAME,
        "checkpoint_revision": RADIO_HF_REVISION,
        "official_implementation": "NVlabs/RADIO",
        "implementation_revision": RADIO_GITHUB_REVISION,
        "adaptor": RADIO_ADAPTOR,
        "adaptor_text_model": SIGLIP2_TEXT_MODEL,
        "compute_dtype": "bfloat16_cuda_autocast",
        "stored_embedding_dtype": INDEX_STORAGE_DTYPE,
        "embedding_dimension": INDEX_EMBEDDING_DIMENSION,
        "input_resolution_height_width": list(INPUT_RESOLUTION),
        "batch_size": INDEX_BATCH_SIZE,
        "sampling_fps": INDEX_SAMPLING_FPS,
        "sampling_rule": "timestamps_0_to_ceil_duration_minus_1_seconds",
        "frame_index_rule": "floor(timestamp_times_average_fps)_clipped",
        "visual_embedding": "official_siglip2-g_summary_l2_normalized",
        "text_embedding": "official_siglip2-g_encode_text_normalize_true",
        "similarity": "cosine_on_l2_normalized_embeddings",
    }
