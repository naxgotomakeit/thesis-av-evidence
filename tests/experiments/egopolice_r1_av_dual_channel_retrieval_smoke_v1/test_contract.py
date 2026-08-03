from __future__ import annotations

from experiments.egopolice_r1_av_dual_channel_retrieval_smoke_v1.core import audio_rank


def test_audio_ranking_preserves_modality_scope() -> None:
    plan = {"search_units": [{"unit_id": "U1", "description": "EMS request", "query_variants": ["call EMS"]}]}
    question = {"question": "Was EMS mentioned?"}
    audio = [{"audio_id": "A1", "start_sec": 1.0, "end_sec": 2.0, "transcript": "Call EMS", "source_type": "speech", "asr_status": "ok"}]
    full, top, _ = audio_rank(plan, question, audio, 1)
    assert len(full) == len(top) == 1
    assert top[0]["support_scope"] == "audible_statement_or_mention_only"
    assert top[0]["visual_confirmation"] is False

