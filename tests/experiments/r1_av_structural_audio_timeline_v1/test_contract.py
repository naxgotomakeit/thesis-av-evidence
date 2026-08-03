from __future__ import annotations

from experiments.r1_av_structural_audio_timeline_v1.core import audio_timeline_node, timeline_sort_key


def test_audio_node_is_conservative() -> None:
    node = audio_timeline_node({"audio_id": "A1", "start_sec": 1.0, "end_sec": 2.0, "transcript": "Call EMS", "source_type": "speech", "asr_status": "ok", "fallback_used": False})
    assert node["node_type"] == "audio_asr"
    assert node["physical_event_claim"] is False
    assert node["visual_confirmation"] is False


def test_timeline_sort_is_deterministic() -> None:
    audio = {"node_type": "audio_asr", "node_id": "A1", "start_sec": 1.0, "end_sec": 2.0}
    visual = {"node_type": "visual_structural", "node_id": "C1", "start_sec": 1.0, "end_sec": 3.0}
    assert sorted([audio, visual], key=timeline_sort_key) == [visual, audio]

