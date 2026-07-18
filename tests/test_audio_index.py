import wave

import numpy as np

from src.audio.index import (
    build_acoustic_regions,
    compute_acoustic_features,
    decode_wav,
    interval_overlap_ratio,
    propose_acoustic_boundaries,
    samples_to_seconds,
    validate_acoustic_region,
    validate_complete_coverage,
    validate_speech_region,
    validate_transcript_segment,
)


def test_wav_decoding(tmp_path):
    path = tmp_path / "tiny.wav"
    samples = (np.sin(2*np.pi*440*np.arange(1600)/16000) * 1000).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(16000); handle.writeframes(samples.tobytes())
    waveform, metadata = decode_wav(path)
    assert waveform.shape == (1600,)
    assert metadata["original_sample_rate"] == 16000 and metadata["channels"] == 1


def test_timestamp_conversion():
    assert samples_to_seconds(8000, 16000) == 0.5


def test_region_schemas():
    speech = {"speech_region_id":"speech_0000","start_time":0.0,"end_time":1.0,"duration":1.0,"transcript":"hi","asr_language":"en","vad_probability_summary":{},"warnings":[]}
    transcript = {"transcript_segment_id":"transcript_0000","speech_region_id":"speech_0000","start_time":0.0,"end_time":1.0,"text":"hi","language":"en","whisper_metadata":{},"warnings":[]}
    acoustic = {"acoustic_region_id":"acoustic_0000","start_time":0.0,"end_time":1.0,"duration":1.0,"mean_rms":.1,"peak_rms":.2,"mean_spectral_change_score":.3,"speech_overlap_ratio":1.0,"low_information_or_silence":False,"warnings":[]}
    assert validate_speech_region(speech)
    assert validate_transcript_segment(transcript)
    assert validate_acoustic_region(acoustic)


def test_short_audio_handling_and_complete_coverage():
    waveform = np.zeros(100, dtype=np.float32)
    features = compute_acoustic_features(waveform, 16000, .5, .1, .7)
    boundaries, _ = propose_acoustic_boundaries(features["times"], features["smoothed_change"], len(waveform)/16000, 2.0)
    regions = build_acoustic_regions(features, boundaries, [], .15)
    assert len(regions) == 1
    assert validate_complete_coverage(regions, len(waveform)/16000)


def test_speech_acoustic_overlap_is_preserved():
    speech = [{"start_time":1.0,"end_time":3.0}]
    assert interval_overlap_ratio(0.0, 4.0, speech) == .5
    features = {"times":np.array([.5,1.5,2.5,3.5]),"rms":np.ones(4),"smoothed_change":np.zeros(4)}
    regions = build_acoustic_regions(features, [0.0,4.0], speech, .15)
    assert regions[0]["speech_overlap_ratio"] == .5
    assert validate_complete_coverage(regions, 4.0)
