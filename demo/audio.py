"""Optional portable AV preparation; no model imports until approved execution."""
import math
import subprocess
import time
import portable as p

ALIGNMENT = 'asr.start_sec < medium.end_sec AND asr.end_sec > medium.start_sec'
BASE = 'experiments/egopolice/final_case_study_v_av_speech/artifacts/organizer_v2_2/'
SOURCES = {
    'prompt': (BASE+'system_prompt_v2_2.txt', '532c58e7df9c2e86e12669f5c9dffc1fc4c607cfb516ce12668f9369be271a1d'),
    'config': (BASE+'organizer_v2_2_config.json', '6d3ec75a23fb62a238dcf9dc8fc046c22ba69b3684b6d65c16a089f8e729eb4a'),
    'historical_audio': ('experiments/hourvideo/school_formal/content/common/legacy_runtime/src/experiments/hourvideo_r1_av_r3_2_single_video_smoke/local_prepare.py', 'ca3d222db95783783613ab9f37658a49d5b41a187d1462e643d89d8c5f1cb2a0'),
}


def profile(mode):
    p.require(mode in ('none', 'whisper'), 'Unknown audio mode.')
    result = {'audio_mode':mode, 'audio_profile':'THESIS_FINAL_VISUAL_ONLY' if mode == 'none' else 'PORTABLE_AV_EXTENSION'}
    if mode == 'whisper':
        for path, sha in SOURCES.values():
            p.verify(p.contained(p.ROOT,path),sha)
        result.update(asr_model='small', asr_language='en', alignment_policy=ALIGNMENT,
                      organizer_profile='AV-aware', organizer_prompt_identity=dict(zip(('path','sha256'),SOURCES['prompt'])),
                      asr_settings={'word_timestamps':False,'condition_on_previous_text':False,
                                    'device_policy':'CUDA when available; otherwise CPU','fp16':'CUDA only'},
                      audio_extraction_argv_template=extraction_command('<source-video>','audio_16khz_mono.wav'),
                      audio_sources={k:dict(zip(('path','sha256'),v)) for k,v in SOURCES.items()})
    return result


def require_stream(video):
    result = subprocess.run(['ffprobe','-v','error','-select_streams','a:0','-show_entries',
                             'stream=index','-of','json',str(video)], capture_output=True,text=True,check=True,timeout=60)
    import json
    p.require(bool(json.loads(result.stdout).get('streams')), 'Whisper mode requires an audio stream; none found. Choose --audio-mode none explicitly.')


def extraction_command(video, wav):
    return ['ffmpeg','-y','-v','error','-i',str(video),'-vn','-ac','1','-ar','16000',str(wav)]


def validate(segments):
    p.require(isinstance(segments,list), 'ASR segments must be a list.')
    ids = set()
    for row in segments:
        p.require(isinstance(row,dict) and set(row) == {'audio_id','start_sec','end_sec','exact_transcript','source_type','language'}, 'Invalid ASR schema.')
        p.require(isinstance(row['audio_id'],str) and row['audio_id'].startswith('A') and row['audio_id'][1:].isdigit() and len(row['audio_id']) >= 5 and row['audio_id'] not in ids, 'Invalid/duplicate ASR identity.')
        ids.add(row['audio_id'])
        p.require(all(type(row[k]) in (int,float) and math.isfinite(row[k]) for k in ('start_sec','end_sec')) and 0 <= row['start_sec'] < row['end_sec'], 'Invalid ASR interval.')
        p.require(row['source_type'] == 'audio_asr' and row['language'] == 'en' and isinstance(row['exact_transcript'],str) and bool(row['exact_transcript'].strip()), 'Invalid ASR text/source/language.')


def attach(mediums, segments):
    validate(segments)
    return {m['medium_id']:[a for a in segments if a['start_sec'] < m['end_sec'] and a['end_sec'] > m['start_sec']] for m in mediums}


def run(video, root, uid):
    """Adapt run_audio: same extraction/transcription settings, no server paths."""
    try:
        import soundfile as sf
        import torch
        import whisper
    except ImportError:
        raise p.DemoError('Whisper mode needs openai-whisper, PyTorch, NumPy and soundfile/libsndfile. Install optional dependencies before execution.') from None
    require_stream(video)
    wav = root/'audio_16khz_mono.wav'
    p.require(not wav.exists(), 'Refusing to overwrite audio.')
    started = time.perf_counter()
    subprocess.run(extraction_command(video,wav),check=True,capture_output=True,timeout=600)
    extraction = time.perf_counter()-started
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    started = time.perf_counter()
    model = whisper.load_model('small',device=device)
    load_time = time.perf_counter()-started
    waveform, sr = sf.read(wav,dtype='float32',always_2d=False)
    p.require(sr == 16000 and waveform.ndim == 1, 'Audio must be mono 16 kHz.')
    started = time.perf_counter()
    result = model.transcribe(waveform,language='en',fp16=device == 'cuda',word_timestamps=False,condition_on_previous_text=False,verbose=False)
    inference = time.perf_counter()-started
    segments = [{'audio_id':f'A{i+1:04d}', 'start_sec':round(float(r['start']),3), 'end_sec':round(float(r['end']),3),
                 'exact_transcript':str(r.get('text','')).strip(), 'source_type':'audio_asr', 'language':result.get('language')}
                for i,r in enumerate(result.get('segments',[])) if str(r.get('text','')).strip()]
    validate(segments)
    p.safe_write(root/'audio_asr.json',{'video_uid':uid,'segments':segments})
    p.safe_write(root/'audio_cost.json',{'model':'openai-whisper-small','package_version':getattr(whisper,'__version__','UNKNOWN'),
        'device':device,'audio_sha256':p.digest(wav),'segment_count':len(segments), 'ffmpeg_extraction_sec':extraction,
        'model_load_sec':load_time,'inference_sec':inference,'api_calls':0,
        'argv_template':extraction_command('<source-video>','audio_16khz_mono.wav')})
    return segments
