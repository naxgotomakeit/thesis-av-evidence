"""Portable, explicitly non-thesis preparation contracts and safety boundaries."""
import ast
import base64
import copy
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Protocol

sys.dont_write_bytecode = True
from run_demo import ROOT, DemoError, SafeParser, contained, require, verify

MODEL = 'claude-haiku-4-5-20251001'
FIDELITY = {
    'preparation_profile': 'PORTABLE_API', 'caption_backend': 'API',
    'caption_model': MODEL, 'thesis_caption_model': 'Qwen2.5-VL-7B-Instruct',
    'model_identity_matches_thesis': False,
    'full_historical_embedding_index': False, 'direct_targeted_preparation': True,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def safe_write(path, value):
    """Private generated metadata only; reject any accidentally reflected credential."""
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    key = os.environ.get('ANTHROPIC_API_KEY', '')
    require(not key or key not in text, 'Credential reflection blocked; output was not saved.')
    Path(path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        stream.write(text)


def contracts():
    manifest = read(ROOT/'demo/portable_contracts.json')
    data = {}
    for name, item in manifest['sources'].items():
        path = contained(ROOT, item['path'])
        verify(path, item['sha256'])
        if path.suffix == '.json':
            data[name] = read(path)
    prompt = data['caption']['prompt']
    require(hashlib.sha256(prompt.encode()).hexdigest() == data['caption']['prompt_sha256'])
    frozen, request = data['organizer_config'], data['organizer_request']
    require(request['system'] == frozen['system_prompt'])
    require(hashlib.sha256(request['system'].encode()).hexdigest() == frozen['prompt_sha256'])
    for field in ('model', 'max_tokens', 'temperature'):
        require(request[field] == frozen[field])
    # Load only the pinned pure conversion function; do not execute server module imports.
    tree = ast.parse(contained(ROOT, manifest['sources']['map_converter']['path']).read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_map_from_output')
    namespace = {'Any': Any}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<pinned-map-converter>', 'exec'), namespace)
    data['convert'] = namespace['_map_from_output']
    return manifest, data


def workdir_path(value):
    """Prototype writes only to the explicitly ignored repository demo_runs root."""
    base = ROOT/'demo_runs'
    require(not base.is_symlink(), 'demo_runs must not be a symlink.')
    path = Path(value).expanduser().resolve()
    require(path.is_relative_to(base) and path != base, 'Use a new directory under repository demo_runs/.')
    return path


def probe(video):
    require(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'Install ffmpeg and ffprobe before preparation.')
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                             '-show_entries', 'stream=width,height:format=duration', '-of', 'json', str(video)],
                            capture_output=True, text=True, timeout=60, check=True)
    data = json.loads(result.stdout)
    duration = float(data['format']['duration'])
    require(math.isfinite(duration) and 0 < duration <= 600, 'Prototype supports videos longer than zero and at most 600 seconds.')
    stream = data['streams'][0]
    require(0 < stream['width'] <= 8000 and 0 < stream['height'] <= 8000,
            'Prototype rejects oversized image dimensions; no silent resizing is performed.')
    return {'duration_sec': duration, 'width': stream['width'], 'height': stream['height']}


def extraction_config():
    """Record relocatable argv; input/output placeholders are explicitly labelled."""
    version = subprocess.run(['ffmpeg','-version'],capture_output=True,text=True,check=True).stdout.splitlines()[0]
    return {'fps':1, 'jpeg_qscale':6, 'resize':False, 'start_number':0,
            'filename_pattern':'frame_%05d.jpg', 'ffmpeg_version':version,
            'argv_template':['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n',
                             '-i','<source-video>','-vf','fps=1','-q:v','6','-start_number','0',
                             '<workdir>/frames_1fps/frame_%05d.jpg']}


def extract(video, directory, duration):
    require(not directory.exists(), 'Refusing to overwrite a frame cache.')
    directory.mkdir(parents=True)
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-n', '-i', str(video),
                    '-vf', 'fps=1', '-q:v', '6', '-start_number', '0', str(directory/'frame_%05d.jpg')],
                   check=True, capture_output=True, timeout=1200)
    frames = sorted(directory.glob('frame_*.jpg'))
    require(math.floor(duration) <= len(frames) <= math.ceil(duration) and bool(frames), 'Frame count differs from duration bounds.')
    require([p.name for p in frames] == [f'frame_{i:05d}.jpg' for i in range(len(frames))])
    require(all(0 < p.stat().st_size <= 5_000_000 for p in frames), 'Empty or oversized JPEG; no silent resizing is performed.')
    return frames


def geometry(count, video_uid):
    require(isinstance(count, int) and count > 0)
    fine, medium = [], []
    for i, start in enumerate(range(0, count, 15), 1):
        end = min(count, start+15)
        center = min(count-1, (start+end-1)//2)
        fine.append({'fine_id': f'F{i:03d}', 'start_sec': float(start), 'end_sec': float(end),
                     'timestamp_sec': float(center), 'frame_index': center,
                     'source_frame_path': f'frames_1fps/frame_{center:05d}.jpg'})
    for i, start in enumerate(range(0, count, 45), 1):
        end = min(count, start+45)
        children = [f for f in fine if start <= f['start_sec'] < end]
        identity = f'M{i:03d}'
        for f in children:
            f['parent_medium_id'] = identity
        medium.append({'medium_id': identity, 'start_sec': float(start), 'end_sec': float(end),
                       'duration_sec': float(end-start), 'source_fine_ids': [f['fine_id'] for f in children],
                       'representative_frame_paths': [f['source_frame_path'] for f in children]})
    return {'schema_version': 'portable-direct-targeted-hierarchy-v1', 'video_uid': video_uid,
            'duration_sec': float(count), 'segmentation_policy': {'fine_fixed_sec':15, 'medium_fixed_sec':45,
            'question_independent':True}, 'fine_nodes':fine, 'medium_nodes':medium}


class CaptionBackend(Protocol):
    """Future backends return the same caption rows; backend identity remains explicit."""
    def build_request(self, medium: dict, workdir: Path, hashes: dict) -> dict: ...
    def caption(self, medium: dict, workdir: Path, hashes: dict, transport: Any) -> tuple[dict, dict]: ...


def response_text(response):
    require(response.get('stop_reason') == 'end_turn', 'Provider response incomplete; no automatic retry.')
    text = ''.join(block['text'] for block in response['content'] if block['type'] == 'text')
    require(bool(text.strip()), 'Empty provider output.')
    return text


class ApiCaptionBackend:
    def __init__(self, frozen):
        self.prompt = frozen['caption']['prompt']
        self.prompt_sha = frozen['caption']['prompt_sha256']

    def build_request(self, medium, workdir, hashes):
        content = []
        for name in medium['representative_frame_paths']:
            path = contained(workdir, name)
            verify(path, hashes[name])
            content.append({'type':'image', 'source':{'type':'base64', 'media_type':'image/jpeg',
                            'data':base64.b64encode(path.read_bytes()).decode('ascii')}})
        content.append({'type':'text', 'text':self.prompt})
        return {'model': MODEL, 'temperature':0.0, 'max_tokens':256,
                'messages':[{'role':'user', 'content':content}]}

    def caption(self, medium, workdir, hashes, transport):
        response = transport.create(**self.build_request(medium, workdir, hashes))
        text = response_text(response)
        row = {k:medium[k] for k in ('medium_id','start_sec','end_sec')}
        row.update({'qwen_caption':text, 'caption_source':'portable_api_action_preserving_v1',
                    'source_frame_paths':medium['representative_frame_paths'], 'prompt_sha256':self.prompt_sha,
                    **FIDELITY})
        return row, response


def caption_backend(name, frozen):
    require(name == 'api', 'local-qwen and remote-qwen are reserved interfaces, not implemented.')
    return ApiCaptionBackend(frozen)


def organizer_request(frozen, hierarchy, captions, segments=None, audio_mode='none'):
    import audio
    audio.profile(audio_mode)
    segments = [] if segments is None else segments
    require(audio_mode == 'whisper' or not segments, 'Visual-only Organizer requires empty ASR.')
    aligned = audio.attach(hierarchy['medium_nodes'],segments)
    require(len(captions) == len(hierarchy['medium_nodes']))
    timeline = []
    for index, (medium, caption) in enumerate(zip(hierarchy['medium_nodes'], captions)):
        require(medium['medium_id'] == caption['medium_id'])
        timeline.append({'medium_index':index, 'interval':[medium['start_sec'],medium['end_sec']],
                         'caption':caption['qwen_caption'], 'overlapping_asr':aligned[medium['medium_id']]})
    request = copy.deepcopy(frozen['organizer_request'])
    if audio_mode == 'whisper':
        request['system'] = contained(ROOT,audio.SOURCES['prompt'][0]).read_text()
        config = read(contained(ROOT,audio.SOURCES['config'][0]))
        require(object_sha(request['output_config']['format']['schema']) == config['provider_schema_canonical_json_sha256'], 'AV Organizer schema differs.')
        for key in ('model','temperature','max_tokens'):
            require(request[key] == config[key], 'AV Organizer request settings differ.')
    payload = {'timeline':timeline, 'contract':{'global_view':True, 'storyline':False, 'hard_filtering':False}}
    request['messages'] = [{'role':'user','content':json.dumps(payload, ensure_ascii=False, separators=(',',':'))}]
    return request


def convert_map(frozen, hierarchy, captions, raw, segments=None, audio_mode='none'):
    import audio
    audio.profile(audio_mode)
    segments = [] if segments is None else segments
    require(audio_mode == 'whisper' or not segments, 'Visual-only map requires empty ASR.')
    audio.validate(segments)
    media = hierarchy['medium_nodes']
    require(len(media) == len(captions) and bool(media), 'Caption/Medium coverage mismatch.')
    previous_end = 0.0
    for medium, caption in zip(media, captions):
        require(medium['start_sec'] == previous_end and medium['end_sec'] > previous_end,
                'Non-contiguous Medium intervals.')
        require(all(caption.get(k) == medium[k] for k in ('medium_id','start_sec','end_sec')),
                'Caption/Medium identity mismatch.')
        require(isinstance(caption['qwen_caption'],str) and bool(caption['qwen_caption'].strip()), 'Missing source caption.')
        previous_end = medium['end_sec']
    if 'duration_sec' in hierarchy:
        require(previous_end == hierarchy['duration_sec'], 'Final video boundary mismatch.')
    require(isinstance(raw, dict) and set(raw) == {'groups'} and isinstance(raw['groups'], list) and bool(raw['groups']), 'Invalid Organizer groups.')
    previous = -1
    for group in raw['groups']:
        require(set(group) == {'end_medium_index','navigation_summary','uncertainty_notes'}, 'Invalid Organizer fields.')
        end = group['end_medium_index']
        require(type(end) is int and previous < end < len(captions), 'Invalid Organizer boundary.')
        require(isinstance(group['navigation_summary'], str) and bool(group['navigation_summary'].strip()))
        require(isinstance(group['uncertainty_notes'], list) and all(isinstance(x,str) for x in group['uncertainty_notes']))
        previous = end
    require(previous == len(captions)-1, 'Organizer does not cover every Medium.')
    return frozen['convert'](hierarchy, captions, segments, raw)


def estimate(medium_count, image_count):
    # Heuristic planning range, not a tokenizer or invoice. Organizer cap remains 64k.
    low = (image_count*300 + medium_count*250 + 1200 + medium_count*100)/1e6 + (medium_count*80+medium_count*60)*5/1e6
    high = (image_count*1600 + medium_count*500 + 3000 + medium_count*350)/1e6 + (medium_count*256+medium_count*160)*5/1e6
    return {'heuristic_usd_range':[round(low,4),round(high,4)],
            'estimated_input_tokens_range':[image_count*300+medium_count*350+1200,
                                            image_count*1600+medium_count*850+3000],
            'estimated_output_tokens_range':[medium_count*140,medium_count*416],
            'conservative_accounting_usd':round(medium_count*(200000+256*5)/1e6+(200000+64000*5)/1e6,6),
            'assumptions':'Caption 300–1600 image tokens/image; text/output allowance by Medium. Not exact tokens or billing guarantee.',
            'rates_usd_per_million':{'input':1.0,'output':5.0},
            'pricing_source':'https://platform.claude.com/docs/en/about-claude/pricing'}


def approval(plan, execute, approval_sha):
    key = os.environ.get('ANTHROPIC_API_KEY', '')
    require(not key or key not in json.dumps(plan), 'Credential reflection blocked; plan suppressed.')
    summary = {k:v for k,v in plan.items() if k not in ('files','implementation','prompt_config_sources')}
    print(json.dumps({'execution_plan_summary':summary, 'plan_sha256':object_sha(plan)}, indent=2))
    print('APPROVAL REQUIRED: inspect this plan, then explicitly rerun with --execute --approve-plan <plan_sha256>.')
    if not execute:
        return False
    require(approval_sha == object_sha(plan), 'Approval does not match this exact plan; no request sent.')
    return True


def credential():
    key = os.environ.get('ANTHROPIC_API_KEY', '')
    require(bool(key.strip()), 'Set ANTHROPIC_API_KEY in the environment; no credential file or CLI key is accepted.')
    require(not any(x in key.lower() for x in ('dummy','fake','placeholder')), 'Placeholder credentials are not sent.')
    return key


class MessagesTransport:
    """No retries, no redirects, no payload logs; failure locks this transport."""
    def __init__(self, key, max_calls, timeout=900):
        self.key, self.max_calls, self.timeout = key, max_calls, timeout
        self.calls, self.failed = 0, False

    def create(self, **payload):
        import urllib.request
        require(not self.failed and self.calls < self.max_calls, 'Request limit reached or prior send uncertain; no resend.')
        require(self.key not in json.dumps(payload), 'Credential in request content blocked; no request sent.')
        self.calls += 1
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        request = urllib.request.Request('https://api.anthropic.com/v1/messages',
                  data=json.dumps(payload, ensure_ascii=False).encode(), method='POST',
                  headers={'content-type':'application/json','x-api-key':self.key,'anthropic-version':'2023-06-01',
                           'anthropic-beta':'prompt-caching-2024-07-31'})
        try:
            with urllib.request.build_opener(NoRedirect).open(request, timeout=self.timeout) as response:
                result = json.loads(response.read())
            require(self.key not in json.dumps(result), 'Credential reflected by provider; response suppressed.')
            require(result.get('model') == payload['model'], 'Unexpected provider model identity.')
            return result
        except Exception:
            self.failed = True
            raise DemoError('Provider request failed or response uncertain. It may be billed. No automatic resend; details suppressed.') from None


def cli_entry(function):
    try:
        return function() or 0
    except DemoError as error:
        print('Stopped: '+str(error), file=sys.stderr)
        return 2
    except Exception:
        print('Stopped: validation/execution failed; exception payload suppressed. No automatic retry.', file=sys.stderr)
        return 2
