"""Opt-in, request-boundary visual telemetry with content-addressed exact bytes."""
from __future__ import annotations
import contextvars, hashlib, json, os, threading, time, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_active=contextvars.ContextVar('visual_input_telemetry_active',default=None)
_identity=contextvars.ContextVar('visual_input_telemetry_identity',default={})
_lock=threading.Lock(); _rounds={}

def _sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def _utc()->str:return datetime.now(timezone.utc).isoformat()
def set_identity(uid:str,video_id:str)->None:
    _identity.set({'uid':str(uid),'video_id':str(video_id)})
def _root()->Path|None:
    raw=(os.getenv('VISUAL_INPUT_TELEMETRY_ROOT') or '').strip()
    return Path(raw) if raw else None
def begin(*,component:str,prompt:str,model:str,backend:str,context:dict[str,Any]|None=None)->object|None:
    root=_root()
    if root is None:return None
    ident=dict(_identity.get() or {}); ctx=dict(context or {})
    uid=str(ctx.pop('uid',None) or ident.get('uid') or os.getenv('VIDEOSEAL_UID') or '')
    video=str(ctx.pop('video_id',None) or ident.get('video_id') or os.getenv('VIDEOSEAL_VIDEO_ID') or '')
    key=(os.getpid(),uid,component)
    with _lock:_rounds[key]=_rounds.get(key,0)+1; round_no=_rounds[key]
    rid=f"{uuid.uuid4()}"
    manifest={
      'telemetry_schema':'visual-input-request-v1','experiment':os.getenv('VISUAL_INPUT_TELEMETRY_EXPERIMENT',''),
      'profile':os.getenv('VISUAL_INPUT_TELEMETRY_PROFILE',''),'uid':uid,'video_id':video,
      'component':component,'call_round':round_no,'request_id':rid,'backend':backend,'model':model,
      'prompt_text_sha256':_sha(prompt.encode('utf-8')),'prompt_utf8_bytes':len(prompt.encode('utf-8')),
      'model_config_profile_sha256':os.getenv('VISUAL_INPUT_MODEL_CONFIG_PROFILE_SHA256',''),
      'request_started_utc':_utc(),'request_ended_utc':None,'status':'prepared','images':[],
      'context':ctx,'process_id':os.getpid()
    }
    mdir=root/'request_manifests'; mdir.mkdir(parents=True,exist_ok=True)
    path=mdir/f"{rid}.json"
    state={'root':root,'path':path,'manifest':manifest,'seen':{}}
    path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return _active.set(state)
def capture_image_bytes(*,encoded_bytes:bytes,position:int,source_path:str='',mime_type:str='image/jpeg',metadata:dict[str,Any]|None=None)->None:
    state=_active.get()
    if not state:return
    digest=_sha(encoded_bytes); previous=state['seen'].get(position)
    if previous:
        if previous!=digest:raise RuntimeError(f'telemetry retry bytes changed at image position {position}')
        return
    root=state['root']; dest=root/'telemetry_images'/'sha256'/digest[:2]/f'{digest}.jpg'; dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        temp=dest.with_name(f'.{dest.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}')
        temp.write_bytes(encoded_bytes)
        if _sha(temp.read_bytes())!=digest:raise RuntimeError('telemetry CAS write verification failed')
        try:os.link(temp,dest)
        except FileExistsError:pass
        finally:temp.unlink(missing_ok=True)
    if _sha(dest.read_bytes())!=digest:raise RuntimeError('telemetry CAS readback verification failed')
    info=dict(metadata or {}); info.update({'position':position,'source_path':source_path,
      'portable_relative_path':str(dest.relative_to(root)),'mime_type':mime_type,'encoded_byte_length':len(encoded_bytes),'encoded_bytes_sha256':digest})
    try:
        from PIL import Image
        with Image.open(dest) as im:info['width'],info['height']=im.size
    except Exception:info['width']=info['height']=None
    state['seen'][position]=digest; state['manifest']['images'].append(info); state['manifest']['images'].sort(key=lambda x:x['position'])
    state['path'].write_text(json.dumps(state['manifest'],indent=2,sort_keys=True)+'\n',encoding='utf-8')
def finish(token:object|None,*,status:str,error:str='')->None:
    if token is None:return
    state=_active.get()
    if state:
        state['manifest']['status']=status; state['manifest']['error']=error
        state['manifest']['request_ended_utc']=_utc()
        state['path'].write_text(json.dumps(state['manifest'],indent=2,sort_keys=True)+'\n',encoding='utf-8')
    _active.reset(token)
