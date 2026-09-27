from __future__ import annotations
import hashlib,json
from pathlib import Path
from . import formal_runtime as gate
from .paired_pilot import run

def freeze_manifest(root:Path,cfg:dict)->str:
 p=root/cfg['pilot_manifest']; base=gate.load_json(root/cfg['base_config']); cap=root/base['frozen_inputs']['local_route_capacity_reference']; rows=[json.loads(x) for x in cap.read_text().splitlines() if x]; r3={x['question_id']:x for x in rows if x['route']=='r3_2'}; ordered=sorted(r3.values(),key=lambda x:x['question_index'])
 payload={'selection_rule':'frozen_eval300_question_index_order; capacity label only; no gold, correctness, completion, or feasibility filtering','questions':[{'question_id':x['question_id'],'previous_r3_capacity':x['formal_status']} for x in ordered],'routes':['r1_av','r3_2']}
 raw=json.dumps(payload,sort_keys=True,separators=(',',':')).encode(); sha=hashlib.sha256(raw).hexdigest()
 if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()!=sha: raise RuntimeError('formal manifest mismatch; refusing to alter frozen population')
 p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists(): p.write_bytes(raw)
 return sha

def launch(root:Path,config_path:Path):
 cfg=gate.load_json(config_path); sha=freeze_manifest(root,cfg); return {'manifest_sha256':sha,**run(root,config_path)}
