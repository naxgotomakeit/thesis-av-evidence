#!/usr/bin/env python3
"""Register only explicitly accepted second-pass batches after structural validation."""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; AUDIT=ROOT/'outputs/abd_eval300_evidence_audit_v1'
VALID={'supported','partially_supported','unsupported','contradicted','unreviewable'}
REQ={'audit_id','option_support','rationale','evidence_refs','map_region_time_range','frame_timestamps','missing_key_evidence','confidence','review_flag','evidence_relationship'}
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--add',action='append',default=[]); args=p.parse_args()
 manifest={b['batch_id']:b for b in json.loads((AUDIT/'batch_manifest.json').read_text())['batches']}
 registry=AUDIT/'valid_second_pass_batches.json'
 data=json.loads(registry.read_text()) if registry.exists() else {'schema_version':'abd_valid_second_pass_batch_registry_v1','gold_loaded':False,'batches':{}}
 accepted={f'B{i:03d}' for i in range(1,21)}|{f'B{i:03d}' for i in range(65,80)}|{f'B{i:03d}' for i in range(129,137)}|set(args.add)
 for bid in sorted(accepted):
  p=AUDIT/'reviews'/f'{bid}.jsonl'
  prior_sha=sha(p); rows=[json.loads(x) for x in p.read_text().splitlines() if x]
  expected=manifest[bid]['audit_ids']
  if {r.get('audit_id') for r in rows}!=set(expected) or len(rows)!=len(expected) or any(set(r)!=REQ or r['option_support'] not in VALID for r in rows): raise SystemExit(f'invalid {bid}')
  rows_by_id={r['audit_id']:r for r in rows}; rows=[rows_by_id[aid] for aid in expected]
  canonical=''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows)
  if p.read_text()!=canonical:
   tmp=p.with_suffix('.jsonl.tmp'); tmp.write_text(canonical); os.replace(tmp,p)
  data['batches'][bid]={'status':'valid_actual_evidence_review','audit_ids':expected,'review_sha256':sha(p),'pre_canonicalization_sha256':prior_sha,'acceptance':'explicit_subagent_actual_map_image_viewing_attested'}
 tmp=registry.with_suffix('.tmp'); tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,sort_keys=True)+'\n'); os.replace(tmp,registry)
 print(json.dumps({'valid_batches':len(data['batches'])}))
if __name__=='__main__': main()
