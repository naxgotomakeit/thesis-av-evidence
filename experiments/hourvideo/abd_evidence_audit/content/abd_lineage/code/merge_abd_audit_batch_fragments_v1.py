#!/usr/bin/env python3
"""Validate and atomically merge independently reviewed opaque package fragments."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'outputs/abd_eval300_evidence_audit_v1'
VALID={'supported','partially_supported','unsupported','contradicted','unreviewable'}
REQUIRED={'audit_id','option_support','rationale','evidence_refs','map_region_time_range','frame_timestamps','missing_key_evidence','confidence','review_flag','evidence_relationship'}

def sha(b: bytes)->str: return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('batch_id'); a=p.parse_args()
 manifest=json.loads((AUDIT/'batch_manifest.json').read_text())['batches']
 batch=next((x for x in manifest if x['batch_id']==a.batch_id),None)
 if not batch: raise SystemExit('unknown batch')
 rows=[]; frags=[]
 for aid in batch['audit_ids']:
  path=AUDIT/'package_fragments'/a.batch_id/f'{aid}.json'
  if not path.is_file(): raise SystemExit(f'missing fragment: {aid}')
  raw=path.read_bytes(); row=json.loads(raw)
  if set(row)!=REQUIRED or row['audit_id']!=aid or row['option_support'] not in VALID: raise SystemExit(f'invalid fragment: {aid}')
  rows.append(row); frags.append({'audit_id':aid,'sha256':sha(raw),'path':str(path)})
 raw=''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows).encode()
 out=AUDIT/'reviews'/f'{a.batch_id}.jsonl'; out.parent.mkdir(exist_ok=True)
 tmp=out.with_suffix('.jsonl.tmp'); tmp.write_bytes(raw); os.replace(tmp,out)
 ledger=AUDIT/'valid_second_pass_batches.json'
 old=json.loads(ledger.read_text()) if ledger.exists() else {'schema_version':'abd_valid_second_pass_batch_registry_v1','gold_loaded':False,'batches':{}}
 old['batches'][a.batch_id]={'status':'valid_actual_evidence_review','audit_ids':batch['audit_ids'],'review_sha256':sha(raw),'fragments':frags}
 tmp=ledger.with_suffix('.tmp'); tmp.write_text(json.dumps(old,ensure_ascii=False,indent=2,sort_keys=True)+'\n'); os.replace(tmp,ledger)
 print(json.dumps({'batch_id':a.batch_id,'rows':len(rows),'sha256':sha(raw)}))
if __name__=='__main__': main()
