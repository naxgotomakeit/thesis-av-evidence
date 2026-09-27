#!/usr/bin/env python3
import json,sys,tempfile,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_v11_unified_recovery_remaining52 as r
from direct_api_v1.formal_runtime import FormalStore,NamespaceWriterLock,NamespaceLockError
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
class C:
 k=None
 def __init__(self,**kw):C.k=kw
def main():
 m,l=r.validate();before=(sha(r.MP),sha(r.RR),sha(r.NS/'budget_ledger.json'));r.validate();after=(sha(r.MP),sha(r.RR),sha(r.NS/'budget_ledger.json'));refs=[x['question_id'] for x in m['completed_references']];new=[x['question_id'] for x in m['routes']]
 with tempfile.TemporaryDirectory() as d:
  p=Path(d);blocked=False
  with NamespaceWriterLock(p):
   try:
    with NamespaceWriterLock(p):pass
   except NamespaceLockError:blocked=True
  s=FormalStore(p,'t',15);s.initialise();s.reserve_attempt('x',.2,route_id='x');s.mark_uncertain('x','test');newstop=bool(s.uncertain_liabilities())
 o=r.build(m,agent_class=C);o.agent_factory(m['routes'][0],object(),object());k=C.k;counts={d:len(list((r.NS/d).glob('*.json'))) for d in ('route_status','route_artifacts','request_payloads','provider_responses')};e=load(r.RR)['eligibility']
 checks={'incident_fully_eligible':all(e.values()),'recovery_zero_send':load(r.RR)['provider_sends']==0,'coverage_123_plus_52':len(refs)==123 and len(new)==52 and len(set(refs+new))==175 and not set(refs)&set(new),'idempotent':before==after and len(set(refs))==123,'budget':abs(l['spent_usd']-6.25493125)<1e-9 and abs(l['reserved_usd']-2.02048)<1e-9 and abs(15-l['spent_usd']-l['reserved_usd']-6.72458875)<1e-9,'new_unknown_stops':newstop,'writer_lock':blocked,'clean':all(v==0 for v in counts.values()) and all(not p.read_text() for p in (r.NS/'journals').glob('*.jsonl')),'QA_unchanged':all(sha(p)==sha(r.SOURCE/p.relative_to(ROOT)) for p in (ROOT/'src/direct_api_v1').glob('*.py')) and k['model']=='claude-haiku-4-5-20251001' and k['max_output_tokens']==512 and k['timeout_sec']==120 and k['max_retries']==1 and k['transport_mode']=='urllib_fallback' and k['require_file_credential'] is True,'inputs_maps_frames':all(sha(r.NS/'inputs'/f"{x['question_id']}.json")==x['question_sha256'] for x in m['routes']) and all(sha(v['r1_visual_only_map'])==v['r1_visual_only_sha256'] and sha(v['frame_sha_manifest'])==v['frame_sha_manifest_sha256'] for v in m['videos'])}
 out={'schema_version':'v11_unified_recovery_validation_v1','api_calls':0,'gold_read':False,'checks':checks,'accepted':123,'remaining':52,'budget':{'spent':l['spent_usd'],'reserved':l['reserved_usd'],'available':15-l['spent_usd']-l['reserved_usd']},'pass':all(checks.values())};p=ROOT/'validation/v11_unified_recovery.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
