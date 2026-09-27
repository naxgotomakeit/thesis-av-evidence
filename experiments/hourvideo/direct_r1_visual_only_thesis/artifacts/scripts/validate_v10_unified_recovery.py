#!/usr/bin/env python3
import json,sys,tempfile,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_v10_unified_recovery_remaining96 as r
from direct_api_v1.formal_runtime import FormalStore,NamespaceWriterLock,NamespaceLockError
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
class C:
 k=None
 def __init__(self,**kw):C.k=kw
def main():
 m,l=r.validate(); before=(sha(r.MP),sha(r.RR),sha(r.NS/'budget_ledger.json'));m2,l2=r.validate();after=(sha(r.MP),sha(r.RR),sha(r.NS/'budget_ledger.json'))
 refs=[x['question_id'] for x in m['completed_references']];new=[x['question_id'] for x in m['routes']]
 with tempfile.TemporaryDirectory() as d:
  p=Path(d);blocked=False
  with NamespaceWriterLock(p):
   try:
    with NamespaceWriterLock(p):pass
   except NamespaceLockError:blocked=True
  s=FormalStore(p,'t',15);s.initialise();s.reserve_attempt('x',.2,route_id='x');s.mark_uncertain('x','test');newstop=bool(s.uncertain_liabilities())
 o=r.build(m,agent_class=C);o.agent_factory(m['routes'][0],object(),object());k=C.k
 counts={d:len(list((r.NS/d).glob('*.json'))) for d in ('route_status','route_artifacts','request_payloads','provider_responses')}
 checks={'incident_QA_complete':load(r.RR)['eligibility']=={'network_result_unknown':True,'in_process_retry_obtained_complete_final':True,'full_QA_success':True,'records_consistent':True,'input_frame_sha_valid':True,'budget_reservation_retained':True},'recovery_zero_send':load(r.RR)['provider_sends']==0,'coverage_79_plus_96':len(refs)==79 and len(new)==96 and len(set(refs+new))==175 and not set(refs)&set(new),'idempotent_no_duplicate':before==after and len(set(refs))==79,'budget_retained':abs(l['spent_usd']-4.16334005)<1e-9 and abs(l['reserved_usd']-1.76792)<1e-9 and abs(15-l['spent_usd']-l['reserved_usd']-9.06873995)<1e-9,'new_unknown_stops':newstop,'writer_lock':blocked,'namespace_clean':all(x==0 for x in counts.values()) and all(not p.read_text() for p in (r.NS/'journals').glob('*.jsonl')),'QA_unchanged':all(sha(p)==sha(r.SOURCE/p.relative_to(ROOT)) for p in (ROOT/'src/direct_api_v1').glob('*.py')) and k['model']=='claude-haiku-4-5-20251001' and k['max_output_tokens']==512 and k['timeout_sec']==120 and k['max_retries']==1 and k['transport_mode']=='urllib_fallback' and k['require_file_credential'] is True,'inputs_maps_frames':all(sha(r.NS/'inputs'/f"{x['question_id']}.json")==x['question_sha256'] for x in m['routes']) and all(sha(v['r1_visual_only_map'])==v['r1_visual_only_sha256'] and sha(v['frame_sha_manifest'])==v['frame_sha_manifest_sha256'] for v in m['videos'])}
 out={'schema_version':'v10_unified_recovery_validation_v1','api_calls':0,'gold_read':False,'checks':checks,'accepted':79,'remaining':96,'budget':{'spent':l['spent_usd'],'reserved':l['reserved_usd'],'available':15-l['spent_usd']-l['reserved_usd']},'pass':all(checks.values())};p=ROOT/'validation/v10_unified_recovery.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
