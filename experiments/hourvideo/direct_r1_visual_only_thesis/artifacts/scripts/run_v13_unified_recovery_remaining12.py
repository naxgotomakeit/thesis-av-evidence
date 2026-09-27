#!/usr/bin/env python3
"""Unified-rule recovery segment: accept verified incident, run remaining 96."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('${PROJECT_MSC_ROOT}/direct_r1_visual_only_correction_v12_unified_recovery')
SNS=SOURCE/'runs/direct_r1_visual_only_correction_v12_unified_recovery_remaining_30'
SM=SNS/'recovery_manifest_v12.json'; SM_SHA='d89d2a8dce094a0207e597ce778ce14105df413525b96607e4d4a0ff0ab2f50c'
SL=SNS/'budget_ledger.json'; SL_SHA='6ad96b45fefb0a1ef8e68ba2073dbb305bcfc22f99bc432530b64ed7a0df9ccf'
INCIDENT='R1VO12:70f2a750-f403-41b8-aabb-480eb3ab4ed4_4_20'; UNKNOWN=INCIDENT+':attempt:1'
EXP='direct_r1_visual_only_correction_v13_unified_recovery_remaining_12'; NS=ROOT/'runs'/EXP
MP=NS/'recovery_manifest_v13.json'; LP=NS/'recovery_launch_lock_v13.json'; RR=ROOT/'recovery/v13_incident_005_qa_success_unknown_cost.json'
sys.path.insert(0,str(ROOT/'src'))
from direct_api_v1.anthropic_provider import AnthropicDirectAgent
from direct_api_v1.formal_runtime import FormalOrchestrator
from direct_api_v1.frame_resolver import FrozenFrameResolver
from direct_api_v1.maps import FORBIDDEN_QUESTION_KEYS
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def write(p,x):
 p=Path(p); s=json.dumps(x,indent=2,ensure_ascii=False)+'\n'
 if p.exists() and p.read_text()!=s:raise RuntimeError('frozen drift '+str(p))
 if not p.exists():p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
def verify(route,unknown):
 ap=SNS/'route_artifacts'/f'{route}.json';sp=SNS/'route_status'/f'{route}.json';a=load(ap);s=load(sp);l=load(SL)
 starts=[x for x in lines(SNS/'journals/request_start.jsonl') if x['route_id']==route];ends=[x for x in lines(SNS/'journals/attempt_end.jsonl') if x['route_id']==route];ctrl=[x for x in lines(SNS/'journals/controller_result.jsonl') if x['route_id']==route]
 ids={x['attempt_id'] for x in starts}; by={x['attempt_id']:x for x in ends}
 if len(starts)!=8 or len(ends)!=8 or len(ctrl)!=7 or ids!=set(by):raise RuntimeError('journal incomplete')
 u=by[unknown]
 if u.get('response_received') or u.get('response_metadata',{}).get('failure_stage')!='send_or_wait_for_response' or u.get('response_metadata',{}).get('http_status') is not None:raise RuntimeError('unknown class mismatch')
 if sum(bool(x.get('response_received')) for x in ends)!=7:raise RuntimeError('durable response count')
 for x in starts:
  h=hashlib.sha256(x['attempt_id'].encode()).hexdigest();req=SNS/'request_payloads'/f'{h}.json';resp=SNS/'provider_responses'/f'{h}.json'
  if not req.exists() or ((x['attempt_id']!=unknown)!=resp.exists()):raise RuntimeError('request/response persistence')
 if a.get('terminal_status')!='final_answer' or a.get('final_prediction') not in 'ABCDE' or a.get('rounds')!=7 or len(a.get('turns',[]))!=7 or a['turns'][-1].get('action_type')!='final_answer':raise RuntimeError('QA result incomplete')
 if a.get('total_api_attempts')!=8 or len(a.get('provider_attempts',[]))!=8:raise RuntimeError('attempt artifact mismatch')
 fs=[f for t in a['turns'] for f in t.get('resolved_frames',[]) if not f.get('duplicate_of_seen_frame')]
 if len(fs)!=a.get('unique_images_transmitted') or any(f['expected_sha256']!=f['observed_sha256'] or sha(f['frame_path'])!=f['expected_sha256'] for f in fs):raise RuntimeError('frame evidence mismatch')
 if abs(sum(float(x.get('cache_aware_usd') or 0) for x in ends)-float(a['total_usd']))>1e-12:raise RuntimeError('known cost mismatch')
 if l['attempts'][unknown]['state']!='uncertain_liability' or l['attempts'][unknown]['reserved_upper_bound_usd']!=.25256:raise RuntimeError('unknown reserve mismatch')
 if s['state']!='blocked_unknown_provider_outcome' or s['unresolved_attempt_ids']!=[unknown]:raise RuntimeError('blocked status mismatch')
 return {'schema_version':'unified_network_recovery_evidence_v1','provider_sends':0,'gold_read':False,'eligibility':{'network_result_unknown':True,'in_process_retry_obtained_complete_final':True,'full_QA_success':True,'records_consistent':True,'input_frame_sha_valid':True,'budget_reservation_retained':True},'decision':'accept_QA_result_keep_unknown_cost_and_skip_route','route_id':route,'question_id':a['question_id'],'source_artifact':str(ap),'source_artifact_sha256':sha(ap),'source_status':str(sp),'source_status_sha256':sha(sp),'unknown_attempt_id':unknown,'unknown_reserved_usd':.25256,'durable_responses':7,'attempts':8,'turns':7,'unique_images':a['unique_images_transmitted'],'original_assets_modified':False}
def prepare():
 if sha(SM)!=SM_SHA or sha(SL)!=SL_SHA:raise RuntimeError('source identity drift')
 for p in (ROOT/'src/direct_api_v1').glob('*.py'):
  if sha(p)!=sha(SOURCE/p.relative_to(ROOT)):raise RuntimeError('QA code drift')
 rec=verify(INCIDENT,UNKNOWN);write(RR,rec);sm=load(SM);sts={p.stem:load(p) for p in (SNS/'route_status').glob('*.json')}
 routes=sm['routes']; done=routes[:17];blocked=routes[17];remain=routes[18:]
 if blocked['route_id']!=INCIDENT or len(remain)!=12 or any(sts.get(x['route_id'],{}).get('state')!='terminal_success' for x in done) or any(x['route_id'] in sts for x in remain):raise RuntimeError('stop boundary mismatch')
 refs=list(sm['completed_references'])
 for r in done:
  ap=SNS/'route_artifacts'/f"{r['route_id']}.json";refs.append({'question_id':r['question_id'],'source_kind':'v9_terminal_success','source_route_id':r['route_id'],'source_artifact':str(ap),'source_artifact_sha256':sha(ap)})
 ap=Path(rec['source_artifact']);refs.append({'question_id':blocked['question_id'],'source_kind':'unified_rule_QA_success_unknown_cost','source_route_id':blocked['route_id'],'source_artifact':str(ap),'source_artifact_sha256':sha(ap),'recovery_record':str(RR),'recovery_record_sha256':sha(RR)})
 vs=[]
 for old in sm['videos']:
  v=dict(old);vid=v['video_id'];v['r1_visual_only_map']=str(ROOT/f'maps/r1_visual_only/{vid}/r1_visual_only_navigation_map.json');v['frame_sha_manifest']=str(ROOT/f'manifests/frame_sha256_current_snapshot/{vid}.json');vs.append(v)
 vb={v['video_id']:v for v in vs};nr=[]
 for old in remain:
  r=dict(old);r['route_id']='R1VO13:'+r['question_id'];r['map_path']=vb[r['video_id']]['r1_visual_only_map'];nr.append(r)
 cfg=load(ROOT/'config/direct_v1_anthropic_smoke.json');cred=Path(cfg['credential_env_path'])
 m={'schema_version':'r1_visual_only_v13_unified_network_recovery_v1','experiment_id':EXP,'gold_loaded':False,'api_calls_at_freeze':0,'total_coverage':175,'accepted_references':163,'new_routes':12,'unified_rule':'network_unknown_plus_complete_in_process_retry_QA_only_keep_reservation_no_resend','rule_failure_action':'stop','hard_budget_usd':15.0,'source_manifest':str(SM),'source_manifest_sha256':SM_SHA,'source_ledger':str(SL),'source_ledger_sha256':SL_SHA,'credential_file_path':str(cred),'credential_file_sha256':sha(cred),'required_transport':'urllib_fallback','protocol_identity':sm['protocol_identity'],'completed_references':refs,'routes':nr,'videos':vs};return m
def create(m):
 if NS.exists():return
 NS.mkdir(parents=True)
 for d in ('inputs','journals','route_status','route_artifacts','route_checkpoints','provider_responses','request_payloads','failure_diagnostics'):(NS/d).mkdir()
 for f in ('request_start.jsonl','attempt_end.jsonl','controller_result.jsonl','lifecycle.jsonl'):(NS/'journals'/f).write_bytes(b'')
 for r in m['routes']:
  p=SNS/'inputs'/f"{r['question_id']}.json"
  if sha(p)!=r['question_sha256'] or set(load(p))&FORBIDDEN_QUESTION_KEYS:raise RuntimeError('input drift')
  shutil.copy2(p,NS/'inputs'/p.name)
 write(MP,m);old=load(SL);ats={}
 for aid,row in old['attempts'].items():
  x={**row,'inherited_from':str(SL),'source_ledger_sha256':SL_SHA}
  if aid==UNKNOWN:x['state']='inherited_unified_rule_approved_unknown'
  elif 'actual_cost_usd' not in row:x['state']='inherited_unknown_liability'
  else:x['state']='inherited_settled_attempt'
  ats[aid]=x
 write(NS/'budget_ledger.json',{'schema_version':'reserved_budget_ledger_v2','cap_usd':15.0,'spent_usd':old['spent_usd'],'reserved_usd':old['reserved_usd'],'attempt_ids':old['attempt_ids'],'attempts':ats,'charges':[{**x,'inherited_from':str(SL)} for x in old['charges']]})
 write(LP,{'schema_version':'v13_unified_recovery_lock_v1','experiment_id':EXP,'manifest_sha256':sha(MP),'runner_sha256':sha(Path(__file__)),'routes':12,'accepted':163,'cap':15.0,'spent':old['spent_usd'],'reserved':old['reserved_usd'],'available':15-old['spent_usd']-old['reserved_usd'],'credential_file_sha256':m['credential_file_sha256'],'transport':'urllib_fallback','real_execution_allowed':True})
def validate():
 m=prepare();create(m);lock=load(LP)
 if load(MP)!=m or lock['manifest_sha256']!=sha(MP) or lock['runner_sha256']!=sha(Path(__file__)) or sha(m['credential_file_path'])!=m['credential_file_sha256']:raise RuntimeError('freeze drift')
 return m,load(NS/'budget_ledger.json')
def build(m,agent_class=AnthropicDirectAgent,orch_class=FormalOrchestrator):
 cfg=load(ROOT/'config/direct_v1_anthropic_smoke.json');vb={v['video_id']:v for v in m['videos']}
 def res(r):v=vb[r['video_id']];return FrozenFrameResolver.from_hierarchy(load(v['hierarchy']),load(v['frame_sha_manifest']))
 def agent(r,l,s):
  z=load(NS/'budget_ledger.json');avail=z['cap_usd']-z['spent_usd']-z['reserved_usd'];return agent_class(credential_env_path=Path(cfg['credential_env_path']),model=cfg['model'],pricing=cfg['anthropic_cache_aware_pricing_usd_per_million_tokens'],max_output_tokens=cfg['max_output_tokens'],timeout_sec=cfg['timeout_sec'],max_retries=cfg['max_retries'],max_total_usd=avail,max_billable_input_tokens=200000,lifecycle=l,transport_mode='urllib_fallback',credential_file_sha256=m['credential_file_sha256'],require_file_credential=True)
 return orch_class(manifest_path=MP,namespace=NS,agent_factory=agent,resolver_factory=res,cap_usd=15.0)
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-real-recovery',action='store_true');p.add_argument('--confirm-routes',type=int);p.add_argument('--confirm-cap',type=float);a=p.parse_args();m,l=validate();avail=15-l['spent_usd']-l['reserved_usd']
 if not a.execute_real_recovery:print(json.dumps({'valid':True,'api_calls':0,'accepted':163,'remaining':12,'spent':l['spent_usd'],'reserved':l['reserved_usd'],'available':avail},indent=2));return
 if a.confirm_routes!=12 or a.confirm_cap!=15:raise SystemExit('exact 12/$15 confirmation required')
 print(json.dumps(build(m).execute(),indent=2))
if __name__=='__main__':main()
