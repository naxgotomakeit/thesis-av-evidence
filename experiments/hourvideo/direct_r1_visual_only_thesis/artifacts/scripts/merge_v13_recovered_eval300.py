#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];M=ROOT/'runs/direct_r1_visual_only_correction_v13_unified_recovery_remaining_12/recovery_manifest_v13.json';N=M.parent;REUSE=ROOT/'manifests/route_reuse_manifest_v8.json';OUT=ROOT/'outputs/r1_visual_only_recovered_eval300_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def cost(a):
 xs=a.get('provider_attempts',[]);v=sum(float(x.get('cache_aware_usd') or 0) for x in xs)
 if abs(v-float(a.get('total_usd') or 0))>1e-8:raise RuntimeError('artifact cost mismatch')
 return v
def main():
 m=load(M);reuse=load(REUSE);sources=[]
 for x in m['completed_references']:
  p=Path(x['source_artifact'])
  if sha(p)!=x['source_artifact_sha256']:raise RuntimeError('reference SHA drift')
  sources.append((x['question_id'],x['source_route_id'],x['source_kind'],p))
 for r in m['routes']:
  p=N/'route_artifacts'/f"{r['route_id']}.json"
  if not p.exists():raise RuntimeError('missing final segment artifact')
  sources.append((r['question_id'],r['route_id'],'v13_new',p))
 q=[x[0] for x in sources]
 if len(q)!=175 or len(set(q))!=175:raise RuntimeError('corrected 175 incomplete/duplicate')
 rr1=reuse['reuse_r1']['routes'];rr3=reuse['reuse_r3']['routes'];q1={x['question_id'] for x in rr1};q3={x['question_id'] for x in rr3}
 if len(q1)!=125 or len(q3)!=300 or set(q)&q1 or set(q)|q1!=q3:raise RuntimeError('175+125/R3 pairing invalid')
 rows=[]
 for qid,rid,kind,p in sources:
  a=load(p)
  if a.get('question_id')!=qid or a.get('method')!='R1':raise RuntimeError('corrected artifact identity')
  rows.append({'method':'R1','question_id':qid,'route_id':rid,'source_kind':kind,'source_artifact':str(p),'source_artifact_sha256':sha(p),'terminal_status':a.get('terminal_status'),'final_prediction':a.get('final_prediction'),'known_settled_route_cost_usd':cost(a),'unknown_reserved_usd':float(a.get('unresolved_reserved_upper_bound_usd') or 0)})
 for method,group in [('R1',rr1),('R3',rr3)]:
  for x in group:
   p=Path(x['source_artifact'])
   if sha(p)!=x['source_artifact_sha256']:raise RuntimeError('reuse SHA drift')
   a=load(p)
   if a.get('question_id')!=x['question_id'] or a.get('method')!=method:raise RuntimeError('reuse identity')
   rows.append({'method':method,'question_id':x['question_id'],'route_id':x['route_id'],'source_kind':'frozen_reuse','source_artifact':str(p),'source_artifact_sha256':sha(p),'terminal_status':a.get('terminal_status'),'final_prediction':a.get('final_prediction'),'known_settled_route_cost_usd':cost(a),'unknown_reserved_usd':0})
 ledger=load(N/'budget_ledger.json');formal_unknown=sum(x['unknown_reserved_usd'] for x in rows if x['method']=='R1' and x['source_kind']!='frozen_reuse')
 summary={'schema_version':'r1_visual_only_recovered_eval300_structural_merge_v1','gold_loaded':False,'accuracy':'not_computed','main_denominator':300,'r1':{'corrected':175,'reused':125,'total':300,'terminal_final_answer':sum(x['method']=='R1' and x['terminal_status']=='final_answer' for x in rows),'failures_retained':sum(x['method']=='R1' and x['terminal_status']!='final_answer' for x in rows)},'r3':{'reused':300,'total':300,'terminal_final_answer':sum(x['method']=='R3' and x['terminal_status']=='final_answer' for x in rows),'failures_retained':sum(x['method']=='R3' and x['terminal_status']!='final_answer' for x in rows)},'pairing_exact':True,'old_replaced_175_excluded':True,'smoke_results_excluded':True,'cost':{'aggregate_ledger_settled_usd':ledger['spent_usd'],'aggregate_unknown_reserved_usd':ledger['reserved_usd'],'corrected_175_known_route_cost_usd':sum(x['known_settled_route_cost_usd'] for x in rows if x['method']=='R1' and x['source_kind']!='frozen_reuse'),'formal_unknown_reserved_usd':formal_unknown,'reservation_is_not_spend':True}}
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'merged_results_no_gold.json').write_text(json.dumps({'summary':summary,'results':rows},indent=2)+'\n');(OUT/'structural_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
