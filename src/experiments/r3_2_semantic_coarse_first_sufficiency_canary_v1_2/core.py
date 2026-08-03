import copy, hashlib, json
from pathlib import Path
from typing import Any

def load(p:Path)->Any:return json.loads(p.read_text(encoding='utf-8'))
def write(p:Path,v:Any)->None:p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()

def project(rows:dict[str,list[dict[str,Any]]],stage2:dict[str,Any])->dict[str,list[dict[str,Any]]]:
 out=copy.deepcopy(rows)
 stage2_ids={qid:{r['requirement_id'] for r in doc.get('output',{}).get('assessments',[])} for qid,doc in stage2.items()}
 for qid,assessments in out.items():
  for row in assessments:
   if row['status']=='supported': row['next_stage']='answer_ready'
   elif row['requirement_id'] in stage2_ids.get(qid,set()): row['next_stage']='fine_visual_review' if row['supporting_coarse_ids'] else 'unresolved'
   elif row['supporting_coarse_ids']: row['next_stage']='coarse_caption_retrieval'
   else: row['next_stage']='global_fallback' if row['status']=='not_found' else 'unresolved'
   row['next_stage_adapter']={'adapter_derived':True,'status_unchanged':True,'rationale_unchanged':True}
 return out

def run(root:Path,config_path:Path)->dict[str,Any]:
 cfg=load(config_path);out=root/cfg['output_root'];out.mkdir(parents=True,exist_ok=True);src=root/cfg['source_root']
 paths={'final':src/'final_requirement_assessments.json','stage2':src/'stage2_sufficiency_outputs.json','validation':src/'validation_report.json','cost':src/'cost_accounting.json'}
 before={k:sha(v) for k,v in paths.items()};rows=project(load(paths['final']),load(paths['stage2']))
 errors=[]
 for qid,items in rows.items():
  for r in items:
   if (r['status']=='supported')!=(r['next_stage']=='answer_ready'):errors.append(f"{qid}/{r['requirement_id']}: status-stage mismatch")
 write(out/'final_requirement_assessments.json',rows)
 write(out/'stage_transition_audit.json',{'status':'passed' if not errors else 'failed','errors':errors,'model_fields_changed':['next_stage'],'semantic_fields_changed':[]})
 requests={qid:[{'requirement_id':r['requirement_id'],'target_coarse_ids':r['supporting_coarse_ids'],'detail_query':r['detail_query']} for r in items if r['next_stage']=='fine_visual_review'] for qid,items in rows.items()};write(out/'selective_review_requests.json',requests)
 after={k:sha(v) for k,v in paths.items()};validation={'source_v1_1_unchanged':before==after,'stage_transition_validation':'passed' if not errors else 'failed','semantic_statuses_unchanged':True,'model_calls':0,'overall_validation':'pending_manual_staged_routing_review' if not errors else 'failed'};write(out/'validation_report.json',validation);write(out/'source_hash_audit.json',{'before':before,'after':after})
 return {'validation':validation,'fine_review_requirements':requests}

