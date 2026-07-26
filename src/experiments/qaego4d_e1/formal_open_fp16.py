from __future__ import annotations
import argparse,gc,json
from pathlib import Path
from src.experiments.qaego4d_e1.core import atomic_write_json,load_json,sha256_file,stable_hash
from src.experiments.qaego4d_e1.runner import DEFAULT_CONFIG,DEFAULT_ANNOTATION_ROOT,_load_all_cases,_checkpoint_path,_record
from src.experiments.qaego4d_e1_fp16_validation.runner import FP16ValidationModel
ROOT=Path(__file__).resolve().parents[3]; AMEND=ROOT/'config/experiments/qaego4d_e1_fp16_amendment_1.json'; AMENDMENT_2=ROOT/'THESIS_EXPERIMENT_FREEZE_V2.2_AMENDMENT_2.md'; OUT=ROOT/'outputs/experiments/qaego4d_e1_open_fp16_v1';CONDS=('blind','uniform_8','oracle_leq8')
def run(a):
 c=load_json(a.config);am=load_json(a.amendment);ch,ah=stable_hash(c),stable_hash(am);a2h=sha256_file(AMENDMENT_2);cases,hs=_load_all_cases(c,a.annotation_root);rows=[];pending=[]
 for case in cases['open']:
  for cond in CONDS:
   p=_checkpoint_path(a.output_root,case,cond);r=load_json(p) if p.exists() else None
   if isinstance(r,dict) and r.get('success') is True and r.get('formal_result') is True and r.get('dtype')=='float16' and r.get('config_hash')==ch and r.get('manifest_hash')==hs['open'] and r.get('protocol_amendment_hash')==ah:rows.append(r)
   else:pending.append((case,cond))
 runid=f'qaego4d-e1-open-fp16-{ch[:12]}-{ah[:12]}';meta={'run_id':runid,'expected_questions':1850,'expected_records':5550,'conditions':CONDS,'model_path':c['answer_models']['formal_candidate']['local_path'],'dtype':'float16','backend':'huggingface','max_pixels':c['decoding']['max_pixels'],'config_hash':ch,'manifest_hash':hs['open'],'amendment_id':am['amendment_id'],'amendment_hash':ah};a.output_root.mkdir(parents=True,exist_ok=True);snapshot=a.output_root/'protocol_snapshot.json';
 if not snapshot.exists():atomic_write_json(snapshot,meta)
 atomic_write_json(a.output_root/'protocol_amendment_2_snapshot.json',{'amendment_id':'V2.2_AMENDMENT_2_ORACLE_ZERO_DECODABLE_FRAME','amendment_hash':a2h,'path':str(AMENDMENT_2),'applies_to':'newly generated records only','run_id':runid});print(json.dumps({**meta,'amendment_2_hash':a2h,'cache_hits':len(rows),'pending':len(pending)}),flush=True)
 m=None
 try:
  if pending:
   s=c['answer_models']['formal_candidate'];m=FP16ValidationModel(model_path=Path(s['local_path']),max_pixels=int(c['decoding']['max_pixels']),seed=int(c['decoding']['seed']),minimum_free_vram_gib=a.minimum_free_vram_gib)
   for i,(case,cond) in enumerate(pending,1):
    checkpoint=_checkpoint_path(a.output_root,case,cond);previous=load_json(checkpoint) if checkpoint.exists() else None
    if isinstance(previous,dict) and previous.get('success') is not True:
     created=str(previous.get('created_at','unknown')).replace(':','-').replace('+','_')
     archive=a.output_root/'failure_history'/case.task/cond/f'{case.question_id}.{created}.json'
     atomic_write_json(archive,previous)
    r=_record(case=case,condition=cond,config=c,config_hash=ch,manifest_hash=hs['open'],model=m,run_id=runid,model_profile='formal_candidate',formal_result=True,run_kind='formal_e1_open_fp16_phase2',protocol_amendment_hash=ah);r.update({'dtype':'float16','backend':'huggingface','max_pixels':int(c['decoding']['max_pixels']),'amendment_id':am['amendment_id'],'protocol_amendment_2_hash':a2h,'protocol_amendment_2_id':'V2.2_AMENDMENT_2_ORACLE_ZERO_DECODABLE_FRAME'});atomic_write_json(checkpoint,r);rows.append(r);atomic_write_json(a.output_root/'progress.json',{'completed':len(rows),'expected':5550,'last':[case.question_id,cond],'success':r['success']})
    if not r['success']:raise RuntimeError(r['error'])
    if i%25==0:print(json.dumps({'completed':len(rows),'expected':5550}),flush=True)
 finally:
  if m:m.close()
  gc.collect()
 atomic_write_json(a.output_root/'records.json',rows);return 0
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=DEFAULT_CONFIG);p.add_argument('--amendment',type=Path,default=AMEND);p.add_argument('--annotation-root',type=Path,default=DEFAULT_ANNOTATION_ROOT);p.add_argument('--output-root',type=Path,default=OUT);p.add_argument('--minimum-free-vram-gib',type=float,default=20.0);return run(p.parse_args(argv))
if __name__=='__main__':raise SystemExit(main())
