#!/usr/bin/env python3
"""Materialise/validate the Direct-v1.2 3/16 formal namespace; default is dry."""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from direct_api_v1.formal_runtime import FormalStore,atomic_json,append_jsonl,fingerprints,sha
EXPERIMENT='direct_v1_2_3x16_r1_r3_eval300_formal_v1';OUT=ROOT/'outputs/direct_v1_formal'/EXPERIMENT
RUNTIME=Path('${LEGACY_RUNTIME}')
SOURCE=RUNTIME/'outputs/experiments/hourvideo_v6_6_2_formal_eval300_v2/source';R3=Path('${R3_V74_WORKSPACE}/work_index/cases');DEV=Path('${HOURVIDEO_ROOT}/benchmark/v1.0_release/json/dev_v1.0.json')
def load(p):return json.loads(Path(p).read_text())
def main():
 if OUT.exists(): raise RuntimeError('formal namespace already exists; refuse mutation')
 cfg=load(ROOT/'config/direct_v1_anthropic_smoke.json'); pop=load(ROOT/'audit/eval300_population_manifest.json');dev=load(DEV);ids=pop['eval300_question_ids'];qv=pop['question_to_video']
 rows={r['qid']:r for v in dev.values() for r in v['benchmark_dataset']}
 if set(ids)!=set(qv) or not set(ids)<=set(rows):raise RuntimeError('population/question closure failure')
 OUT.mkdir(parents=True);(OUT/'inputs').mkdir();store=FormalStore(OUT,EXPERIMENT,50.0);store.initialise()
 videos=[];routes=[]
 for vid in dict.fromkeys(qv[q] for q in ids):
  r1=SOURCE/'video_assets'/vid/'r1_av_navigation_map.json';r3=R3/vid/'r3_2_navigation_map.json';h=R3/vid/'shared_hierarchy.json'
  if not all(x.is_file() for x in (r1,r3,h)):raise RuntimeError('missing frozen asset')
  qids=[q for q in ids if qv[q]==vid]; videos.append({'video_id':vid,'question_ids':qids,'r1_map':str(r1),'r1_sha256':sha(r1),'r3_map':str(r3),'r3_sha256':sha(r3),'hierarchy':str(h),'hierarchy_sha256':sha(h)})
  for method,path in [('R1',r1),('R3',r3)]:
   for q in qids:
    r=rows[q];doc={'question_id':q,'video_uid':vid,'question_text':r['question'],'answer_options':[{'option_id':chr(65+i),'text':r[f'answer_{i+1}']} for i in range(5)]};atomic_json(OUT/'inputs'/f'{q}.json',doc);routes.append({'route_id':f'{method}:{q}','method':method,'question_id':q,'video_id':vid,'map_path':str(path),'map_sha256':sha(path)})
 fp=fingerprints(ROOT,cfg);manifest={'experiment_id':EXPERIMENT,'population_manifest':str(ROOT/'audit/eval300_population_manifest.json'),'population_manifest_sha256':sha(ROOT/'audit/eval300_population_manifest.json'),'question_count':300,'route_count':600,'videos':videos,'routes':routes,'fingerprints':fp,'formal_hard_budget_usd':50.0,'gold_loaded':False,'state':'dry_materialised'};atomic_json(OUT/'formal_manifest.json',manifest);atomic_json(OUT/'dry_validation.json',{'valid':len(routes)==600 and sum(x['method']=='R1' for x in routes)==300 and sum(x['method']=='R3' for x in routes)==300,'provider_attempts':0,'terminal_routes':0,'starting_spend_usd':store.spend(),'gold_loaded':False})
 print(OUT/'formal_manifest.json')
if __name__=='__main__':main()
