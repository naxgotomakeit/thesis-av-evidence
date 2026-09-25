"""Static, offline-only final audit before V7.4 packaging."""
import hashlib, json, pathlib
HERE=pathlib.Path(__file__).resolve().parent; WORK=HERE.parent; ROOT=WORK.parent
def h(p):
 d=hashlib.sha256(); d.update(p.read_bytes()); return d.hexdigest()
def main():
 ref=WORK/'reference_runtime'; final=WORK/'runnable_runtime'; changed=[]; missing=[]
 for p in sorted(x for x in ref.rglob('*') if x.is_file() and '__pycache__' not in x.parts):
  q=final/p.relative_to(ref)
  if not q.exists(): missing.append(str(p.relative_to(ref)))
  elif h(p)!=h(q): changed.append(str(p.relative_to(ref)))
 cfg=json.loads((HERE/'config_v7_4.json').read_text()); agent=(final/'videoseal/agents/tool_agent.py').read_text(); runner=(final/'videoseal/runner/per_question_runner.py').read_text(); adapter=(final/'videoseal/tools/retrieval_adapter.py').read_text()
 run_text=(HERE/'run.py').read_text()
 checks={
  'reference_files_missing':not missing,
  'only_audited_reference_files_changed':changed in (["videoseal/tools/tool_map.py","videoseal/tools/visual_tools.py"],["videoseal/tools/retrieval_adapter.py","videoseal/tools/visual_tools.py"]),
  'adapter_additive':(final/'videoseal/tools/retrieval_adapter.py').is_file(),
  'flat_top30':cfg['retrieval']['visual_top_k']==30,
  'summary_enabled':cfg['retrieval']['summary_enabled'] is True,
  'planner_visible_spans_disabled':'VISUAL_RETRIEVE_RETURN_SPANS": "0"' in (HERE/'run.py').read_text(),
  'max_steps_16':cfg['runtime']['max_steps']==16,
  'messages_0':cfg['planner']['api_use_messages']==0,
  'timeout_1000':cfg['runtime']['task_timeout_sec']==1000,
  'step15_reference_code_preserved':'step_idx == max_steps - 1' in agent,
  'fallback_64_reference_code_preserved':'INSPECT_MAX_TOTAL_IMAGES' in agent and '"64"' in agent and 'forced full-video visual_inspect fallback' in agent,
  'shared_summarizer_call':'reference_visual_tools.summarize_visual_retrieval_candidates' in adapter,
  'legacy_duplicate_not_runtime_called':'_legacy_summarize_like_reference_unused' in adapter,
  'deployment_index_paths_environment_first':'env.get("INDEX_ROOT")' in run_text and 'env.get("PAIRED_FLAT_INDEX_ROOT")' in run_text,
  'offline_preflight_index_root_portable':'os.getenv("INDEX_ROOT")' in (HERE/'offline_contract_v74.py').read_text() and "os.getenv('INDEX_ROOT')" in (HERE/'test_summarizer_contract_v74.py').read_text(),
 }
 summary=json.loads((ROOT/'outputs/summarizer_contract_v74.json').read_text()); offline=json.loads((ROOT/'outputs/offline_contract_v74.json').read_text())
 checks['summarizer_contract_pass']=summary['status']=='PASS'; checks['offline_contract_pass']=offline['status']=='PASS'
 report={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'changed_reference_paths':changed,'missing_reference_paths':missing,'tool_input':'visual_retrieve(query: string)','internal_raw_candidate':['start_time','end_time','caption'],'planner_visible_response':{'summary':'string'},'api_called':False,'model_loaded':False}
 (ROOT/'outputs/pre_package_audit_v74.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); print(json.dumps({'status':report['status'],'failed':[k for k,v in checks.items() if not v],'changed':changed}))
 raise SystemExit(0 if report['status']=='PASS' else 1)
if __name__=='__main__': main()
