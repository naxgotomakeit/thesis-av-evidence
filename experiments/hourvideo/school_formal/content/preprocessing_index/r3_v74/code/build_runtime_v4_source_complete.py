from __future__ import annotations
import hashlib,json,os,pathlib,platform,re,shutil,subprocess,sys,tarfile,tempfile,time
ROOT=pathlib.Path(__file__).resolve().parents[2]; WORK=ROOT/'work_src'
BASE=pathlib.Path('${LEGACY_RUNTIME}/src/experiments/hourvideo_v7_3_videoseal_eval300_runtime_aligned_paired_v1/reference_runtime')
SNAP=pathlib.Path('${LEGACY_RUNTIME}/outputs/experiments/hourvideo_v7_3_videoseal_eval300_runtime_aligned_paired_v1/actual_config_and_code_hash_snapshot.json')
REF48=ROOT/'immutable_sources/reference_runtime_48'; STAGE=ROOT/'runtime_final_v4_source_complete_staging'
PKG=ROOT/'packages'; NAME='hourvideo_v7_4_variant_c_budgets_v1_runtime_final_v4_source_complete'
PY=pathlib.Path('${SCHOOL_PROJECT_ROOT}/miniconda3/envs/qaego4d_vllm/bin/python')
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def files(p):return sorted(x for x in p.rglob('*') if x.is_file() and not x.is_symlink() and '__pycache__' not in x.parts and x.suffix!='.pyc')
def cp(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists():dst.chmod(0o644)
 shutil.copy2(src,dst)
def copytree(src,dst):
 for p in files(src):cp(p,dst/p.relative_to(src))
def main():
 if STAGE.exists() and (STAGE/'MANIFEST.json').exists():raise FileExistsError(STAGE)
 for p in [PKG/f'{NAME}.tar.gz',PKG/f'{NAME}.manifest.json',PKG/f'{NAME}.contents.txt']:
  if p.exists():raise FileExistsError(p)
 STAGE.mkdir(exist_ok=True); copytree(BASE,STAGE/'reference_runtime'); copytree(BASE,STAGE/'runnable_runtime')
 # Deterministic frozen runtime overlay, then V7.4 audited files.
 for p in files(REF48):cp(p,STAGE/'runnable_runtime'/p.relative_to(REF48))
 for rel in ['videoseal/tools/tool_map.py','videoseal/tools/visual_tools.py','videoseal/tools/retrieval_adapter.py']:
  cp(WORK/'runnable_runtime'/rel,STAGE/'runnable_runtime'/rel)
 copytree(WORK/'runtime_support',STAGE/'runtime_support')
 selected=['config_v7_4.json','run.py','offline_contract_v74.py','test_summarizer_contract_v74.py','pre_package_audit_v74.py','test_contract.py','DATA_MACHINE_RUNBOOK.md','DATA_MACHINE_TRANSFER_AND_PREFLIGHT.md']
 for n in selected:cp(WORK/'experiment'/n,STAGE/'experiment'/n)
 snap=json.loads(SNAP.read_text()); rec=[]
 replaced={'run.py','test_contract.py','retrieval_adapter.py','visual_tools.py'}
 for r in snap['code_hashes']:
  p=pathlib.Path(r['path']); actual=sha(p) if p.is_file() else None; status='matched' if actual==r['sha256'] else ('superseded_by_v7_4' if p.name in replaced else 'unresolved')
  rec.append({'path':str(p),'smoke_sha256':r['sha256'],'current_sha256':actual,'status':status})
 if any(x['status']=='unresolved' for x in rec):raise RuntimeError('unresolved smoke base hashes')
 smoke={'evidence_status':{'v7_3_base':'live_smoke_validated','v7_4_overlay':'offline_contract_validated','v7_4_live_smoke_status':'pending_data_machine_validation'},'snapshot':str(SNAP),'snapshot_records':len(rec),'matched':sum(x['status']=='matched' for x in rec),'superseded':sum(x['status']=='superseded_by_v7_4' for x in rec),'records':rec,'question_content_included':False}
 (STAGE/'smoke_actual_source_resolution.json').write_text(json.dumps(smoke,indent=2,sort_keys=True)+'\n')
 versions=subprocess.check_output([str(PY),'-c','import importlib.metadata as m,json,platform,sys; n=["torch","transformers","tokenizers","numpy","pyarrow","openai","requests","opencv-python","opencv-python-headless","vllm","pillow","scipy"]; print(json.dumps({"python":sys.version,"executable":sys.executable,"platform":platform.platform(),"architecture":platform.machine(),"packages":{x:m.version(x) for x in n if next(iter([True]),False)}}))'],text=True)
 env=json.loads(versions); (STAGE/'environment_smoke_source.json').write_text(json.dumps(env,indent=2,sort_keys=True)+'\n')
 (STAGE/'requirements_runner_v74.lock').write_text('\n'.join(f'{k}=={v}' for k,v in sorted(env['packages'].items()))+'\n')
 (STAGE/'DATA_MACHINE_ARM_ENVIRONMENT.md').write_text('# ARM runner environment\n\nSource smoke used Python 3.10 on x86_64. Create a fresh ARM/aarch64 environment; do not copy this environment or wheels. Match Python 3.10 and the locked Python package versions where ARM-compatible builds exist. Install the DGX/CUDA-compatible torch and vLLM builds first, then transformers, tokenizers, numpy, pyarrow, OpenCV, OpenAI and requests. Map the existing SigLIP cache for `google/siglip-base-patch16-224`; no model files are included.\n')
 diff=[]
 for p in files(STAGE/'reference_runtime'):
  rel=p.relative_to(STAGE/'reference_runtime'); q=STAGE/'runnable_runtime'/rel
  if not q.exists() or sha(p)!=sha(q):diff.append({'path':str(rel),'base_sha256':sha(p),'final_sha256':sha(q) if q.exists() else None})
 for p in files(STAGE/'runnable_runtime'):
  rel=p.relative_to(STAGE/'runnable_runtime')
  if not (STAGE/'reference_runtime'/rel).exists():diff.append({'path':str(rel),'base_sha256':None,'final_sha256':sha(p)})
 (STAGE/'reference_to_v7_4_complete_diff.json').write_text(json.dumps(diff,indent=2,sort_keys=True)+'\n')
 report={'v7_3_base':'live_smoke_validated','v7_4_overlay':'offline_contract_validated','v7_4_live_smoke_status':'pending_data_machine_validation','reference48_sha256':'ffcb174919b40dc3d278ce325e054c83716353c61de1c496600fbd960a51fe84','index_bundle_status':'unchanged_final_v3'}
 (STAGE/'SOURCE_COMPLETE_RUNTIME_REPORT.md').write_text('# Source-complete V7.4 runtime\n\n- V7.3 base: live smoke validated.\n- V7.4 overlay: offline contract validated.\n- V7.4 live smoke: pending data-machine validation.\n- Index final v3 is unchanged and is not included.\n')
 # Static scans before validation.
 bad=[]
 for p in files(STAGE):
  rel=str(p.relative_to(STAGE)); low=rel.lower(); data=p.read_bytes()
  if any(x in low for x in ['/.git/','/.venv/','three_question_uids','eval300_v1_uids']) or p.suffix.lower() in {'.mp4','.mkv','.jpg','.png','.safetensors','.pt','.pth'}:bad.append(rel)
  if re.search(rb'sk-[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA|OPENSSH) PRIVATE KEY-----',data):bad.append(rel)
 if bad:raise RuntimeError(f'forbidden payload: {bad[:10]}')
 # Validate in a fresh copy; only dependency site-packages may come from the interpreter.
 with tempfile.TemporaryDirectory(prefix='v74-source-complete-',dir=ROOT/'logs') as td:
  temp=pathlib.Path(td)/NAME; copytree(STAGE,temp)
  (pathlib.Path(td)/'outputs').mkdir()
  envv=dict(os.environ, PYTHONPATH=f"{temp/'runnable_runtime'}:{temp/'runtime_support'}:{temp/'runtime_support/src'}", INDEX_ROOT=str(ROOT/'work_index'))
  probe='import json,sys; mods=["videoseal","videoseal.tools.tool_map","videoseal.tools.visual_tools","videoseal.tools.retrieval_adapter","videoseal.utils.video.chunking","videoseal.prompts.caption_prompts","videoseal.agents.tool_agent","videoseal.utils.agent.tool_agent_parsing","videoseal.utils.video.frames","videoseal.utils.video.tooling","videoseal.cli.run_from_parquet","experiments.hourvideo_v7_4_variant_c_budgets_v1.retriever"]; r={};\nfor n in mods:\n m=__import__(n,fromlist=["*"]); r[n]=m.__file__; assert str(m.__file__).startswith("'+str(temp)+'")\nprint(json.dumps(r))'
  imports=subprocess.check_output([str(PY),'-c',probe],env=envv,cwd=temp,text=True)
  subprocess.check_call([str(PY),'-m','videoseal.cli.run_from_parquet','--help'],env=envv,cwd=temp,stdout=subprocess.DEVNULL)
  subprocess.check_call([str(PY),str(temp/'experiment/test_summarizer_contract_v74.py')],env=envv,cwd=temp,stdout=subprocess.DEVNULL)
  subprocess.check_call([str(PY),str(temp/'experiment/offline_contract_v74.py')],env=envv,cwd=temp,stdout=subprocess.DEVNULL)
  validation={'imports':json.loads(imports),'run_from_parquet_help':True,'summarizer_contract':'PASS','offline_contract':'PASS','api_called':False,'model_loaded':False}
 (STAGE/'source_complete_validation.json').write_text(json.dumps(validation,indent=2,sort_keys=True)+'\n')
 payload=[{'path':str(p.relative_to(STAGE)),'size_bytes':p.stat().st_size,'sha256':sha(p)} for p in files(STAGE)]
 manifest={**report,'payload_file_count':len(payload),'payload_size_bytes':sum(x['size_bytes'] for x in payload),'files':payload,'path_traversal':0,'symlink_hardlink_device':0}
 (STAGE/'MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n'); (STAGE/'CONTENTS.sha256').write_text(''.join(f"{x['sha256']}  {x['path']}\n" for x in payload))
 PKG.mkdir(exist_ok=True); archive=PKG/f'{NAME}.tar.gz'
 with tarfile.open(archive,'w:gz') as tf:tf.add(STAGE,arcname=NAME,recursive=True)
 allr=[{'path':str(p.relative_to(STAGE)),'sha256':sha(p),'size_bytes':p.stat().st_size} for p in files(STAGE)]
 (PKG/f'{NAME}.contents.txt').write_text(''.join(f"{x['sha256']}  {x['path']}\n" for x in allr))
 side={'archive':str(archive),'sha256':sha(archive),'size_bytes':archive.stat().st_size,'file_count':len(allr),'payload_file_count':len(payload),'v7_4_live_smoke_status':'pending_data_machine_validation','validation':validation}
 (PKG/f'{NAME}.manifest.json').write_text(json.dumps(side,indent=2,sort_keys=True)+'\n')
 print(json.dumps(side,sort_keys=True))
if __name__=='__main__':main()
