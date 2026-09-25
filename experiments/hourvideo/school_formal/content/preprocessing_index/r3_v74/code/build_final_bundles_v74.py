"""Build and round-trip verify the two local V7.4 transfer bundles."""
from __future__ import annotations
import hashlib, json, os, pathlib, re, shutil, socket, tarfile, tempfile, time

HERE=pathlib.Path(__file__).resolve().parent; WORK=HERE.parent; ROOT=WORK.parent
PKG=ROOT/'packages'; IMM=ROOT/'immutable_sources'; INDEX=ROOT/'work_index'
NAMES={'runtime':'hourvideo_v7_4_variant_c_budgets_v1_runtime_final_v3','index':'hourvideo_v7_4_variant_c_budgets_v1_index_eval300_12video_final_v3'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def regular(root): return sorted(p for p in root.rglob('*') if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.suffix!='.pyc')
def copy(src,dst): dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
def runtime_sources():
 out=[]
 for p in regular(WORK/'runnable_runtime'): out.append((p,pathlib.Path('runnable_runtime')/p.relative_to(WORK/'runnable_runtime')))
 selected=['DATA_MACHINE_RUNBOOK.md','DATA_MACHINE_TRANSFER_AND_PREFLIGHT.md','DIFF_FROM_REFERENCE_48.md','README.md','config_v7_4.json','run.py','offline_contract_v74.py','test_summarizer_contract_v74.py','pre_package_audit_v74.py','test_contract.py','pre_eval300_audit.py','paired_eval_statistics.py','build_final_bundles_v74.py']
 for name in selected: out.append((HERE/name,pathlib.Path('experiment')/name))
 for rel in ['src/experiments/planner_medium_retrieval/__init__.py','src/experiments/planner_medium_retrieval/core.py','src/experiments/planner_medium_retrieval/schemas.py','src/experiments/planner_medium_retrieval/validation.py','src/experiments/hourvideo_v7_4_variant_c_budgets_v1/__init__.py','src/experiments/hourvideo_v7_4_variant_c_budgets_v1/retriever.py']:
  out.append((WORK/'runtime_support'/rel,pathlib.Path('runtime_support')/rel))
 for name in ['reference_to_final_code_manifest.json','provenance_audit_v2.json','isolation_completion_summary_v2.json']:
  out.append((ROOT/'manifests'/name,pathlib.Path('audit')/name))
 for name in ['offline_contract_v74.json','summarizer_contract_v74.json','pre_package_audit_v74.json']:
  out.append((ROOT/'outputs'/name,pathlib.Path('audit')/name))
 return out
def index_sources():
 out=[(p,pathlib.Path('work_index')/p.relative_to(INDEX)) for p in regular(INDEX)]
 prov=IMM/'indexer_build_provenance'; out += [(p,pathlib.Path('indexer_build_provenance')/p.relative_to(prov)) for p in regular(prov)]
 for name in ['provenance_audit_v2.json','isolation_completion_summary_v2.json']:
  out.append((ROOT/'manifests'/name,pathlib.Path('audit')/name))
 return out
def scan(stage,kind):
 bad=[]; secret=re.compile(rb'(sk-[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA|OPENSSH) PRIVATE KEY-----|data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=]{100,})')
 blocked_ext={'.mp4','.mkv','.avi','.mov','.webm','.jpg','.jpeg','.png','.webp','.gif','.safetensors','.pt','.pth','.bin'}
 for p in regular(stage):
  rel=p.relative_to(stage); lower=[x.lower() for x in rel.parts]
  if any(x in {'.env','cache','frames','videos','models','predictions','trajectories','metrics'} for x in lower): bad.append([str(rel),'forbidden_path'])
  if p.suffix.lower() in blocked_ext: bad.append([str(rel),'forbidden_extension'])
  data=p.read_bytes()
  if secret.search(data): bad.append([str(rel),'secret_or_embedded_image'])
  if re.search(rb'[0-9a-f]{8}-[0-9a-f-]{27}_[0-9]+_[0-9]+',data,re.I): bad.append([str(rel),'question_uid_value'])
 return bad
def build(kind,sources):
 name=NAMES[kind]; archive=PKG/f'{name}.tar.gz'; contents=PKG/f'{name}.contents.txt'; side=PKG/f'{name}.manifest.json'
 for p in (archive,contents,side):
  if p.exists(): raise FileExistsError(f'refusing to overwrite {p}')
 with tempfile.TemporaryDirectory(prefix=f'{name}-',dir=PKG) as td:
  stage=pathlib.Path(td)/name; stage.mkdir(); sanitizations=[]
  for src,rel in sources:
   if not src.is_file(): raise FileNotFoundError(src)
   dst=stage/rel; copy(src,dst)
   if kind=='index' and rel.name=='source_artifact_audit.json':
    dst.chmod(0o644)
    obj=json.loads(dst.read_text()); old=obj.pop('question_id',None)
    if old is not None:
     obj['question_id_removed_for_transfer']=True; obj['source_sha256_before_sanitization']=sha(src)
     dst.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n'); sanitizations.append(str(rel))
  bad=scan(stage,kind)
  if bad: raise RuntimeError(f'{kind} scan failed: {bad[:10]}')
  records=[{'path':str(p.relative_to(stage)),'size_bytes':p.stat().st_size,'sha256':sha(p)} for p in regular(stage)]
  manifest={'bundle':name,'kind':kind,'created_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'hostname':socket.gethostname(),'payload_file_count':len(records),'payload_size_bytes':sum(x['size_bytes'] for x in records),'files':records,'sanitized_question_id_records':sanitizations,'scans':{'whitelist':True,'path_traversal':True,'symlink_count':0,'secret_env':True,'qa_uid_answer':True,'model_video_frame_cache':True}}
  (stage/'MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
  (stage/'CONTENTS.sha256').write_text(''.join(f"{x['sha256']}  {x['path']}\n" for x in records))
  with tarfile.open(archive,'w:gz') as tf: tf.add(stage,arcname=name,recursive=True)
  all_records=[{'path':str(p.relative_to(stage)),'size_bytes':p.stat().st_size,'sha256':sha(p)} for p in regular(stage)]
  contents.write_text(''.join(f"{x['sha256']}  {x['path']}\n" for x in all_records))
  verify_dir=pathlib.Path(td)/'verify'; verify_dir.mkdir()
  with tarfile.open(archive,'r:gz') as tf:
   members=tf.getmembers()
   if any(m.issym() or m.islnk() or pathlib.PurePosixPath(m.name).is_absolute() or '..' in pathlib.PurePosixPath(m.name).parts for m in members): raise RuntimeError('unsafe archive member')
   tf.extractall(verify_dir)
  extracted=verify_dir/name; got={str(p.relative_to(extracted)):sha(p) for p in regular(extracted)}; expected={x['path']:x['sha256'] for x in all_records}
  missing=sorted(set(expected)-set(got)); extra=sorted(set(got)-set(expected)); mismatch=sorted(k for k in expected.keys()&got.keys() if expected[k]!=got[k])
  if missing or extra or mismatch: raise RuntimeError(f'roundtrip mismatch {missing=} {extra=} {mismatch=}')
  side_obj={'archive':str(archive),'archive_sha256':sha(archive),'archive_size_bytes':archive.stat().st_size,'archive_file_count':len(all_records),'payload_file_count':len(records),'roundtrip':{'missing':0,'extra':0,'mismatch':0},'manifest':manifest}
  side.write_text(json.dumps(side_obj,indent=2,sort_keys=True)+'\n')
  return side_obj
def main():
 PKG.mkdir(exist_ok=True)
 runtime=build('runtime',runtime_sources()); index=build('index',index_sources())
 report={'status':'PASS','runtime':runtime,'index':index,'api_called':False,'model_loaded':False,'online_run_started':False}
 (PKG/'package_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 (PKG/'PACKAGE_REPORT.md').write_text(
  '# V7.4 package report\n\n'
  f"- Runtime: `{runtime['archive']}`\n- Runtime SHA-256: `{runtime['archive_sha256']}`\n"
  f"- Runtime files: {runtime['archive_file_count']}; bytes: {runtime['archive_size_bytes']}\n"
  f"- Index: `{index['archive']}`\n- Index SHA-256: `{index['archive_sha256']}`\n"
  f"- Index files: {index['archive_file_count']}; bytes: {index['archive_size_bytes']}\n"
  '- Both round-trip checks: 0 missing, 0 extra, 0 mismatch.\n'
 )
 shutil.copy2(HERE/'DATA_MACHINE_TRANSFER_AND_PREFLIGHT.md',PKG/'DATA_MACHINE_TRANSFER_AND_PREFLIGHT.md')
 print(json.dumps({'status':'PASS','runtime':{k:runtime[k] for k in ['archive','archive_sha256','archive_size_bytes','archive_file_count']},'index':{k:index[k] for k in ['archive','archive_sha256','archive_size_bytes','archive_file_count']}},sort_keys=True))
if __name__=='__main__': main()
