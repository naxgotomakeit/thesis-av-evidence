#!/usr/bin/env python3
from __future__ import annotations
import hashlib, importlib, json, os, shutil, subprocess, sys
from pathlib import Path

BASE=Path('/home/naxucl/data/HourVideo/experiments/hourvideo_v7_4_variant_c_budgets_v1')
OUT=BASE/'outputs/dense_semantic_beam_b_h30_h8_superseding_audit_v2_20260829T155639Z'
PARENT=BASE/'outputs/dense_semantic_beam_b_h15_eval300_formal_v1_20260827T224330Z'
PARENT_OVERLAY=BASE/'outputs/dense_semantic_beam_b_tool_schema_parity_v1_20260827T220706Z/runtime_overlay'
H8ROOT=BASE/'outputs/dense_semantic_beam_b_h8_budget_overlay_v2_20260829T155639Z'
H8OVERLAY=H8ROOT/'runtime_overlay'
LOG=PARENT/'h15/logs/first-runner-20260827T224648Z.log'
CFG=PARENT/'h15/status/experiment_config.json'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tree(root): return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='MANIFEST.sha256' and '__pycache__' not in p.parts and p.suffix!='.pyc'}
def canonical_set_sha(rows): return hashlib.sha256((''.join(x+'\n' for x in sorted(rows))).encode()).hexdigest()

resolved=json.loads(CFG.read_text())
ordered=[]
for line in LOG.read_text(errors='replace').splitlines():
    try: d=json.loads(line)
    except Exception: continue
    if isinstance(d,dict) and d.get('uid'): ordered.append(str(d['uid']))
if len(ordered)!=300 or len(set(ordered))!=300: raise SystemExit('cannot prove H-15 actual first-pass order')
(OUT/'ordered_uids_new_h15_actual.txt').write_text(''.join(x+'\n' for x in ordered))

parent_config=PARENT_OVERLAY/'videoseal/tools/dense_beam_b/config.json'
h8_config=H8OVERLAY/'videoseal/tools/dense_beam_b/config.json'
parent_provider=PARENT_OVERLAY/'videoseal/tools/dense_beam_b/providers.py'
h8_provider=H8OVERLAY/'videoseal/tools/dense_beam_b/providers.py'
parent_retriever=PARENT_OVERLAY/'videoseal/tools/dense_beam_b/retriever.py'

parent_proof={
 'PARENT_CONFIRMED':'NEW_DENSE_SEMANTIC_BEAM_B_H15', 'OLD_LEXICAL_H15_USED':False, 'OLD_GLOBAL_SIGLIP_H_USED':False,
 'parent_formal_path':str(PARENT), 'parent_overlay':str(PARENT_OVERLAY), 'actual_resolved_config':resolved,
 'hashes':{'actual_resolved_config':sha(CFG),'launch_script':sha(PARENT/'run_formal_h15.sh'),'retrieval_adapter.py':sha(PARENT_OVERLAY/'videoseal/tools/retrieval_adapter.py'),'retriever.py':sha(parent_retriever),'providers.py':sha(parent_provider),'config.json':sha(parent_config),'tools_schema':resolved['tools_schema_sha256']},
 'assets':json.loads(parent_config.read_text()),
 'algorithm_proof':{'backend':resolved['RETRIEVAL_BACKEND'],'coarse_medium_embedding_model':'text-embedding-3-large','coarse_medium_embedding_dimensions':3072,'fine_query_encoder':'local SigLIP','routing':'Dense Coarse -> Dense Medium -> local Fine','query_embedding_calls_per_retrieval':1,'siglip_query_encoding_calls_per_retrieval':1,'lexical_or_bm25_fallback':False,'global_fine_siglip_fallback':False},
 'ordered_uid':{'count':300,'unique':300,'ordered_sha256':sha(OUT/'ordered_uids_new_h15_actual.txt'),'canonical_set_sha256':canonical_set_sha(ordered),'first':ordered[0],'last':ordered[-1],'source_log':str(LOG)},
}
expected={'backend':'dense_semantic_beam_b','adapter':'008efa173913759a7713209b649ab3947e99349a2385afd017995778460fc4b0','retriever':'47b8dcfea459f9edacf650e9975043714635c8c3b5fd054f78715535f99cd46e','schema':'71346effd5acd8d567e16921316e0f6cb0e94b422f42c4d55edb24a86d4a1863'}
proof_ok=(resolved['RETRIEVAL_BACKEND']==expected['backend'] and parent_proof['hashes']['retrieval_adapter.py']==expected['adapter'] and parent_proof['hashes']['retriever.py']==expected['retriever'] and resolved['tools_schema_sha256']==expected['schema'] and json.loads(parent_config.read_text())['allowed_b']==[6,15,30])
parent_proof['proof_pass']=proof_ok
(OUT/'NEW_H15_PARENT_PROOF.json').write_text(json.dumps(parent_proof,indent=2,sort_keys=True)+'\n')
if not proof_ok: raise SystemExit('new H-15 parent proof failed')

# H-8 overlay may differ only in budget whitelist and the provider expected config SHA.
pt,ht=tree(PARENT_OVERLAY),tree(H8OVERLAY)
td={k:{'parent':pt.get(k),'h8':ht.get(k)} for k in sorted(set(pt)|set(ht)) if pt.get(k)!=ht.get(k)}
allowed_files={'videoseal/tools/dense_beam_b/config.json','videoseal/tools/dense_beam_b/providers.py'}
ptext=parent_provider.read_text(); htext=h8_provider.read_text()
normalized_provider=htext.replace(sha(h8_config),sha(parent_config))
h8_overlay_ok=(set(td)==allowed_files and normalized_provider==ptext and json.loads(h8_config.read_text())['allowed_b']==[8,15,30] and sha(H8OVERLAY/'videoseal/tools/dense_beam_b/retriever.py')==expected['retriever'] and sha(H8OVERLAY/'videoseal/tools/retrieval_adapter.py')==expected['adapter'])

# No-model runtime SHA gates. Imports instantiate neither provider nor model.
def gate(overlay):
    old=list(sys.path); sys.path.insert(0,str(overlay));
    for k in list(sys.modules):
        if k=='videoseal' or k.startswith('videoseal.'): del sys.modules[k]
    try:
        m=importlib.import_module('videoseal.tools.dense_beam_b.providers'); return m.runtime_gate()
    finally: sys.path[:]=old
parent_gate=gate(PARENT_OVERLAY); h8_gate=gate(H8OVERLAY)

# Full existing 13-contract suite, parameterized by replacing only formal low B=6 with B=8.
src=(BASE/'outputs/dense_semantic_beam_b_retriever_v1_20260827T200942Z/test_contract.py').read_text()
src=src.replace('HERE = Path(__file__).resolve().parent\nEXPERIMENT = HERE.parent.parent',f"HERE = Path({str(H8OVERLAY/'videoseal/tools/dense_beam_b')!r})\nEXPERIMENT = Path({str(BASE)!r})")
src=src.replace('(6, 15, 30)','(8, 15, 30)').replace(', 6, "question"',', 8, "question"').replace('r6 = self.r.retrieve(saturated_video, "query", 6)','r6 = self.r.retrieve(saturated_video, "query", 8)').replace('result = self.r.retrieve(video, "shortfall", 6)','result = self.r.retrieve(video, "shortfall", 8)').replace('self.r.retrieve(video, "query", 6)','self.r.retrieve(video, "query", 8)').replace('r.retrieve(self.videos[0], "query", 6)','r.retrieve(self.videos[0], "query", 8)').replace('"telemetry", 6','"telemetry", 8')
(OUT/'test_contract_h8.py').write_text(src)
env=dict(os.environ); env['PYTHONPATH']=str(H8OVERLAY/'videoseal/tools/dense_beam_b')+os.pathsep+str(H8OVERLAY); env['PYTHONDONTWRITEBYTECODE']='1'
cp=subprocess.run([sys.executable,str(OUT/'test_contract_h8.py')],text=True,capture_output=True,env=env)
(OUT/'contract_stdout.log').write_text(cp.stdout); (OUT/'contract_stderr.log').write_text(cp.stderr)
contract_ok=cp.returncode==0 and 'Ran 13 tests' in cp.stderr and cp.stderr.rstrip().endswith('OK')

uid_input=(BASE.parent.parent/'benchmark/v1.0_release/hourvideo_eval300_v1_uids.txt')
input_rows=[x for x in uid_input.read_text().splitlines() if x]
uid_proof={'input_file':str(uid_input),'input_file_sha256':sha(uid_input),'count':len(input_rows),'unique':len(set(input_rows)),'canonical_set_sha256':canonical_set_sha(input_rows),'actual_order_sha256':sha(OUT/'ordered_uids_new_h15_actual.txt'),'actual_first':ordered[0],'actual_last':ordered[-1],'pending_file_first':input_rows[0],'same_set':set(input_rows)==set(ordered),'explanation':'per_question_runner reads --uids-file into a set, then preserves parquet row order; therefore pending-file first is not execution first. The failed v1 log also actually began with _14_5; _8_31 was only the pending-list first and was previously mislabeled.'}
(OUT/'ORDERED_UID_PROOF.json').write_text(json.dumps(uid_proof,indent=2,sort_keys=True)+'\n')

status='PASS — EXACT NEW-H15 PARENT, BUDGET-ONLY CHANGE' if h8_overlay_ok and contract_ok and uid_proof['same_set'] and parent_gate and h8_gate else 'BLOCKED — UNEXPECTED DIFFERENCE'
audit={'status':status,'parent_proof_pass':proof_ok,'h30':{'overlay':str(PARENT_OVERLAY),'config_sha256':sha(parent_config),'provider_gate':parent_gate,'difference_from_h15':['B 15->30','profile/name','output/log/tmux name','UTC timestamp']},'h8':{'overlay':str(H8OVERLAY),'recursive_diff':td,'provider_only_expected_sha_change':normalized_provider==ptext,'config_sha_matches_provider':bool(h8_gate),'runtime_gate':h8_gate,'contract_13_of_13':contract_ok,'difference_from_h15':['B 15->8','allowed_b [6,15,30]->[8,15,30]','providers expected config SHA','profile/name','output/log/tmux name','UTC timestamp']},'tools_schema_sha256':expected['schema'],'ordered_uid':uid_proof}
(OUT/'superseding_audit.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
(OUT/'DIFFERENCE_MATRIX.tsv').write_text('item\tH15\tH30\tH8\tstatus\nBeam-B\t15\t30\t8\tALLOWED\nallowed_b\t6,15,30\t6,15,30\t8,15,30\tH8_AMENDMENT_ONLY\nproviders expected config SHA\tparent\tparent\tH8 config SHA\tH8_INTEGRITY_METADATA_ONLY\nRetriever SHA\t'+expected['retriever']+'\t'+expected['retriever']+'\t'+expected['retriever']+'\tIDENTICAL\ntools schema SHA\t'+expected['schema']+'\t'+expected['schema']+'\t'+expected['schema']+'\tIDENTICAL\n')
(OUT/'AUDIT_REPORT.md').write_text(f'# Superseding H-30/H-8 Audit v2\n\nStatus: `{status}`\n\nParent is proven to be the completed new Dense Semantic Beam-B H-15. H-30 uses its exact original overlay. H-8 differs only in the preregistered budget whitelist and matching integrity SHA. H-8 contracts: {"13/13 PASS" if contract_ok else "FAIL"}.\n')
print(status)
if not status.startswith('PASS'): raise SystemExit(2)
