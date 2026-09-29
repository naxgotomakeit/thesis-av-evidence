#!/usr/bin/env python3
import csv, glob, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
PREP=Path('/cs/student/project_msc/2025/rai/xinanx01/r1_visual_only_evidence_audit_preparation_v1')
THESIS=Path('/cs/student/project_msc/2025/rai/xinanx01/direct_visual_only_eval300_thesis_data_v1')
OLD=Path('/cs/student/project_msc/2025/rai/xinanx01/msc_thesis/main_system/isolated_workspaces/hourvideo_direct_api_eval300_v1/audit/codex_evidence_audit_r1_r3_gens_v3_v1/evidence_audit_frozen.jsonl')

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def load_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]

schema=json.loads((PREP/'audit_inputs/review_output_schema.json').read_text())
new=[]
for p in sorted((HERE/'reviews').glob('B*.jsonl')): new += load_jsonl(p)
assert len(new)==175 and len({x['neutral_id'] for x in new})==175
assert {x['neutral_id'] for x in new}=={f'E{i:04d}' for i in range(1,176)}
for x in new:
    assert set(schema['required_fields']) <= set(x)
    assert x['option_support'] in schema['option_support']
    assert x['confidence'] in schema['confidence']
    assert x['direct_evidence_source'] in schema['direct_evidence_source']
    assert all(v in schema['reason_basis'] for v in x['reason_basis'])
    if x['option_support']!='no_final_answer': assert x['direct_evidence_source']!='not_applicable'
assert sum(x['option_support']=='no_final_answer' for x in new)==1

mapping={x['neutral_id']:x for x in json.loads((PREP/'audit_inputs/identity_mapping_PRIVATE.json').read_text())['rows']}
reuse=json.loads((PREP/'reuse_125_label_validation.json').read_text())
assert reuse['count']==reuse['directly_reusable']==125 and not reuse['gold_loaded']
assert all(x['label_reusable'] for x in reuse['rows'])
old_by_id={x['neutral_id']:x for x in load_jsonl(OLD)}

records=[]
for x in new:
    m=mapping[x['neutral_id']]
    rec={'question_id':m['question_id'],'audit_origin':'new_visual_only_review','review_neutral_id':x['neutral_id'],**x}
    rec['label_source_path']=str(HERE/'reviews')
    records.append(rec)
for r in reuse['rows']:
    o=old_by_id[r['old_neutral_id']]
    rec={'question_id':r['question_id'],'audit_origin':'reused_old_verified','review_neutral_id':r['old_neutral_id'],**o}
    rec['label_source_path']=r['old_label_source_path']; rec['label_source_sha256']=r['old_label_source_sha256']
    records.append(rec)
assert len(records)==300 and len({x['question_id'] for x in records})==300
records.sort(key=lambda x:x['question_id'])
frozen=HERE/'r1_visual_only_evidence_labels_frozen.jsonl'
frozen.write_text(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in records))

# Gold/correctness is loaded only after the 300-label artifact above exists and validates.
with open(THESIS/'direct_visual_only_eval300_per_question.csv',newline='') as f: rows={x['question_id']:x for x in csv.DictReader(f)}
assert set(rows)=={x['question_id'] for x in records}
detail=[]; by=defaultdict(lambda:Counter(total=0,correct=0))
for x in records:
    d=rows[x['question_id']]; label=x['option_support']; correct=d['new_r1_correct'].lower()=='true'
    by[label]['total']+=1; by[label]['correct']+=int(correct)
    detail.append({'question_id':x['question_id'],'option_support':label,'audit_origin':x['audit_origin'],'r1_status':d['new_r1_status'],'r1_answer':d['new_r1_answer'],'r1_correct':correct,'source_artifact':d['new_r1_source_path'],'source_sha256':d['new_r1_source_sha256']})
with open(HERE/'r1_visual_only_evidence_labels_with_outcomes.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=detail[0].keys()); w.writeheader(); w.writerows(detail)
summary={'total':300,'new_review':175,'new_answer_reviewed':174,'no_final_answer':1,'reused_verified':125,'label_counts':dict(Counter(x['option_support'] for x in records)),'by_label':{k:dict(v) for k,v in sorted(by.items())},'frozen_labels_sha256':sha(frozen),'old_frozen_labels_sha256':sha(OLD),'gold_linkage_source':str(THESIS/'direct_visual_only_eval300_per_question.csv'),'gold_linkage_source_sha256':sha(THESIS/'direct_visual_only_eval300_per_question.csv')}
(HERE/'r1_visual_only_evidence_audit_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(summary,indent=2,ensure_ascii=False))
