#!/usr/bin/env python3
"""Read-only integrity checks for the frozen ABD evidence-audit inputs."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "outputs/abd_eval300_formal_v1"
INPUTS = ROOT / "drafts/abd_direct_eval300_v1/inputs"
AUDIT = ROOT / "outputs/abd_eval300_evidence_audit_v1"

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def main():
    rows={a:{} for a in 'ABD'}
    for a in rows:
        for line in (INPUTS/f'{a}.jsonl').read_text().splitlines():
            d=json.loads(line); rows[a][d['question_id']]=d
    starts=[json.loads(x) for x in (FORMAL/'journals/request_starts.jsonl').read_text().splitlines() if x]
    responses=[json.loads(x) for x in (FORMAL/'journals/provider_responses.jsonl').read_text().splitlines() if x]
    parsers=[json.loads(x) for x in (FORMAL/'journals/parser_results.jsonl').read_text().splitlines() if x]
    statuses=list((FORMAL/'task_status').glob('*.json'))
    status=[json.loads(x.read_text()) for x in statuses]
    maps=sum(rows['A'][q].get('map') != rows['D'][q].get('map') for q in rows['A'])
    images=sum(rows['B'][q].get('images',[]) != rows['D'][q].get('images',[]) for q in rows['B'])
    core=lambda d: (d['question'],d['options'],d.get('images',[]))
    bd=sum(core(rows['B'][q]) != core(rows['D'][q]) for q in rows['B'])
    payload={
      'schema_version':'abd_evidence_audit_input_integrity_v1', 'gold_loaded':False,
      'historical_audit_labels_loaded':False, 'counts':{'A':len(rows['A']),'B':len(rows['B']),'D':len(rows['D']),'tasks':len(starts),'responses':len(responses),'parser_results':len(parsers),'task_status_files':len(status)},
      'terminal_success':sum(x.get('state')=='terminal_success' for x in status),
      'valid_answer':sum(x.get('result_class')=='valid_answer' for x in status),
      'duplicate_task_ids':len(starts)-len({x['task_id'] for x in starts}),
      'duplicate_attempt_ids':len(starts)-len({x['attempt_id'] for x in starts}),
      'duplicate_response_ids':len(responses)-len({x.get('response_sha256') for x in responses}),
      'alignment':{'A_D_map_mismatches':maps,'B_D_image_mismatches':images,'B_D_nonmap_input_mismatches':bd},
      'anchor_files':{
        'scientific_manifest_sha256':'24e6305d798378c352099ee963b00efb54d5b40b545c34221949dfba36cd36dc',
        'raw_closure_sha256':'b16175b602bd62e82dd42e44ec5a4249654acf0d75a90a2fda623ad81057649d',
        'archive_sha256':'d9f1273d1463b1a68e6a2a60455895072d27bc05f157dabfd89ae7f73af45c31'},
      'status':'PASS' if len(starts)==len(responses)==len(parsers)==len(status)==900 and maps==images==bd==0 and all(x.get('state')=='terminal_success' for x in status) and all(x.get('result_class')=='valid_answer' for x in status) else 'FAIL'
    }
    (AUDIT/'raw_integrity.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
