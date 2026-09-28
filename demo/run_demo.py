#!/usr/bin/env python3
"""Two read-only thesis replays; full fresh live execution is not implemented."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]


class DemoError(Exception):
    pass


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        # argparse normally echoes unknown argument values, possibly a secret.
        self.exit(2, 'Invalid command-line arguments. Use --help. Credentials must be environment-only.\n')


def contained(root, relative):
    root = root.resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise DemoError("Path escapes its permitted root.")
    return path


def verify(path, digest):
    if not path.is_file():
        raise DemoError("Required source/asset is missing; see the manifest and asset guide.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise DemoError("Source/asset SHA256 mismatch. No automatic repair is performed.")


def load_bundle(name, root=ROOT):
    manifest = json.loads((root / 'demo/examples' / name).read_text())
    if manifest['schema_version'] != 1:
        raise DemoError('Unsupported demo manifest schema.')
    data = {}
    for role, item in manifest['sources'].items():
        path = contained(root, item['path'])
        verify(path, item['sha256'])
        if path.suffix == '.json':
            data[role] = json.loads(path.read_text())
        elif path.suffix == '.jsonl':
            data[role] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        elif path.suffix == '.csv':
            with path.open(encoding='utf-8', newline='') as handle:
                data[role] = list(csv.DictReader(handle))
    return manifest, data


def require(condition, message='Frozen identity mismatch.'):
    if not condition:
        raise DemoError(message)


def excerpt(value, limit=240):
    value = ' '.join(str(value).split())
    return value if len(value) <= limit else value[:limit] + ' ... [excerpt]'


def replay_staged(manifest, data):
    qid, route, video = (manifest[k] for k in ('question_id', 'route', 'video_uid'))
    require(manifest['expected_identity'] == {'question_id': qid, 'route': route})
    require(data['config']['experiment'] == manifest['experiment'])
    question = data['input']['question']
    planner, investigation, answer = data['planner'], data['investigation'], data['answer']['answer']
    for obj in (question, planner, investigation, answer, data['status']):
        require(obj['question_id'] == qid)
    require(question['video_uid'] == planner['video_uid'] == data['status']['video_uid'] == video)
    require(planner['route'] == data['status']['side'] == route)
    require(data['status']['execution_status'] == 'normal_success')
    filtered = {}
    for role in ('attempts', 'events'):
        rows = data[role]
        require(all(r['question_id'] == qid and r['video_uid'] == video for r in rows))
        filtered[role] = [r for r in rows if r['route'] == route]
        require(bool(filtered[role]), 'No exact-identity telemetry rows.')
        require(all(r.get('side', route) == route for r in filtered[role]))
    end = [r for r in filtered['events'] if r['event'] == 'route_end']
    require(len(end) == 1, 'Ambiguous terminal event.')
    evidence = {e['evidence_id']: e for e in investigation['evidence']}
    require(all(e in evidence for e in answer['supporting_evidence_ids']))
    plan = next(p for p in planner['output']['requirement_plans']
                if p['requirement_id'].endswith('option_' + answer['selected_option_id'].lower()))
    cited = [evidence[i] for i in investigation['cited_evidence_ids']]
    def short(e):
        return e['evidence_id'].split('::')[-1]
    def chain(subject):
        coarse = next(e for e in cited if e['evidence_type'] == 'semantic_coarse_summary'
                      and subject.lower() in e['source_content'].lower() and short(e) in plan['selected_coarse_ids'])
        medium = next(e for e in cited if e['evidence_type'] == 'vlm_caption'
                      and subject.lower() in e['source_content'].lower()
                      and coarse['interval'][0] <= e['interval'][0] <= e['interval'][1] <= coarse['interval'][1])
        observations = [e for e in cited if e['evidence_type'] == 'reviewed_visual_observation'
                        and subject.lower() in json.loads(e['source_content'])['finding'].lower()
                        and medium['interval'][0] <= e['timestamp_sec'] <= medium['interval'][1]]
        return ' → '.join(short(e) for e in [coarse, medium, *observations])
    stages = {r['stage'] for r in filtered['attempts']}
    require({'shared_investigation', 'claim_execution_batch', 'direct_final'} <= stages)
    print('MODE: API-Planner + Local Downstream Frozen Pipeline Replay')
    print('Read-only preserved formal run; not a fully local Planner run.')
    print('No model inference is performed.')
    print('No retrieval, visual inspection, or experiment is re-run.')
    print('EXPERIMENT: ' + manifest['experiment'])
    print(f'Question ID: {qid}; route: {route}')
    print('PIPELINE SUMMARY')
    print('Question → Planner for option ' + answer['selected_option_id'] + ': ' + ' + '.join(plan['selected_coarse_ids']))
    print('→ Phone: ' + chain('Phone'))
    print('→ Kettle: ' + chain('Kettle'))
    print('→ Investigation ' + investigation['investigation_status'] + ' → Answer ' + answer['selected_option_id'])
    print('QUESTION\n' + question['question_text'])
    print(' | '.join(f"{o['option_id']}: {o['text']}" for o in question['answer_options']))
    print('PREPARED REPRESENTATION')
    print('Recorded coarse/medium/fine excerpts; full source map is not loaded.')
    print('Recorded map SHA256 (reference only): ' + planner['input_asset_sha256'])
    print('RECORDED PLANNER DECISION / RATIONALE')
    print(' | '.join(f"{p['requirement_id'].split('::')[-1]}: {', '.join(p['selected_coarse_ids'])}" for p in planner['output']['requirement_plans']))
    print('Recorded rationale: ' + plan['selection_reason'])
    print('Note: model-authored rationale uses ordinal wording "region 3/8"; structured recorded selections are ' + '/'.join(plan['selected_coarse_ids']) + '. Original wording is preserved.')
    print('RECORDED DECISION TRACE (presentation summary)')
    print('Planner → Shared investigation → Evidence inspection → Investigation ' + investigation['investigation_status'] + ' → Direct final answer')
    print('Exact route events: route_events.jsonl; low-level model calls: model_attempts.jsonl.')
    print('HIERARCHICAL RETRIEVAL / RECORDED EVIDENCE')
    for identity in investigation['cited_evidence_ids']:
        e = evidence[identity]
        content = e['source_content']
        if e['evidence_type'] == 'reviewed_visual_observation':
            content = json.loads(content)['finding']
        support = ' | recorded supporting inspection; not a final citation' if e['evidence_type'] == 'reviewed_visual_observation' and identity not in answer['supporting_evidence_ids'] else ''
        print(f"{identity.split('::')[-1]} | {e.get('interval', e.get('timestamp_sec'))} s | {e['evidence_type']}" + support)
        print('  ' + excerpt(content, 180))
    print('INSPECTION / SUFFICIENCY')
    print('Recorded investigation status: ' + investigation['investigation_status'])
    print('Recorded established facts: ' + excerpt(investigation['established_facts']))
    print('FROZEN FINAL ANSWER')
    print(answer['selected_option_id'] + ' — ' + answer['answer_text'])
    print('Recorded rationale: ' + excerpt(answer['reason']))
    print('Final citations: ' + ', '.join(e.split('::')[-1] for e in answer['supporting_evidence_ids']))
    print('Termination: ' + answer['final_status'] + '; correctness: not asserted by this replay.')
    print('RECORDED RESOURCE USE (not replay resource use)')
    usage = planner['usage']
    print(f"Planner: {usage['input_tokens']} input / {usage['output_tokens']} output tokens; estimated USD {usage['estimated_cost_usd']}")
    rows = filtered['attempts']
    print(f"Post-Planner: {sum(r['input_tokens'] for r in rows)} input / {sum(r['output_tokens'] for r in rows)} output tokens; {sum(r['physical_image_transmissions'] for r in rows)} image transmissions")
    print(f"Post-Planner elapsed: {end[0]['e2e_sec']:.3f} s; scope: {end[0]['e2e_scope']}")
    print('PROVENANCE — all source SHA256 checks passed')
    print('Manifest / full paths and hashes: demo/examples/local-staged-replay.json')
    print('Artifacts: ' + ', '.join(Path(manifest['sources'][role]['path']).name for role in ('planner', 'investigation', 'answer', 'attempts', 'events')))
    print('LIMITATIONS')
    print(' '.join(manifest['known_limitations']))


def main(argv=None):
    parser = SafeParser(description=__doc__)
    parser.add_argument('--mode', choices=('replay-staged', 'replay-direct', 'live'), required=True)
    args = parser.parse_args(argv)
    try:
        if args.mode == 'replay-staged':
            replay_staged(*load_bundle('local-staged-replay.json'))
        elif args.mode == 'replay-direct':
            from replay_direct import replay_direct
            replay_direct(*load_bundle('replay-direct.json'))
        else:
            print('FULL LIVE MODE IS NOT IMPLEMENTED YET.')
            print('See docs/USAGE.md for the planned scope and reproduction requirements.')
            return 2
        return 0
    except DemoError as error:
        print('Demo stopped: ' + str(error), file=sys.stderr)
        return 2
    except Exception:
        # Never emit provider payloads, exception strings, paths or credentials.
        print('Demo stopped: validation or execution failed. No exception payload is displayed.', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.modules['run_demo'] = sys.modules[__name__]
    raise SystemExit(main())
