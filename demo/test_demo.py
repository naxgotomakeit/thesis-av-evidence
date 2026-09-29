"""Offline tests: corrupt copies in memory, never preserved artifacts."""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
import run_demo as demo
from replay_direct import replay_direct


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.network = patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden'))
        self.network.start()
        self.addCleanup(self.network.stop)

    def test_both_replays_without_network(self):
        for mode in ('replay-staged', 'replay-direct'):
            with self.subTest(mode=mode), contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(demo.main(['--mode', mode]), 0)
            expected = ('No inference, API calls, retrieval, or inspection are re-run.'
                        if mode == 'replay-staged' else 'No model inference is performed.')
            self.assertIn(expected, out.getvalue())
            self.assertLessEqual(len(out.getvalue().splitlines()), 80)
        self.assertIn('E — Bowl', out.getvalue())
        self.assertIn('correctness: True', out.getvalue())
        self.assertIn('payload is not preserved', out.getvalue())

    def test_different_cwd(self):
        for mode in ('replay-staged', 'replay-direct'):
            result = subprocess.run([sys.executable, '-B', str(demo.ROOT/'demo/run_demo.py'), '--mode', mode],
                                    cwd=tempfile.gettempdir(), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_live_is_inert(self):
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY':'private-sentinel'}), patch.object(demo, 'load_bundle', side_effect=AssertionError('No assets should load')), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(demo.main(['--mode', 'live']), 2)
        self.assertEqual(out.getvalue(), 'FULL LIVE MODE IS NOT IMPLEMENTED YET.\nSee docs/USAGE.md for the planned scope and reproduction requirements.\n')
        self.assertNotIn('private-sentinel', out.getvalue())

    def test_obsolete_cli_rejected_without_echoing_values(self):
        for args in (['--mode','replay'], ['--mode','live-direct'],
                     ['--mode','live','--api-key','private-sentinel'],
                     ['--mode','live','--asset-root','private-sentinel'],
                     ['--mode','live','--save-output','private-sentinel']):
            with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit) as cm:
                demo.main(args)
            self.assertEqual(cm.exception.code, 2)
            self.assertNotIn('private-sentinel', err.getvalue())

    def test_source_missing_and_hash_mismatch_through_loader(self):
        for name in ('local-staged-replay.json', 'replay-direct.json'):
            original, _ = demo.load_bundle(name)
            for failure in ('missing', 'hash', 'escape'):
                with self.subTest(name=name, failure=failure), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    (root/'demo/examples').mkdir(parents=True)
                    (root/'source.json').write_text('{}')
                    source = {'path':'source.json', 'sha256':'0'*64}
                    if failure == 'missing':
                        source['path'] = 'absent.json'
                    if failure == 'escape':
                        source['path'] = '../outside.json'
                    original['sources'] = {'test':source}
                    (root/'demo/examples'/name).write_text(json.dumps(original))
                    with self.assertRaises(demo.DemoError):
                        demo.load_bundle(name, root)

    def test_direct_identity_conflicts_before_presentation(self):
        manifest, data = demo.load_bundle('replay-direct.json')
        cases = []
        for field in ('question_id', 'route', 'method', 'video_uid'):
            m = copy.deepcopy(manifest)
            m[field] = 'wrong'
            cases.append((m, data))
        for role, field, value in [('trace','method','R1'), ('trace','question_id','wrong'),
                                   ('status','route_id','R1:'+manifest['question_id'])]:
            d = copy.deepcopy(data)
            d[role][field] = value
            cases.append((manifest, d))
        d = copy.deepcopy(data)
        d['question']['question']['question_id'] = 'wrong'
        cases.append((manifest, d))
        for m, d in cases:
            with self.assertRaises(demo.DemoError), contextlib.redirect_stdout(io.StringIO()) as out:
                replay_direct(m, d)
            self.assertEqual(out.getvalue(), '')

    def test_direct_mixed_scoring_and_ambiguous_reuse(self):
        m, d = demo.load_bundle('replay-direct.json')
        qid = m['question_id']
        bad = copy.deepcopy(d)
        row = next(r for r in bad['scored']['per_question'] if r['question_id']==qid and r['method']=='R3')
        row['method'] = 'R1'
        with self.assertRaises(demo.DemoError):
            replay_direct(m, bad)
        bad = copy.deepcopy(d)
        rows = bad['reuse']['reuse_r3']['routes']
        rows.append(copy.deepcopy(next(r for r in rows if r['question_id']==qid)))
        with self.assertRaises(demo.DemoError):
            replay_direct(m, bad)

    def test_direct_tool_frame_and_provenance_conflicts(self):
        m, d = demo.load_bundle('replay-direct.json')
        for target in ('frame','tool','provenance','score','resources'):
            bad = copy.deepcopy(d)
            if target == 'frame':
                bad['trace']['turns'][0]['resolved_frames'][0]['observed_sha256'] = '0'*64
            elif target == 'tool':
                bad['trace']['provider_attempts'][0]['response_metadata']['tool_metadata'][0]['normalised_input']['timestamps_sec'] = [999]
            elif target == 'provenance':
                row = next(r for r in bad['source_manifest'] if r['staging_path'].endswith('/'+m['route']+'.json') and '/route_artifacts/' in r['staging_path'])
                row['original_sha256'] = '0'*64
            elif target == 'score':
                next(r for r in bad['scored']['per_question'] if r['question_id']==m['question_id'] and r['method']=='R3')['correct'] = False
            else:
                bad['trace']['total_input_tokens'] += 1
            with self.subTest(target=target), self.assertRaises(demo.DemoError), contextlib.redirect_stdout(io.StringIO()) as out:
                replay_direct(m, bad)
            self.assertEqual(out.getvalue(), '')

    def test_staged_integrity_and_exact_telemetry_identity(self):
        m, d = demo.load_bundle('local-staged-replay.json')
        for field in ('question_id','route','experiment'):
            bad = copy.deepcopy(m)
            bad[field] = 'wrong'
            with self.assertRaises(demo.DemoError):
                demo.replay_staged(bad, d)
        bad = copy.deepcopy(d)
        bad['attempts'][0]['video_uid'] = 'wrong'
        with self.assertRaises(demo.DemoError):
            demo.replay_staged(m, bad)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            demo.replay_staged(m, d)
        self.assertIn('22145 input / 2808 output', out.getvalue())
        self.assertIn('API-Planner + Local Downstream Frozen Pipeline Replay', out.getvalue())

    def test_staged_presentation_preserves_recorded_content(self):
        m, d = demo.load_bundle('local-staged-replay.json')
        with contextlib.redirect_stdout(io.StringIO()) as out:
            demo.replay_staged(m, d)
        text = out.getvalue()
        headings = ['Replay 1 — Staged pipeline','\nQUESTION\n','\nPLANNER DECISION\n',
                    '\nRECORDED PLANNER RATIONALE','\nEVIDENCE TRACE\n',
                    '\nINSPECTION / SUFFICIENCY\n','\nFINAL ANSWER (frozen)\n',
                    '\nRECORDED RESOURCE USE','\nPROVENANCE / TECHNICAL DETAILS\n','\nLIMITATIONS\n']
        positions = [text.index(h) for h in headings]
        self.assertEqual(positions, sorted(positions))
        for plan in d['planner']['output']['requirement_plans']:
            option = plan['requirement_id'].split('::')[-1].removeprefix('option_').upper()
            self.assertIn(option + ' → ' + ' + '.join(plan['selected_coarse_ids']),text)
            if option == 'B': self.assertIn(plan['selection_reason'],text)
        self.assertIn('Phone: C04 → M012 → F036',text)
        self.assertIn('Kettle: C09 → M033',text)
        for e in d['investigation']['evidence']:
            if e['evidence_id'] in d['investigation']['cited_evidence_ids']:
                content = json.loads(e['source_content'])['finding'] if e['evidence_type']=='reviewed_visual_observation' else e['source_content']
                self.assertIn(demo.excerpt(content,180),text)
        self.assertIn('recorded supporting inspection; not a final citation',text)
        self.assertIn(d['investigation']['established_facts'],text)
        self.assertIn('none requested in the final snapshot; requested_coarse_ids=[]',text)
        answer=d['answer']['answer']
        self.assertIn(answer['reason'],text)
        self.assertIn('Final citations: '+', '.join(e.split('::')[-1] for e in answer['supporting_evidence_ids']),text)
        self.assertIn('→ Final answer generation',text)
        self.assertNotIn('→ Direct final answer',text)
        self.assertIn('Full source map exists in the archived Variant-C package',text)


if __name__ == '__main__':
    unittest.main()
