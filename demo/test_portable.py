"""Offline integration tests: generated test video and mocked responses only."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import portable as p
import prepare
import ask


class MockTransport:
    requests = []
    def __init__(self,*args,**kwargs):
        self.calls = 0
        self.failed = False
    def create(self,**request):
        self.calls += 1
        self.requests.append(copy.deepcopy(request))
        usage = {'input_tokens':100,'output_tokens':30,'cache_creation_input_tokens':0,'cache_read_input_tokens':0}
        if 'tools' in request:
            name = 'inspect_frames' if self.calls == 1 else 'final_answer'
            value = {'timestamps_sec':[0,15],'reason':'Synthetic test only.'} if self.calls == 1 else {'selected_option_id':'B','reason':'Synthetic test only.'}
            return {'id':'test','type':'message','role':'assistant','model':p.MODEL,'stop_reason':'tool_use',
                    'content':[{'type':'tool_use','id':f'tool{self.calls}','name':name,'input':value}],'usage':usage}
        text = 'Synthetic test caption, not experimental evidence.'
        if 'output_config' in request:
            count = len(json.loads(request['messages'][0]['content'])['timeline'])
            text = json.dumps({'groups':[{'end_medium_index':count-1,'navigation_summary':text,'uncertainty_notes':[]}]})
        return {'id':'test','stop_reason':'end_turn','content':[{'type':'text','text':text}],'usage':usage}


class PortableTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base/'workspace'
        self.video = self.base/'test.mp4'
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=blue:s=64x64:r=1',
                        '-t','46','-pix_fmt','yuv420p',str(self.video)],check=True,capture_output=True)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch('socket.socket.connect',side_effect=AssertionError('Network forbidden')))
        self.stack.enter_context(patch.object(p,'workdir_path',return_value=self.root))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        MockTransport.requests = []
    def prep(self,execute=False,sha=None):
        return prepare.prepare(SimpleNamespace(video=str(self.video),workdir=str(self.root),caption_backend='api',execute=execute,approve_plan=sha))
    def ready(self):
        self.prep()
        plan = p.read(self.root/'plan.json')
        with patch.object(p,'credential',return_value='TEST_ONLY'), patch.object(p,'MessagesTransport',MockTransport):
            self.prep(True,p.object_sha(plan))
        return plan
    def test_frame_geometry_and_dry_requests(self):
        self.prep()
        plan = p.read(self.root/'plan.json')
        self.assertEqual((plan['frame_count'],plan['fine_count'],plan['medium_count']),(46,4,2))
        h = p.read(self.root/'shared_hierarchy.json')
        self.assertEqual([f['frame_index'] for f in h['fine_nodes']],[7,22,37,45])
        self.assertFalse((self.root/'preparation_manifest.json').exists())
        self.assertEqual(MockTransport.requests,[])
        for seconds,counts in [(120,(8,3)),(300,(20,7))]:
            h=p.geometry(seconds,'test')
            self.assertEqual((len(h['fine_nodes']),len(h['medium_nodes'])),counts)
    def test_consent_and_integrity(self):
        self.prep()
        with self.assertRaises(p.DemoError): self.prep(True,'wrong')
        with patch.object(p,'digest',return_value='changed'):
            with self.assertRaises(p.DemoError): self.prep()
        with patch.object(p,'verify',side_effect=p.DemoError('SHA mismatch')):
            with self.assertRaises(p.DemoError): self.prep()
    def test_full_mocked_fresh_pipeline(self):
        self.ready()
        with self.assertRaises(p.DemoError): self.prep()
        manifest=p.read(self.root/'preparation_manifest.json')
        self.assertFalse(manifest['model_identity_matches_thesis'])
        self.assertEqual(len(MockTransport.requests),3)
        q=self.base/'question.json'
        p.safe_write(q,{'question_id':'test-q','question_text':'Which color?',
                       'answer_options':{x:x for x in 'ABCDE'}})
        args=SimpleNamespace(workdir=str(self.root),question_json=str(q),execute=False,approve_plan=None)
        plans=[]
        with patch.object(p,'approval',side_effect=lambda plan,*a: plans.append(plan) or False): ask.ask(args)
        self.assertEqual(len(MockTransport.requests),3)
        args.execute=True
        args.approve_plan=p.object_sha(plans[0])
        with patch.object(p,'credential',return_value='TEST_ONLY'),patch.object(p,'MessagesTransport',MockTransport): ask.ask(args)
        result=p.read(next((self.base/'workspace-outputs/qa_runs').glob('*/result.json')))
        self.assertEqual(result['terminal_status'],'final_answer')
        self.assertEqual(result['final_prediction'],'B')
        self.assertEqual(result['transport_calls'],2)
        self.assertEqual(len(result['inspected_frames']),2)
        self.assertEqual(p.digest(self.root/'preparation_manifest.json'),plans[0]['workspace_sha256'])
    def test_missing_key_and_reserved_backend(self):
        with patch.dict('os.environ',{},clear=True):
            with self.assertRaises(p.DemoError): p.credential()
        with self.assertRaises(p.DemoError): p.caption_backend('local-qwen',{})
        with self.assertRaises(p.DemoError): ask.Adapter().create()
    def test_real_two_and_five_minute_extraction(self):
        for seconds in (120,300):
            video=self.base/f'test{seconds}.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=blue:s=64x64:r=1',
                            '-t',str(seconds),'-pix_fmt','yuv420p',str(video)],check=True,capture_output=True)
            info=p.probe(video)
            frames=p.extract(video,self.base/f'frames{seconds}',info['duration_sec'])
            self.assertEqual(len(frames),seconds)
    def test_secret_reflection_and_transport_failure(self):
        secret='TEST_SECRET_DO_NOT_PRINT'
        with patch.dict('os.environ',{'ANTHROPIC_API_KEY':secret}):
            with self.assertRaises(p.DemoError): p.safe_write(self.base/'secret.json',{'value':secret})
            with self.assertRaises(p.DemoError): p.approval({'value':secret},False,None)
        opener=SimpleNamespace(open=lambda *a,**k: (_ for _ in ()).throw(RuntimeError(secret)))
        with patch('urllib.request.build_opener',return_value=opener):
            transport=p.MessagesTransport(secret,2)
            with self.assertRaises(p.DemoError) as caught: transport.create(model=p.MODEL)
            self.assertNotIn(secret,str(caught.exception))
            with self.assertRaises(p.DemoError): transport.create(model=p.MODEL)
            self.assertEqual(transport.calls,1)
    def test_wrong_workdir_and_missing_workspace(self):
        with self.assertRaises(p.DemoError):
            ask.ask(SimpleNamespace(workdir=str(self.root),question_json='none',execute=False,approve_plan=None))
        with self.assertRaises(p.DemoError): p.contained(self.root,'../escape')
    def test_converter_matches_preserved_native_map(self):
        contract,frozen=p.contracts()
        case=p.contained(p.ROOT,contract['sources']['organizer_request']['path']).parent
        expected=p.read(case/'parsed_map.json')
        captions=[caption for group in expected['coarse_regions'] for caption in group['exact_source_captions']]
        hierarchy={'medium_nodes':[{'medium_id':c['medium_id'],'start_sec':c['start_sec'],'end_sec':c['end_sec']} for c in captions]}
        self.assertEqual(p.convert_map(frozen,hierarchy,captions,p.read(case/'parsed_output.json')),expected)
    def test_frozen_organizer_contract(self):
        _,frozen=p.contracts()
        h=p.geometry(46,'test')
        captions=[{'medium_id':m['medium_id'],'qwen_caption':'test'} for m in h['medium_nodes']]
        request=p.organizer_request(frozen,h,captions)
        for key in ('system','output_config','model','temperature','max_tokens'):
            self.assertEqual(request[key],frozen['organizer_request'][key])
        with self.assertRaises(p.DemoError): p.convert_map(frozen,h,captions,{'groups':[]})

    def question(self):
        q=self.base/'question.json'
        p.safe_write(q,{'question_id':'new-q','question_text':'Which color?', 'answer_options':{x:x for x in 'ABCDE'}})
        return SimpleNamespace(workdir=str(self.root),question_json=str(q),execute=False,approve_plan=None)

    def test_missing_ffmpeg_and_corrupt_video(self):
        with patch('shutil.which',return_value=None):
            with self.assertRaises(p.DemoError): self.prep()
        self.assertFalse(self.root.exists())
        bad=self.base/'bad.mp4'
        p.safe_write(bad,{'not':'video'})
        with self.assertRaises(subprocess.CalledProcessError): p.probe(bad)

    def test_existing_unrelated_workdir_refused(self):
        self.root.mkdir()
        p.safe_write(self.root/'keep.json',{'keep':True})
        with self.assertRaises(p.DemoError): self.prep()
        self.assertEqual(p.read(self.root/'keep.json'),{'keep':True})

    def test_different_cwd(self):
        import os
        previous=Path.cwd()
        try:
            os.chdir(self.base)
            self.prep()
        finally:
            os.chdir(previous)
        self.assertEqual(p.read(self.root/'plan.json')['frame_count'],46)

    def test_caption_order_and_failure(self):
        import base64
        self.prep()
        _,frozen=p.contracts()
        plan=p.read(self.root/'plan.json')
        h=p.read(self.root/'shared_hierarchy.json')
        backend=p.caption_backend('api',frozen)
        for medium in h['medium_nodes']:
            request=backend.build_request(medium,self.root,plan['files'])
            images=request['messages'][0]['content'][:-1]
            self.assertEqual(len(images),len(medium['source_fine_ids']))
            for image,name in zip(images,medium['representative_frame_paths']):
                self.assertEqual(base64.b64decode(image['source']['data']),(self.root/name).read_bytes())
        transport=SimpleNamespace(create=lambda **kwargs: {'stop_reason':'max_tokens','content':[]})
        with self.assertRaises(p.DemoError): backend.caption(h['medium_nodes'][0],self.root,plan['files'],transport)

    def test_frame_count_mismatch(self):
        with self.assertRaises(p.DemoError): p.extract(self.video,self.base/'bad-count',100)

    def test_invalid_organizer_boundaries(self):
        _,frozen=p.contracts()
        h=p.geometry(46,'test')
        captions=[{**m,'qwen_caption':'test'} for m in h['medium_nodes']]
        for ends in ([0],[1,0],[0,0,1],[2],[True]):
            raw={'groups':[{'end_medium_index':end,'navigation_summary':'test','uncertainty_notes':[]} for end in ends]}
            with self.assertRaises(p.DemoError): p.convert_map(frozen,h,captions,raw)
        captions[0]['medium_id']='wrong'
        with self.assertRaises(p.DemoError): p.convert_map(frozen,h,captions,{'groups':[]})

    def test_ask_ready_hash_and_mcq_checks(self):
        self.prep()
        args=self.question()
        with self.assertRaises(p.DemoError): ask.ask(args)
        plan=p.read(self.root/'plan.json')
        with patch.object(p,'credential',return_value='TEST_ONLY'),patch.object(p,'MessagesTransport',MockTransport):
            self.prep(True,p.object_sha(plan))
        with patch.object(p,'verify',side_effect=p.DemoError('SHA mismatch')):
            with self.assertRaises(p.DemoError): ask.ask(args)
        for value in ({}, {'question_id':'','question_text':'q','answer_options':{x:x for x in 'ABCDE'}},
                      {'question_id':'q','question_text':'q','answer_options':{'A':'only'}},
                      {**p.read(args.question_json),'correct_answer':'B'}):
            with self.assertRaises(p.DemoError): ask.normalize_question(value)

    def test_ask_approval_key_and_isolation(self):
        self.ready()
        args=self.question()
        plans=[]
        with patch.object(p,'approval',side_effect=lambda plan,*a: plans.append(plan) or False): ask.ask(args)
        args.execute=True
        args.approve_plan='wrong'
        with self.assertRaises(p.DemoError): ask.ask(args)
        args.approve_plan=p.object_sha(plans[0])
        with patch.dict('os.environ',{},clear=True):
            with self.assertRaises(p.DemoError): ask.ask(args)
        self.assertEqual(len(MockTransport.requests),3)
        self.assertFalse(list(self.root.rglob('qa_runs')))
        self.assertTrue(list((self.base/'workspace-outputs/qa_runs').glob('*/ask_plan.json')))

    def test_frame_resolver_rejects_changed_frame(self):
        self.ready()
        # Import is made through the verified entrypoint first.
        args=self.question()
        ask.ask(args)
        from direct_api_v1.frame_resolver import FrozenFrameResolver
        h=p.read(self.root/'shared_hierarchy.json')
        for fine in h['fine_nodes']: fine['source_frame_path']=str(self.root/fine['source_frame_path'])
        frame_manifest=p.read(self.root/'frame_sha256.json')
        frame_manifest['frames'][0]['sha256']='0'*64
        resolver=FrozenFrameResolver.from_hierarchy(h,frame_manifest)
        with self.assertRaises(ValueError): resolver.resolve(0)

    def test_actual_workspace_tampering_rejected(self):
        self.ready()
        args=self.question()
        with (self.root/'r3_2_navigation_map.json').open('ab') as stream:
            stream.write(b'\n')  # Test-owned temporary workspace only.
        with self.assertRaises(p.DemoError): ask.ask(args)

    def test_prepare_failure_never_becomes_ready(self):
        self.prep()
        plan=p.read(self.root/'plan.json')
        failing=SimpleNamespace(create=lambda **kwargs: {'stop_reason':'max_tokens','content':[]})
        with patch.object(p,'credential',return_value='TEST_ONLY'),patch.object(p,'MessagesTransport',return_value=failing):
            with self.assertRaises(p.DemoError): self.prep(True,p.object_sha(plan))
        self.assertFalse((self.root/'preparation_manifest.json').exists())
        with self.assertRaises(p.DemoError): self.prep()

    def test_prepare_once_ask_many_fresh_sessions(self):
        self.ready()
        initial_hashes={str(f.relative_to(self.root)):p.digest(f) for f in self.root.rglob('*') if f.is_file()}
        for qid in ('fresh-1','fresh-2'):
            question=self.base/(qid+'.json')
            p.safe_write(question,{'question_id':qid,'question_text':'New question '+qid,
                                  'answer_options':{x:x for x in 'ABCDE'}})
            args=SimpleNamespace(workdir=str(self.root),question_json=str(question),execute=False,approve_plan=None)
            plans=[]
            with patch.object(p,'approval',side_effect=lambda plan,*a: plans.append(plan) or False): ask.ask(args)
            self.assertEqual(plans[0]['question_id'],qid)
            args.execute=True
            args.approve_plan=p.object_sha(plans[0])
            before=len(MockTransport.requests)
            with patch.object(p,'credential',return_value='TEST_ONLY'),patch.object(p,'MessagesTransport',MockTransport): ask.ask(args)
            requests=MockTransport.requests[before:]
            self.assertEqual(len(requests),2)
            self.assertEqual(len(requests[0]['messages']),1)
            self.assertIn(qid,requests[0]['messages'][0]['content'][0]['text'])
            self.assertNotIn('tool_result',json.dumps(requests[0]['messages']))
            self.assertIn('tool_result',json.dumps(requests[1]['messages']))
        final_hashes={str(f.relative_to(self.root)):p.digest(f) for f in self.root.rglob('*') if f.is_file()}
        self.assertEqual(initial_hashes,final_hashes)
        self.assertEqual(len(list((self.base/'workspace-outputs/qa_runs').glob('*/result.json'))),2)


if __name__=='__main__': unittest.main()
