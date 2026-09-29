"""Offline AV extension tests; no Whisper loading or external transport."""
import copy
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock
import audio
import portable as p
import prepare
import ask
from test_portable import PortableTests, MockTransport


def segment(start=40,end=50):
    return {'audio_id':'A0001','start_sec':start,'end_sec':end,'exact_transcript':'Test speech.',
            'source_type':'audio_asr','language':'en'}


class AudioUnits(unittest.TestCase):
    def test_transcriber_settings_mocked_only(self):
        for cuda in (False,True):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model = Mock()
                model.transcribe.return_value = {'language':'en','segments':[{'start':1,'end':2,'text':' Test speech. '}]}
                whisper = SimpleNamespace(load_model=Mock(return_value=model),__version__='mock')
                modules = {'whisper':whisper,'torch':SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:cuda)),
                           'soundfile':SimpleNamespace(read=lambda *a,**k:(SimpleNamespace(ndim=1),16000))}
                def extract(*a,**k): (root/'audio_16khz_mono.wav').write_bytes(b'mocked-waveform')
                with patch.dict(sys.modules,modules), patch.object(audio,'require_stream'), patch.object(audio.subprocess,'run',side_effect=extract):
                    rows = audio.run('test.mp4',root,'test-video')
                self.assertEqual(rows,[segment(1,2)])
                whisper.load_model.assert_called_once_with('small',device='cuda' if cuda else 'cpu')
                self.assertEqual(model.transcribe.call_args.kwargs,dict(language='en',fp16=cuda,word_timestamps=False,condition_on_previous_text=False,verbose=False))
    def test_command_and_missing_stream(self):
        self.assertEqual(audio.extraction_command('in.mp4','out.wav'),
                         ['ffmpeg','-y','-v','error','-i','in.mp4','-vn','-ac','1','-ar','16000','out.wav'])
        with patch.object(audio.subprocess,'run',return_value=SimpleNamespace(stdout='{"streams":[]}')):
            with self.assertRaises(p.DemoError): audio.require_stream('video')
    def test_schema_and_overlap(self):
        media = p.geometry(90,'test')['medium_nodes']
        for start,end,counts in [(1,2,[1,0]),(40,50,[1,1]),(0,45,[1,0]),(45,90,[0,1])]:
            self.assertEqual([len(v) for v in audio.attach(media,[segment(start,end)]).values()],counts)
        for change in [{'end_sec':0},{'start_sec':float('nan')},{'language':'fr'},{'source_type':'invented'}]:
            with self.assertRaises(p.DemoError): audio.validate([{**segment(),**change}])
        with self.assertRaises(p.DemoError): audio.validate([segment(),segment()])
    def test_request_map_and_profile(self):
        _,frozen = p.contracts()
        h = p.geometry(90,'test')
        captions = [{**m,'qwen_caption':'Visual test.'} for m in h['medium_nodes']]
        visual = p.organizer_request(frozen,h,captions)
        self.assertEqual(visual['system'],frozen['organizer_request']['system'])
        self.assertTrue(all(not r['overlapping_asr'] for r in json.loads(visual['messages'][0]['content'])['timeline']))
        with self.assertRaises(p.DemoError): p.organizer_request(frozen,h,captions,[segment()])
        av = p.organizer_request(frozen,h,captions,[segment()],'whisper')
        self.assertEqual(av['system'],p.contained(p.ROOT,audio.SOURCES['prompt'][0]).read_text())
        self.assertTrue(all(r['overlapping_asr'] == [segment()] for r in json.loads(av['messages'][0]['content'])['timeline']))
        raw = {'groups':[{'end_medium_index':1,'navigation_summary':'Test','uncertainty_notes':[]}]}
        native = p.convert_map(frozen,h,captions,raw,[segment()],'whisper')
        self.assertEqual(native['coarse_regions'][0]['exact_source_asr'],[segment()])
        self.assertEqual(native['coarse_regions'][0]['exact_source_captions'],captions)
        self.assertEqual(audio.profile('none')['audio_profile'],'THESIS_FINAL_VISUAL_ONLY')
    def test_cli(self):
        for flag in ([],['--audio-mode','whisper']):
            with patch.object(sys,'argv',['prepare.py','--video','v','--workdir','w']+flag), patch.object(prepare,'prepare') as run:
                prepare.main()
                self.assertEqual(run.call_args.args[0].audio_mode,'whisper' if flag else 'none')


class AudioIntegration(PortableTests):
    # Inherit the visual-only regression suite as well as its isolated fixture.
    def av_ready(self):
        args = SimpleNamespace(video=str(self.video),workdir=str(self.root),caption_backend='api',
                               audio_mode='whisper',execute=False,approve_plan=None)
        with patch.object(audio,'require_stream'), patch.object(audio,'run') as runner:
            prepare.prepare(args)
            runner.assert_not_called()
        plan = p.read(self.root/'plan.json')
        def mocked_asr(video,root,uid):
            # Test-only bytes, never benchmark evidence or live input.
            (root/'audio_16khz_mono.wav').write_bytes(b'test-only-mocked-waveform')
            p.safe_write(root/'audio_asr.json',{'video_uid':uid,'segments':[segment(1,2)]})
            p.safe_write(root/'audio_cost.json',{'api_calls':0})
            return [segment(1,2)]
        args.execute = True
        args.approve_plan = p.object_sha(plan)
        with patch.object(audio,'run',side_effect=mocked_asr), patch.object(p,'credential',return_value='TEST_ONLY'), patch.object(p,'MessagesTransport',MockTransport):
            prepare.prepare(args)
        return p.read(self.root/'preparation_manifest.json')
    def test_av_ready_and_direct(self):
        manifest = self.av_ready()
        self.assertEqual(manifest['audio_profile'],'PORTABLE_AV_EXTENSION')
        self.assertEqual(manifest['segment_count'],1)
        self.assertEqual(manifest['audio_sha256'],p.digest(self.root/'audio_16khz_mono.wav'))
        q = self.base/'question.json'
        p.safe_write(q,{'question_id':'new-av-question','question_text':'Test?', 'answer_options':{k:k for k in 'ABCDE'}})
        args = SimpleNamespace(workdir=str(self.root),question_json=str(q),execute=False,approve_plan=None)
        plans = []
        with patch.object(p,'approval',side_effect=lambda plan,*a: plans.append(plan) or False):
            ask.ask(args)
        args.execute = True
        args.approve_plan = p.object_sha(plans[0])
        with patch.object(p,'credential',return_value='TEST_ONLY'), patch.object(p,'MessagesTransport',MockTransport):
            ask.ask(args)
        result = p.read(next((self.base/'workspace-outputs/qa_runs').glob('*/result.json')))
        self.assertEqual(result['audio_profile'],'PORTABLE_AV_EXTENSION')
        self.assertEqual(result['terminal_status'],'final_answer')
        self.assertEqual(result['transport_calls'],2)
    def test_audio_tamper_rejected(self):
        self.av_ready()
        q = self.base/'question.json'
        p.safe_write(q,{'question_id':'new','question_text':'Test?', 'answer_options':{k:k for k in 'ABCDE'}})
        args = SimpleNamespace(workdir=str(self.root),question_json=str(q),execute=False,approve_plan=None)
        for name in ('audio_16khz_mono.wav','audio_asr.json'):
            path = self.root/name
            original = path.read_bytes()
            path.write_bytes(original+b' ')
            with self.assertRaises(p.DemoError): ask.ask(args)
            path.write_bytes(original)
    def test_none_no_audio_files(self):
        self.ready()
        self.assertEqual(p.read(self.root/'preparation_manifest.json')['audio_mode'],'none')
        self.assertFalse(list(self.root.glob('audio_*')))


if __name__ == '__main__':
    unittest.main()
