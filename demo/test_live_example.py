"""Wrapper tests: no network, inference or paid calls."""
import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import run_live_example as runner
import portable as p


class ExampleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch('socket.socket.connect',side_effect=AssertionError('Network forbidden')))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.real_root = p.ROOT
        self.stack.enter_context(patch.object(p,'ROOT',self.root))
        for kind in ('audio','visual'):
            q=p.read(self.real_root/f'demo/example_live/question_{kind}.json')
            self.assertEqual(set(q),{'question_id','question_text','answer_options'})
            self.assertNotIn('packet',str(q).lower())
            p.safe_write(self.root/f'demo/example_live/question_{kind}.json',q)
        self.workspace=self.root/'demo_runs/public_sous_vide/workspace'
    def args(self,**kwargs):
        return SimpleNamespace(**dict({'question':'audio','stage':'auto','execute':False,'approve_plan':None},**kwargs))
    def invoke(self,args):
        with patch.object(runner,'download') as download, patch.object(runner.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as call:
            runner.run(args)
            return call.call_args,download.called
    def test_default_prepare_no_execute(self):
        call,download=self.invoke(self.args())
        self.assertTrue(download)
        argv=call.args[0]
        self.assertTrue(argv[2].endswith('prepare.py'))
        self.assertNotIn('--execute',argv)
        self.assertEqual(argv[-4:],['--caption-backend','api','--audio-mode','whisper'])
        self.assertEqual(call.kwargs['cwd'],self.root)
    def test_ready_ask_questions_and_explicit_approval(self):
        p.safe_write(self.workspace/'preparation_manifest.json',{'status':'READY'})
        for kind in ('audio','visual'):
            call,download=self.invoke(self.args(question=kind))
            self.assertFalse(download)
            self.assertTrue(call.args[0][-1].endswith(f'question_{kind}.json'))
            self.assertNotIn('--execute',call.args[0])
        call,_=self.invoke(self.args(stage='ask',execute=True,approve_plan='test-sha'))
        self.assertEqual(call.args[0][-3:],['--execute','--approve-plan','test-sha'])
    def test_approval_errors_before_any_side_effect(self):
        for args in (self.args(execute=True),self.args(approve_plan='sha'),self.args(execute=True,approve_plan='sha'),self.args(stage='ask')):
            with patch.object(runner,'download') as d,patch.object(runner.subprocess,'run') as s:
                with self.assertRaises(p.DemoError):runner.run(args)
                d.assert_not_called();s.assert_not_called()
    def test_prepare_approval_and_no_chaining(self):
        with patch.object(runner,'download'),patch.object(runner.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as call:
            runner.run(self.args(stage='prepare',execute=True,approve_plan='prepare-sha'))
            call.assert_called_once()
            self.assertEqual(call.call_args.args[0][-3:],['--execute','--approve-plan','prepare-sha'])
        p.safe_write(self.workspace/'preparation_manifest.json',{'status':'READY'})
        with self.assertRaises(p.DemoError):self.invoke(self.args(stage='prepare'))
    def test_core_failure_no_retry(self):
        with patch.object(runner,'download'),patch.object(runner.subprocess,'run',return_value=SimpleNamespace(returncode=2)) as call:
            with self.assertRaises(p.DemoError):runner.run(self.args())
            call.assert_called_once()
    def test_symlink_rejected(self):
        (self.root/'demo_runs').symlink_to(self.root,target_is_directory=True)
        with self.assertRaises(p.DemoError): self.invoke(self.args())
    def test_download_hash_and_cached_identity(self):
        body=b'public-example-test-only'
        path=self.root/'media.webm'
        with patch.object(runner,'MEDIA_BYTES',len(body)),patch.object(runner,'MEDIA_SHA',hashlib.sha256(body).hexdigest()),patch.object(runner.urllib.request,'urlopen',return_value=io.BytesIO(body)) as get:
            runner.download(path)
            self.assertEqual(path.read_bytes(),body)
            runner.download(path)
            get.assert_called_once()
            path.write_bytes(b'bad')
            with self.assertRaises(p.DemoError):runner.download(path)
    def test_bad_download_not_published(self):
        path=self.root/'media.webm'
        with patch.object(runner.urllib.request,'urlopen',return_value=io.BytesIO(b'bad')):
            with self.assertRaises(p.DemoError):runner.download(path)
        self.assertFalse(path.exists())
        self.assertFalse(list(self.root.glob('*.download')))


if __name__ == '__main__':
    unittest.main()
