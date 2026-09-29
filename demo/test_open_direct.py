"""Network-blocked tests of the explicit open protocol and shared inspection loop."""
import copy
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch
import portable as p
import ask
import test_portable as baseline


class OpenTransport(baseline.MockTransport):
    inspect_first = True
    malformed = False
    empty = False
    def create(self,**request):
        if 'tools' not in request:
            return super().create(**request)
        self.calls += 1
        self.requests.append(copy.deepcopy(request))
        inspect = self.inspect_first and self.calls == 1
        name = 'inspect_frames' if inspect else 'final_answer'
        value = {'timestamps_sec':[0,15],'reason':'Check the visible setup.'} if inspect else {
            'answer':'' if self.empty else 'A blue test frame is visible.', 'reason':'The supplied evidence supports this description.'}
        content = [{'type':'tool_use','id':f't{self.calls}','name':name,'input':value}]
        if self.malformed: content = [{'type':'text','text':'Not a structured answer.'}]
        return {'id':'test','model':p.MODEL,'stop_reason':'tool_use','content':content,
                'usage':{'input_tokens':100,'output_tokens':30,'cache_creation_input_tokens':0,'cache_read_input_tokens':0}}


class OpenTests(baseline.PortableTests):
    def args(self,**changes):
        return SimpleNamespace(**dict({'workdir':str(self.root),'question_json':None,'answer_mode':'open',
            'question':'Describe the visible setup.','question_id':'open-test-new','execute':False,'approve_plan':None},**changes))
    def plan(self,args):
        plans=[]
        with patch.object(p,'approval',side_effect=lambda plan,*a:plans.append(plan) or False):ask.ask(args)
        return plans[0]
    def execute_open(self,transport=OpenTransport):
        self.ready()
        args=self.args()
        plan=self.plan(args)
        args.execute=True;args.approve_plan=p.object_sha(plan)
        OpenTransport.requests=[]
        with patch.object(p,'credential',return_value='TEST_ONLY'),patch.object(p,'MessagesTransport',transport):ask.ask(args)
        result=p.read(next((self.base/'workspace-outputs/qa_runs').glob('*/result.json')))
        return plan,result
    def test_inspect_then_open_answer(self):
        plan,result=self.execute_open()
        self.assertEqual(result['final_answer'],'A blue test frame is visible.')
        self.assertNotIn('final_prediction',result)
        self.assertEqual(result['transport_calls'],2)
        self.assertEqual(result['inspected_timestamps_sec'],[0,15])
        self.assertFalse(plan['thesis_preserved_protocol'])
        self.assertEqual(plan['protocol_profile'],'PORTABLE_OPEN_ENDED_EXTENSION')
        request=OpenTransport.requests[0]
        text=json.dumps(request)
        self.assertNotIn('answer_options',text)
        self.assertNotIn('selected_option_id',text)
        self.assertNotIn('Options:',text)
        self.assertNotIn('multiple-choice',text)
        self.assertNotIn('option',request['system'][0]['text'].lower())
        self.assertEqual(len(request['messages']),1)
        self.assertIn('0 inspected',request['messages'][0]['content'][0]['text'])
        self.assertEqual(request['tools'][-1]['input_schema']['required'],['answer','reason'])
        self.assertEqual(request['tools'][-1]['input_schema']['properties']['answer']['minLength'],1)
        self.assertEqual(OpenTransport.requests[1]['messages'][-1]['content'][0]['type'],'tool_result')
    def test_open_no_inspection(self):
        with patch.object(OpenTransport,'inspect_first',False):
            _,result=self.execute_open()
        self.assertEqual(result['terminal_status'],'final_answer')
        self.assertEqual(result['transport_calls'],1)
        self.assertEqual(result['inspected_frames'],[])
    def test_empty_and_malformed_rejected(self):
        with patch.object(OpenTransport,'empty',True),patch.object(OpenTransport,'inspect_first',False):
            _,result=self.execute_open()
        self.assertIsNone(result['final_answer'])
        self.assertNotEqual(result['terminal_status'],'final_answer')
    def test_malformed_response_rejected(self):
        with patch.object(OpenTransport,'malformed',True):
            _,result=self.execute_open()
        self.assertIsNone(result['final_answer'])
        self.assertNotEqual(result['terminal_status'],'final_answer')
    def test_explicit_mode_and_inputs(self):
        with self.assertRaises(p.DemoError):ask.ask(self.args(answer_mode='mcq'))
        with self.assertRaises(p.DemoError):ask.ask(self.args(question=' '))
        with self.assertRaises(p.DemoError):ask.normalize_question({'question_id':'q','question_text':'q'})
        with self.assertRaises(p.DemoError):ask.ask(self.args(question_json='q.json'))
    def test_approval_and_tamper(self):
        self.ready()
        args=self.args(execute=True,approve_plan='wrong')
        with patch.object(p,'MessagesTransport') as transport:
            with self.assertRaises(p.DemoError):ask.ask(args)
            transport.assert_not_called()
        (self.root/'audio_fake').write_text('unexpected')
        with self.assertRaises(p.DemoError):ask.ask(self.args())
    def test_generated_ids_and_prompt_identity(self):
        self.ready()
        a=self.plan(self.args(question_id=None));b=self.plan(self.args(question_id=None))
        self.assertNotEqual(a['question_id'],b['question_id'])
        self.assertNotEqual(a['original_prompt_sha256'],a['derived_prompt_sha256'])
        self.assertIn('extension_source_sha256',a)
    def test_mcq_final_cannot_enter_open_protocol(self):
        self.ready()
        self.plan(self.args())
        import open_direct as opened
        response=ask.namespace({'content':[{'type':'tool_use','id':'t','name':'final_answer',
            'input':{'selected_option_id':'A','reason':'test'}}]})
        action,_,_=opened.OpenAgent._tool_action(response)
        self.assertEqual(action,{'_invalid_category':'invalid_open_final'})
        self.assertEqual([t['name'] for t in opened.tools(0)],['final_answer'])
        self.assertNotIn('selected_option_id',json.dumps(opened.tools(0)))
        from direct_api_v1.anthropic_provider import direct_action_tools
        self.assertIn('selected_option_id',json.dumps(direct_action_tools()))
    def test_cli_explicit_open(self):
        with patch.object(sys,'argv',['ask.py','--workdir','x','--answer-mode','open','--question','Test?']),patch.object(ask,'ask') as call:
            ask.main()
            self.assertEqual(call.call_args.args[0].answer_mode,'open')
            self.assertIsNone(call.call_args.args[0].question_json)
    def test_reader_compatibility_is_exact(self):
        from prepare import implementation_hashes
        saved=implementation_hashes()
        saved['ask.py']='fed80c6194f35d54c99c03bf9c189504b9979f1a105be3a49f0e8fc71ea4bf48'
        ask.validate_reader_implementation(saved)
        saved['ask.py']='0'*64
        with self.assertRaises(p.DemoError):ask.validate_reader_implementation(saved)
        saved=implementation_hashes();saved['audio.py']='0'*64
        with self.assertRaises(p.DemoError):ask.validate_reader_implementation(saved)
