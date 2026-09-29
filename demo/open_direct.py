"""Explicit open-ended extension of the pinned Direct loop; never MCQ parsing."""
import copy
import hashlib
import uuid
import portable as p
from direct_api_v1.anthropic_provider import AnthropicDirectAgent, direct_action_tools
from direct_api_v1.controller import DirectController, DirectStateError
from direct_api_v1.maps import DirectInput, _load_json
from direct_api_prep.evidence import assert_safe_direct_evidence
from direct_api_v1.prompt import DIRECT_V1_SYSTEM_PROMPT

PROFILE = 'PORTABLE_OPEN_ENDED_EXTENSION'


def derived_prompt():
    text = DIRECT_V1_SYSTEM_PROMPT
    replacements = {
        'Answer one multiple-choice question': 'Answer one open-ended question',
        'the question, options A-E, an offline video map': 'the question, an offline video map',
        'eliminable options, and remaining answer-critical uncertainty': 'relevant evidence, and remaining answer-critical uncertainty',
        'reliably distinguishes the options': 'reliably answers the question',
        'select the option best supported': 'give an answer best supported',
        'current answer choice': 'current answer',
        'one option is materially better supported': 'the answer is sufficiently supported',
        'change that choice': 'change that answer',
        'select exactly one of A, B, C, D, E': 'return a non-empty free-text answer grounded only in the supplied map and inspected frames; state uncertainty when evidence is insufficient',
    }
    for old,new in replacements.items():
        p.require(text.count(old) == 1, 'Pinned prompt derivation no longer matches.')
        text = text.replace(old,new)
    return text


def identity(contract):
    return {'answer_mode':'open','protocol_profile':PROFILE,'thesis_preserved_protocol':False,
            'original_prompt_source':contract['sources']['direct_api_v1/prompt.py'],
            'original_prompt_sha256':hashlib.sha256(DIRECT_V1_SYSTEM_PROMPT.encode()).hexdigest(),
            'derived_prompt_sha256':hashlib.sha256(derived_prompt().encode()).hexdigest(),
            'extension_source_sha256':p.digest(p.ROOT/'demo/open_direct.py')}


def load_input(question_text, question_id, map_path, expected_sha):
    p.require(isinstance(question_text,str) and bool(question_text.strip()), 'Open mode requires a non-empty --question.')
    question_id = question_id or 'open-'+uuid.uuid4().hex
    p.require(isinstance(question_id,str) and bool(question_id.strip()), 'Empty question ID.')
    question = {'question_id':question_id,'question_text':question_text}
    assert_safe_direct_evidence(question)
    p.verify(map_path,expected_sha)
    native = _load_json(map_path)
    assert_safe_direct_evidence(native)
    return DirectInput(question,'R3',native,str(map_path),expected_sha)


def tools(max_new_images=3):
    result = copy.deepcopy(direct_action_tools(max_new_images))
    result[-1] = {'name':'final_answer','description':'Finish with a grounded free-text answer and brief recorded rationale.',
                  'input_schema':{'type':'object','additionalProperties':False,
                    'properties':{'answer':{'type':'string','minLength':1},
                                  'reason':{'type':'string','minLength':1,'maxLength':240}},
                    'required':['answer','reason']}}
    return result


def request_tools(request):
    """Replace only the final tool contract; retain original inspection budget schema."""
    request = copy.deepcopy(request)
    inspection = [t for t in request['tools'] if t['name'] == 'inspect_frames']
    maximum = inspection[0]['input_schema']['properties']['timestamps_sec']['maxItems'] if inspection else 0
    request['tools'] = tools(maximum)
    return request


def valid_final(value):
    return (isinstance(value,dict) and set(value) == {'action','answer','reason'}
            and value['action'] == 'final_answer'
            and isinstance(value['answer'],str) and bool(value['answer'].strip())
            and isinstance(value['reason'],str) and 0 < len(value['reason'].strip()) <= 240)


class OpenAgent(AnthropicDirectAgent):
    def _system(self,state):
        system = super()._system(state)
        system[0]['text'] = derived_prompt()
        return system

    def _initial_message(self,state):
        return {'role':'user','content':[{'type':'text','text':
            f"Question: {state.direct_input.question['question_text']}\n{self._budget_state(state)}\nReturn the next Direct action."}]}

    @staticmethod
    def _tool_action(response):
        action,metadata,tool = AnthropicDirectAgent._tool_action(response)
        if tool and tool['name'] == 'final_answer' and not metadata.get('malformed_response_category'):
            value = tool['input']
            candidate = {'action':'final_answer',**value}
            if set(value) != {'answer','reason'} or not valid_final(candidate):
                metadata['malformed_response_category'] = 'invalid_open_final'
                return {'_invalid_category':'invalid_open_final'},metadata,tool
            return candidate,metadata,tool
        return action,metadata,tool

    def next_action(self,state,correction_message=None):
        if correction_message:
            correction_message = correction_message.replace('with exactly one option A, B, C, D, or E',
                                                            'with a non-empty free-text answer and brief reason')
        return super().next_action(state,correction_message)


class OpenController(DirectController):
    def apply_action(self,state,telemetry,value,metadata=None,correction_allowed=False,correction_exhausted=False):
        if isinstance(value,dict) and value.get('action') == 'final_answer':
            if state.is_terminal:
                raise DirectStateError('action after terminal Direct session')
            if not valid_final(value):
                self._reject(state,telemetry,'final_answer',[],'invalid_open_final',metadata,
                             terminal=not correction_allowed,correction_exhausted=correction_exhausted)
                return 'invalid_open_final' if correction_allowed else None
            self._record_validation(state,'accepted',value,metadata)
            state.rounds += 1
            state.final_prediction = value['answer']
            self._append_turn(state,telemetry,self._turn_from_metadata(state.rounds,'final_answer',value['reason'],[],[],0,0,metadata))
            if metadata is not None:
                self._persist_provider_attempts(telemetry,metadata.provider_attempt_records,'accepted')
            self._close(state,telemetry,'final_answer')
            self._checkpoint(state,telemetry)
            return None
        return super().apply_action(state,telemetry,value,metadata,correction_allowed,correction_exhausted)
