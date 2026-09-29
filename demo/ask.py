"""Fresh preserved Direct protocol on a verified PORTABLE_API workspace."""
import sys
sys.dont_write_bytecode = True
import copy
from pathlib import Path
from types import SimpleNamespace
import time
import tempfile
import portable as p
from prepare import implementation_hashes


def namespace(value):
    if isinstance(value,dict):
        return SimpleNamespace(**{k:(v if k == 'input' else namespace(v)) for k,v in value.items()})
    if isinstance(value,list):
        return [namespace(v) for v in value]
    return value


class Adapter:
    def __init__(self,transport=None):
        self.transport = transport
    def create(self,**request):
        p.require(self.transport is not None,'Dry run cannot send requests.')
        return namespace(self.transport.create(**request))


def normalize_question(value):
    """Public A-E object to the preserved loader's ordered option-row schema."""
    p.require(isinstance(value,dict) and set(value) == {'question_id','question_text','answer_options'},
              'MCQ requires only question_id, question_text and answer_options; no gold or session state.')
    for field in ('question_id','question_text'):
        p.require(isinstance(value[field],str) and bool(value[field].strip()), 'Missing MCQ identity/text.')
    options = value['answer_options']
    p.require(isinstance(options,dict) and set(options) == set('ABCDE'), 'Exactly A-E options are required.')
    p.require(all(isinstance(text,str) and bool(text.strip()) for text in options.values()), 'Empty/non-text option.')
    return {**value,'answer_options':[{'option_id':letter,'text':options[letter]} for letter in 'ABCDE']}


def ask(args):
    contract,frozen = p.contracts()
    root = p.workdir_path(args.workdir)
    p.require((root/'preparation_manifest.json').is_file(), 'READY preparation_manifest.json is required.')
    manifest = p.read(root/'preparation_manifest.json')
    p.require(manifest['schema_version'] == 2, 'Unsupported workspace schema; no automatic migration.')
    p.require(manifest['status'] == 'READY', 'Preparation has not completed.')
    for field,value in p.FIDELITY.items():
        p.require(manifest[field] == value,'Preparation profile mismatch.')
    for name,sha in manifest['files'].items():
        p.verify(p.contained(root,name),sha)
    required = {'plan.json','shared_hierarchy.json','frame_sha256.json','medium_captions.json',
                'organizer_request.json','organizer_response.json','organizer_output.json','r3_2_navigation_map.json'}
    p.require(required <= manifest['files'].keys(), 'Incomplete prepared workspace manifest.')
    actual = {str(f.relative_to(root)) for f in root.rglob('*') if f.is_file()}
    p.require(actual == set(manifest['files']) | {'preparation_manifest.json'}, 'Prepared workspace inventory changed.')
    preparation = p.read(root/'plan.json')
    import audio
    mode = preparation.get('audio_mode','none')
    for field,value in audio.profile(mode).items():
        p.require(manifest.get(field) == value, 'Audio profile mismatch.')
    segments = []
    if mode == 'whisper':
        p.require({'audio_16khz_mono.wav','audio_asr.json','audio_cost.json'} <= manifest['files'].keys(), 'Missing audio integrity records.')
        asr = p.read(root/'audio_asr.json')
        p.require(asr['video_uid'] == manifest['video_uid'], 'ASR video identity mismatch.')
        segments = asr['segments']
        audio.validate(segments)
        p.require(manifest['audio_sha256'] == manifest['files']['audio_16khz_mono.wav'] and manifest['segment_count'] == len(segments), 'Audio identity mismatch.')
    p.require(p.object_sha(preparation) == manifest['plan_sha256'])
    p.require(preparation['implementation'] == implementation_hashes(),'Implementation changed since preparation.')
    sys.path.insert(0,str(p.contained(p.ROOT,contract['source_root'])))
    from direct_api_v1.maps import load_direct_input
    from direct_api_v1.frame_resolver import FrozenFrameResolver
    from direct_api_v1.state import DirectSessionState
    from direct_api_v1.controller import DirectController
    from direct_api_v1.anthropic_provider import AnthropicDirectAgent, direct_action_tools
    question_path = Path(args.question_json).resolve()
    public_question = p.read(question_path)
    normalized = normalize_question(public_question)
    # Preserve the original loader unchanged. This temporary representation is not evidence.
    with tempfile.TemporaryDirectory(prefix='portable-direct-question-') as temporary:
        normalized_path = Path(temporary)/'question.json'
        p.safe_write(normalized_path,normalized)
        direct_input = load_direct_input(question_path=normalized_path,map_path=root/'r3_2_navigation_map.json',method='R3',
                                         expected_map_sha256=manifest['files']['r3_2_navigation_map.json'])
    question = direct_input.question
    p.require(question.get('video_uid',manifest['video_uid']) == manifest['video_uid'],'Wrong video identity.')
    p.require(question['question_id'].strip() and question['question_text'].strip())
    p.require(all(isinstance(o.get('text'),str) and o['text'].strip() for o in question['answer_options']))
    hierarchy = copy.deepcopy(p.read(root/'shared_hierarchy.json'))
    p.require(hierarchy['video_uid'] == manifest['video_uid'])
    p.require(hierarchy == p.geometry(preparation['frame_count'],manifest['video_uid']), 'Hierarchy geometry mismatch.')
    native = p.convert_map(frozen,hierarchy,p.read(root/'medium_captions.json'),p.read(root/'organizer_output.json'),segments,mode)
    p.require(native == direct_input.map_document, 'Native map/source-caption binding mismatch.')
    for fine in hierarchy['fine_nodes']:
        fine['source_frame_path'] = str(p.contained(root,fine['source_frame_path']))
    resolver = FrozenFrameResolver.from_hierarchy(hierarchy,p.read(root/'frame_sha256.json'))
    p.require(len(list((root/'frames_1fps').glob('*.jpg'))) == preparation['frame_count'],'Frame pool changed.')
    state = DirectSessionState(direct_input)
    cfg = frozen['direct_config']
    adapter = Adapter()
    agent = AnthropicDirectAgent(credential_env_path=Path('UNUSED'),model=cfg['model'],
        pricing=cfg['anthropic_cache_aware_pricing_usd_per_million_tokens'],
        max_output_tokens=cfg['max_output_tokens'],timeout_sec=cfg['timeout_sec'],
        max_retries=cfg['max_retries'],max_total_usd=cfg['hard_api_budget_usd'],
        client=SimpleNamespace(messages=adapter))
    preview = {'system':agent._system(state),'messages':[agent._initial_message(state)],'tools':direct_action_tools()}
    plan = {**p.FIDELITY,**audio.profile(mode),'mode':'FRESH_DIRECT','thesis_result':False,
            'workspace_sha256':p.digest(root/'preparation_manifest.json'),
            'question_sha256':p.digest(question_path),'question_id':question['question_id'],
            'duration_sec':preparation['duration_sec'], 'frame_count':preparation['frame_count'],
            'fine_count':preparation['fine_count'],'medium_count':preparation['medium_count'],
            'caption_api_calls':0,'organizer_api_calls':0,
            'expected_direct_calls':'Adaptive: planning estimate 2–8; may stop after one; transport ceiling 66.',
            'heuristic_usd_range':[0.01,0.20], 'provider_accounting_budget_usd':cfg['hard_api_budget_usd'],
            'estimated_input_tokens_range':[5000,120000], 'estimated_output_tokens_range':[100,4096],
            'conservative_maximum_allowance_usd':cfg['hard_api_budget_usd'],
            'cost_caveat':'Heuristic for short videos, not a billing guarantee; unknown failed requests may be billed.',
            'model':cfg['model'],'request_preview_sha256':p.object_sha(preview),
            'implementation':implementation_hashes()}
    output_root = root.parent/(root.name+'-outputs')
    p.require(not output_root.is_symlink(), 'Output directory must not be a symlink.')
    qa_root = output_root/'qa_runs'
    p.require(not qa_root.is_symlink(), 'qa_runs must not be a symlink.')
    output = qa_root/p.object_sha(plan)
    p.require(not output.is_symlink(), 'Run directory must not be a symlink.')
    if (output/'ask_plan.json').exists():
        p.require(p.read(output/'ask_plan.json') == plan, 'Stored ASK plan mismatch.')
    else:
        p.safe_write(output/'ask_plan.json',plan)
    if not p.approval(plan,args.execute,args.approve_plan):
        return
    key = p.credential()
    p.require(not (output/'EXECUTION_STARTED.json').exists(),'This exact execution already started; no automatic retry.')
    p.safe_write(output/'EXECUTION_STARTED.json',plan)
    transport = p.MessagesTransport(key,66,timeout=cfg['timeout_sec'])
    adapter.transport = transport
    started = time.monotonic()
    telemetry = DirectController(resolver=resolver,max_turns=32,enable_action_correction=True).run(state=state,agent=agent)
    result = {**p.FIDELITY,**audio.profile(mode),'new_execution_not_thesis_result':True,'question_id':state.question_id,
              'route':'R3','final_prediction':state.final_prediction,'terminal_status':state.terminal_status,
              'transport_calls':transport.calls,'latency_sec':time.monotonic()-started,
              'recorded_api_cost_usd':agent.total_cache_aware_usd,
              'usage':{field:getattr(telemetry,field) for field in
                  ('total_input_tokens','total_output_tokens','total_cache_creation_input_tokens',
                   'total_cache_read_input_tokens','unique_images_transmitted','correction_attempts')},
              'recorded_decisions':[{'action':t.action_type,'reason':t.action_reason,
                                     'requested_timestamps_sec':t.requested_timestamps_sec}
                                    for t in telemetry.turns],
              'unknown_failed_request_cost_possible':transport.failed,
              'inspected_frames':[{'frame_index':f.frame_index,'sha256':f.observed_sha256} for f in state.inspected_frames]}
    p.safe_write(output/'result.json',result)
    print(__import__('json').dumps(result,indent=2))


def main():
    parser = p.SafeParser(description=__doc__)
    parser.add_argument('--workdir',required=True)
    parser.add_argument('--question-json',required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--dry-run',action='store_true')
    group.add_argument('--execute',action='store_true')
    parser.add_argument('--approve-plan')
    ask(parser.parse_args())


if __name__ == '__main__':
    sys.exit(p.cli_entry(main))
