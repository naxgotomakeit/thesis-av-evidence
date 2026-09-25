"""Offline summarizer wiring contract; provider and encoder are deterministic stubs."""
from __future__ import annotations
import json, os, pathlib, sys, tempfile, types
import numpy as np

HERE=pathlib.Path(__file__).resolve().parent; WORK=HERE.parent; ROOT=WORK.parent
INDEX=pathlib.Path(os.getenv('INDEX_ROOT') or ROOT/'work_index').expanduser().resolve()
sys.path[:0]=[str(WORK/'runnable_runtime'),str(WORK/'runtime_support'),str(WORK/'runtime_support/src')]
sys.modules.setdefault('cv2', types.ModuleType('cv2'))
import videoseal
prompt_pkg=types.ModuleType('videoseal.prompts'); prompt_pkg.__path__=[str(WORK/'runnable_runtime/videoseal/prompts')]
sys.modules['videoseal.prompts']=prompt_pkg
video_pkg=types.ModuleType('videoseal.utils.video'); video_pkg.__path__=[str(WORK/'runnable_runtime/videoseal/utils/video')]
sys.modules['videoseal.utils.video']=video_pkg
from videoseal.tools import retrieval_adapter as adapter
from videoseal.tools import visual_tools
from experiments.hourvideo_v7_4_variant_c_budgets_v1 import retriever

class Encoder:
    def encode(self, texts):
        row=np.linspace(-1,1,768,dtype=np.float32); row/=np.linalg.norm(row)
        return np.stack([row for _ in texts])
class Client:
    fail=False; calls=0
    def generate_text(self,*args,**kwargs):
        type(self).calls+=1
        if type(self).fail: raise RuntimeError('stub summarizer failure')
        return 'stub summary'
    def get_last_usage(self): return {'prompt_tokens':3,'completion_tokens':2,'total_tokens':5}

def run():
    os.environ.update(VISUAL_RETRIEVE_SUMMARY_ENABLED='1',RETRIEVE_SUMMARY_ENABLED='1',VISUAL_RETRIEVE_RETURN_SPANS='0',VISUAL_RETRIEVE_TOPK='30',RETRIEVE_MIN_TIME_GAP_SEC='15',PAIRED_HIERARCHICAL_ROOT=str(INDEX))
    retriever._encoder=lambda *args,**kwargs: Encoder()
    visual_tools.build_mllm_client_from_env_prefix=lambda prefix: Client()
    original=visual_tools.summarize_visual_retrieval_candidates; observed=[]
    def spy(**kwargs): observed.append([dict(x) for x in kwargs['out_items']]); return original(**kwargs)
    visual_tools.summarize_visual_retrieval_candidates=spy
    case=sorted((INDEX/'cases').iterdir())[0]; vid=case.name
    rows={}; internal={}
    for profile,limit in [('h6',6),('h15',15),('h30',30)]:
        os.environ.update(RETRIEVAL_BACKEND='hierarchical',HIERARCHICAL_PROFILE=profile)
        tool=adapter.build_retrieve_tool()()
        before=len(observed); result=tool.forward(query='person performing an action',video_id=vid,original_question='offline synthetic prompt')
        assert not result.error and set(result.output)=={'summary'} and isinstance(result.output['summary'],str)
        assert len(observed)==before+1 and len(observed[-1])<=limit
        assert all(set(x)=={'start_time','end_time','caption'} for x in observed[-1])
        assert 'raw_candidates' not in result.output and 'retrieval_backend' not in result.output
        assert result.metadata['raw_candidate_count']==len(observed[-1])
        paths=result.metadata['coarse_medium_fine_paths']; assert paths
        assert all('source_frame_path' in p['fine'] and 'frame_index' in p['fine'] for p in paths)
        rows[profile]={'raw_count':len(observed[-1]),'summary_calls':1,'planner_visible_keys':sorted(result.output)}
        internal[profile]=result.metadata
    with tempfile.TemporaryDirectory(prefix='v74-flat-contract-') as td:
        root=pathlib.Path(td); vid2='synthetic-flat'; idx=root/vid2; idx.mkdir()
        caps={f'{i*30}_{i*30+10}':{'caption':f'caption {i}'} for i in range(35)}
        (idx/'semantic_captions.json').write_text(json.dumps(caps))
        visual_tools.query_embed=lambda path,query,topk:[(k,float(35-i)) for i,k in enumerate(caps)][:topk]
        os.environ.update(RETRIEVAL_BACKEND='videoseal_flat',PAIRED_FLAT_INDEX_ROOT=str(root),SEMANTIC_RETRIEVE_MIX='embed')
        tool=adapter.build_retrieve_tool()(); before=len(observed); flat=tool.forward(query='person action',video_id=vid2,original_question='offline synthetic prompt')
        assert not flat.error and set(flat.output)=={'summary'} and len(observed)==before+1 and len(observed[-1])==30
        shared_impl=visual_tools.summarize_visual_retrieval_candidates is spy
        Client.fail=True
        herr=adapter.build_retrieve_tool()  # flat type retained here
        flat_error=herr() .forward(query='person action',video_id=vid2,original_question='offline synthetic prompt')
        os.environ.update(RETRIEVAL_BACKEND='hierarchical',HIERARCHICAL_PROFILE='h6')
        hierarchy_error=adapter.build_retrieve_tool()().forward(query='person action',video_id=vid,original_question='offline synthetic prompt')
        assert flat_error.error and hierarchy_error.error
        assert flat_error.metadata['error_type']==hierarchy_error.metadata['error_type']=='other'
    schema=adapter.build_retrieve_tool()().json
    params=schema['function']['parameters']; assert schema['function']['name']=='visual_retrieve' and params['required']==['query'] and set(params['properties'])=={'query'}
    report={'status':'PASS','tool_input_schema':'visual_retrieve(query: string)','internal_raw_candidate_schema':['start_time','end_time','caption'],'planner_visible_response_schema':{'summary':'string'},'profiles':rows,'shared_reference_summarizer':shared_impl,'summarizer_failure_parity':True,'source_frame_path_internal_only':True,'api_called':False,'model_loaded':False}
    out=ROOT/'outputs/summarizer_contract_v74.json'; out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); print(json.dumps(report,sort_keys=True))
if __name__=='__main__': run()
