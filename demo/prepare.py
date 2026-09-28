"""Fresh Direct-targeted preparation; default execution stops before paid calls."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import portable as p


def implementation_hashes():
    return {name:p.digest(p.ROOT/'demo'/name) for name in
            ('portable.py','prepare.py','ask.py','portable_contracts.json')}


def prepare(args):
    contract, frozen = p.contracts()
    backend = p.caption_backend(args.caption_backend, frozen)
    root = p.workdir_path(args.workdir)
    video = Path(args.video).expanduser().resolve()
    source_sha = p.digest(video)
    if not root.exists():
        info = p.probe(video)
        root.mkdir(parents=True,mode=0o700)
        frames = p.extract(video, root/'frames_1fps', info['duration_sec'])
        uid = 'portable-'+source_sha[:20]
        hierarchy = p.geometry(len(frames), uid)
        hashes = {str(f.relative_to(root)):p.digest(f) for f in frames}
        p.safe_write(root/'shared_hierarchy.json', hierarchy)
        p.safe_write(root/'frame_sha256.json', {'video_uid':uid, 'frames':[
            {'frame_index':i, 'path':str(f.relative_to(root)), 'sha256':hashes[str(f.relative_to(root))]}
            for i,f in enumerate(frames)]})
        previews = []
        for medium in hierarchy['medium_nodes']:
            request = backend.build_request(medium, root, hashes)
            previews.append({'medium_id':medium['medium_id'], 'request_sha256':p.object_sha(request),
                             'ordered_images':medium['representative_frame_paths'],
                             'model':request['model'], 'max_tokens':request['max_tokens']})
        p.safe_write(root/'request_preview.json', {'caption_requests':previews,
            'organizer':{'status':'INPUT_PENDING_REAL_CAPTIONS', 'model':frozen['organizer_request']['model'],
                         'max_tokens':frozen['organizer_request']['max_tokens'],
                         'frozen_request_sha256':contract['sources']['organizer_request']['sha256']}})
        hashes.update({name:p.digest(root/name) for name in
                       ('shared_hierarchy.json','frame_sha256.json','request_preview.json')})
        m = len(hierarchy['medium_nodes'])
        plan = {**p.FIDELITY, 'status':'PLAN_ONLY', 'source_video_sha256':source_sha,
                'frame_extraction':p.extraction_config(),
                'organizer_model':frozen['organizer_request']['model'],
                'prompt_config_sources':{name:contract['sources'][name] for name in
                    ('caption','organizer_config','organizer_request','map_converter','geometry_reference')},
                'video_uid':uid, **info, 'frame_count':len(frames),
                'effective_cache_duration_sec':len(frames), 'fine_count':len(hierarchy['fine_nodes']),
                'medium_count':m, 'caption_api_calls':m, 'organizer_api_calls':1,
                'direct_api_calls':0, 'ask_calls':'Separate approval; adaptive, not known before inference.',
                'estimate':p.estimate(m,len(hierarchy['fine_nodes'])),
                'implementation':implementation_hashes(), 'files':hashes}
        p.safe_write(root/'plan.json', plan)
    else:
        p.require((root/'plan.json').is_file(), 'Incomplete workspace; choose a new workdir. No automatic repair.')
        plan = p.read(root/'plan.json')
        p.require(plan['source_video_sha256'] == source_sha, 'Different source video.')
        p.require(plan['implementation'] == implementation_hashes(), 'Implementation changed; choose a new workdir.')
        for name, sha in plan['files'].items():
            p.verify(p.contained(root,name),sha)
    p.require(not (root/'EXECUTION_STARTED.json').exists(), 'This workspace already started execution; no automatic retry or overwrite.')
    if not p.approval(plan,args.execute,args.approve_plan):
        return
    key = p.credential()
    p.safe_write(root/'EXECUTION_STARTED.json', {'plan_sha256':p.object_sha(plan), 'automatic_resume':False})
    transport = p.MessagesTransport(key, plan['caption_api_calls']+1)
    hierarchy = p.read(root/'shared_hierarchy.json')
    captions = []
    for medium in hierarchy['medium_nodes']:
        caption,response = backend.caption(medium,root,plan['files'],transport)
        captions.append(caption)
        p.safe_write(root/'caption_responses'/f"{medium['medium_id']}.json",response)
    p.safe_write(root/'medium_captions.json',captions)
    request = p.organizer_request(frozen,hierarchy,captions)
    p.safe_write(root/'organizer_request.json',request)
    response = transport.create(**request)
    p.safe_write(root/'organizer_response.json',response)
    raw = json.loads(p.response_text(response))
    native = p.convert_map(frozen,hierarchy,captions,raw)
    p.safe_write(root/'organizer_output.json',raw)
    p.safe_write(root/'r3_2_navigation_map.json',native)
    files = {str(f.relative_to(root)):p.digest(f) for f in root.rglob('*') if f.is_file()}
    p.safe_write(root/'preparation_manifest.json', {**p.FIDELITY, 'schema_version':2,
        'status':'READY', 'video_uid':plan['video_uid'], 'plan_sha256':p.object_sha(plan),
        'files':files, 'organizer_model':request['model'], 'new_execution_not_thesis_result':True})
    print('READY: new PORTABLE_API workspace. Not an exact thesis preprocessing reproduction.')


def main():
    parser = p.SafeParser(description=__doc__)
    parser.add_argument('--video',required=True)
    parser.add_argument('--workdir',required=True)
    parser.add_argument('--caption-backend',choices=['api','local-qwen','remote-qwen'],default='api')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--dry-run',action='store_true',help='Default: local extraction and request construction only.')
    group.add_argument('--execute',action='store_true',help='Paid execution only with matching approved plan.')
    parser.add_argument('--approve-plan')
    prepare(parser.parse_args())


if __name__ == '__main__':
    sys.exit(p.cli_entry(main))
