"""Present preserved Direct records without rebuilding requests or inspecting media."""
from pathlib import PurePosixPath
import math
import re

from run_demo import require, excerpt


def only(rows):
    require(len(rows) == 1, 'Missing or ambiguous exact-identity record.')
    return rows[0]


def validate(manifest, data):
    qid, route, video = (manifest[k] for k in ('question_id', 'route', 'video_uid'))
    method = manifest['method']
    require(manifest['expected_identities'] == {'question_id': qid, 'route': route, 'method': method, 'video_uid': video})
    require(method == 'R3' and route == method + ':' + qid and qid.startswith(video + '_'))
    trace, status = data['trace'], data['status']
    question = data['question']['question']
    require(question['question_id'] == trace['question_id'] == qid)
    require(question['video_uid'] == video and trace['method'] == method and status['route_id'] == route)
    options = question['answer_options']
    require(len(options) == 5 and {o['option_id'] for o in options} == set('ABCDE'))
    scored = only([r for r in data['scored']['per_question'] if r['question_id'] == qid and r['method'] == method])
    reuse = only([r for r in data['reuse']['reuse_r3']['routes'] if r['question_id'] == qid and r['route_id'] == route])
    require(scored['route_id'] == reuse['route_id'] == route and reuse['video_id'] == video)
    require(scored['source_kind'] == 'frozen_reuse' and reuse['reuse_eligible'] is True)
    require(scored['terminal_status'] == reuse['terminal_status'] == trace['terminal_status'] == 'final_answer')
    require(status['state'] == 'terminal_success' and status['category'] == 'final_answer')
    require(status['prediction'] == scored['final_prediction'] == trace['final_prediction'] == scored['gold'])
    require(scored['correct'] is True and trace['correction_attempts'] == 0)
    require(reuse['original_input_sha256'] == reuse['copied_input_sha256'])
    # Bind redacted staged bytes to the ORIGINAL artifact identity in the thesis merge.
    relative = manifest['sources']['trace']['path'].removeprefix(manifest['archive_root'] + '/')
    source = only([r for r in data['source_manifest'] if r['staging_path'] == relative])
    require(source['staged_sha256'] == manifest['sources']['trace']['sha256'])
    require(source['original_sha256'] == reuse['source_artifact_sha256'] == scored['source_artifact_sha256'])
    require(source['source_path_portable'] == reuse['source_artifact'] == scored['source_artifact'] == status['artifact_path'])
    turns, attempts = trace['turns'], trace['provider_attempts']
    require(len(turns) == trace['rounds'] == len(attempts) == trace['total_api_attempts'] == 2)
    require([t['action_type'] for t in turns] == ['inspect_frames', 'final_answer'])
    for index, (turn, attempt) in enumerate(zip(turns, attempts), 1):
        require(turn['turn_index'] == attempt['turn_index'] == index and attempt['attempt_index'] == 1)
        require(turn['provider_status'] == attempt['controller_validation'] == 'accepted')
        require(attempt['response_received'] is True and not attempt['structural_correction'])
        tool = only(attempt['response_metadata']['tool_metadata'])
        require(tool['name'] == turn['action_type'] == attempt['parsed_action']['action'])
        require(tool['normalised_input']['reason'] == turn['action_reason'] == attempt['parsed_action']['reason'])
        require(bool(tool['id']))
        require(turn['requested_timestamps_sec'] == attempt['requested_timestamps_sec'])
    inspection, final = turns
    first_tool = attempts[0]['response_metadata']['tool_metadata'][0]
    require(first_tool['normalised_input']['timestamps_sec'] == inspection['requested_timestamps_sec'])
    require(attempts[1]['response_metadata']['tool_metadata'][0]['normalised_input']['selected_option_id'] == trace['final_prediction'])
    frames = inspection['resolved_frames']
    require(len(frames) == inspection['images_transmitted'] == trace['unique_images_transmitted'] == 3)
    require(final['images_transmitted'] == 0 and final['resolved_frames'] == [])
    require([f['requested_timestamp_sec'] for f in frames] == inspection['requested_timestamps_sec'])
    for frame in frames:
        require(frame['expected_sha256'] == frame['observed_sha256'])
        require(re.fullmatch('[0-9a-f]{64}', frame['expected_sha256']) is not None)
        require('/' + video + '/frames_1fps/' in frame['frame_path'])
        require(PurePosixPath(frame['frame_path']).name == f"frame_{frame['frame_index']:05d}.jpg")
        require(frame['resolved_timestamp_sec'] == frame['frame_index'])
        require(not frame['duplicate_of_seen_frame'] and not frame['duplicate_of_turn_frame'])
    for field in ('input_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens', 'output_tokens'):
        require(sum(t[field] for t in turns) == trace['total_' + field])
    require(math.isclose(sum(t['api_cost_usd'] for t in turns), trace['total_usd']))
    require(math.isclose(status['resource_totals']['usd'], trace['total_usd']))
    require(status['resource_totals']['images'] == trace['unique_images_transmitted'])
    require(status['resource_totals']['rounds'] == trace['rounds'])
    require(status['resource_totals']['provider_attempts'] == trace['total_api_attempts'])
    return question, reuse, scored, inspection, final


def replay_direct(manifest, data):
    question, reuse, scored, inspection, final = validate(manifest, data)
    trace = data['trace']
    print('MODE: FROZEN DIRECT REPLAY')
    print('No model inference is performed.')
    print('No frame retrieval or inspection is re-run.')
    print('Route: ' + manifest['route'] + ' | thesis lineage: frozen_reuse')
    print('Storage note: this R3 route is physically stored under the earlier formal archive, but the final thesis lineage explicitly retains it as frozen_reuse.')
    print('\nQUESTION\n' + question['question_text'])
    for option in question['answer_options']:
        print(f"{option['option_id']}: {option['text']}")
    print('Text source: matching frozen HV2 question record, not the omitted Direct input bytes.')
    print('\nPREPARED CONTEXT / MAP SUMMARY')
    print('R3 native map reference: ' + PurePosixPath(reuse['map_path']).name)
    print('Recorded map SHA256: ' + reuse['map_sha256'])
    print('Full source map exists in the archived Variant-C package but is not loaded by this compact replay.')
    print('\nRECORDED DIRECT DECISION')
    print('Turn 1: ' + inspection['action_type'])
    print('Recorded rationale: ' + excerpt(inspection['action_reason']))
    print('\nRECORDED INSPECTION REQUEST')
    tool = trace['provider_attempts'][0]['response_metadata']['tool_metadata'][0]
    print('Tool-call ID: ' + tool['id'])
    print('Requested timestamps (seconds): ' + ', '.join(str(t) for t in inspection['requested_timestamps_sec']))
    print('\nSELECTED FRAME TIMESTAMPS / HASHES')
    for f in inspection['resolved_frames']:
        print(f"{PurePosixPath(f['frame_path']).name} | {f['resolved_timestamp_sec']} s")
        print('  Recorded expected = observed SHA256: ' + f['expected_sha256'])
    print('These are recorded frame hashes; JPEG bytes are not re-verified or displayed.')
    print('\nRECORDED TOOL RESULT')
    print(f"Controller/provider records: {inspection['images_transmitted']} images transmitted; {inspection['duplicate_requests']} duplicates; disposition: {inspection['provider_status']}.")
    print('The image-bearing tool_result payload is not preserved locally.')
    print('No separate visual observation text is invented; the final rationale is shown below.')
    print('\nFROZEN FINAL ANSWER')
    selected = next(o for o in question['answer_options'] if o['option_id'] == trace['final_prediction'])
    print(selected['option_id'] + ' — ' + selected['text'])
    print('Recorded rationale: ' + excerpt(final['action_reason']))
    print('Route: terminal_success / final_answer; correctness: True (final scored artifact).')
    print('\nRECORDED RESOURCE USE (not replay resource use)')
    print(f"{trace['total_api_attempts']} API attempts; {trace['rounds']} turns; {trace['unique_images_transmitted']} images; {trace['correction_attempts']} corrections")
    print(f"Ordinary input: {trace['total_input_tokens']}; cache-write input: {trace['total_cache_creation_input_tokens']}; cache-read input: {trace['total_cache_read_input_tokens']}; output: {trace['total_output_tokens']} tokens")
    print(f"Recorded cache-aware cost: USD {trace['total_usd']:.7f}; route wall time: {trace['route_wall_time_sec']:.3f} s")
    print('\nPROVENANCE — all source identities and SHA-256 checks passed')
    print('Artifacts:')
    for label in ('route trace', 'route status', 'frozen-reuse manifest', 'final scored result', 'question record'):
        print('- ' + label)
    print('Manifest / full paths / hashes:')
    print('demo/examples/replay-direct.json')
    print('\nLIMITATIONS')
    for item in manifest['known_limitations']:
        print('- ' + item)
