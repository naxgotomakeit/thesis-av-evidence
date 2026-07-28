"""Manual-review audio enhancement diagnostic; independent of frozen V0 answer generation."""
from __future__ import annotations
import argparse, csv, json, os, resource, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.experiments.egopolice_audio_enhancement_diagnostic.core import *  # noqa: F401,F403,E402

def rss(): return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
def preflight(cfg):
    mount=Path(cfg['required_mount']); runtime=Path(cfg['runtime_root'])
    if not mount.is_dir(): raise RuntimeError(f'External mount unavailable: {mount}')
    if not runtime.is_relative_to(mount): raise RuntimeError('runtime_root must be on external mount')
    runtime.mkdir(parents=True,exist_ok=True); free=os.statvfs(runtime).f_bavail*os.statvfs(runtime).f_frsize
    if free < 20*2**30: raise RuntimeError(f'Less than 20 GiB free on external volume: {free/2**30:.2f} GiB')
    probe=runtime/'.write_probe'; probe.write_text('ok'); probe.unlink()
    return {'free_gib':round(free/2**30,2),'mount':str(mount),'runtime_root':str(runtime)}

def review_html(path, video_id, duration, vad, original, enhanced, windows, clips, wav):
    rows=[]
    for r in vad:
        o=next((x for x in original if x['segment_id']==r['segment_id']),{}); e=next((x for x in enhanced if x['segment_id']==r['segment_id']),{})
        rows.append(f"<tr><td>{r['start_sec']:.3f}–{r['end_sec']:.3f}</td><td>{o.get('raw_transcript','')}</td><td>{e.get('raw_transcript','')}</td><td>{e.get('asr_status','')}</td><td>{e.get('avg_logprob','')}</td></tr>")
    sed=''.join(f"<tr><td>{w['start_sec']:.1f}–{w['end_sec']:.1f}</td><td>{json.dumps(w['top_labels'],ensure_ascii=False)}</td><td>{json.dumps(w['target_event_scores'],ensure_ascii=False)}</td><td>{','.join(w['candidate_types'])}</td></tr>" for w in windows)
    aud=''.join(f"<li>{c['start_sec']:.3f}–{c['end_sec']:.3f} <audio controls preload='none' src='{c['path']}'></audio> <code>{c['candidate_type']}</code></li>" for c in clips)
    html=f"""<!doctype html><meta charset='utf-8'><title>Audio review {video_id}</title><style>body{{font:14px system-ui;max-width:1600px;margin:24px;color:#182433}}table{{border-collapse:collapse;width:100%;margin:12px 0}}td,th{{border:1px solid #ccd5dd;padding:6px;vertical-align:top;text-align:left}}th{{background:#eaf0f5}}audio{{width:260px}}code{{white-space:pre-wrap}}</style><h1>Audio manual review: {video_id}</h1><p>Duration: {duration:.3f}s. Full audio: <audio controls preload='none' src='file://{wav}'></audio></p><h2>VAD / ASR timeline</h2><table><tr><th>Time</th><th>Original ASR</th><th>Enhanced ASR</th><th>Status</th><th>avg_logprob</th></tr>{''.join(rows)}</table><h2>AST windows</h2><table><tr><th>Time</th><th>Top labels</th><th>Target scores</th><th>Candidates</th></tr>{sed}</table><h2>Candidate clips</h2><ul>{aud}</ul>"""
    path.write_text(html,encoding='utf-8')

def process(item,cfg,out_root):
    import whisper
    t0=time.perf_counter(); stages={}; vid=item['dataset_video_id']; safe=safe_id(vid); runtime=Path(cfg['runtime_root'])/safe; audio=runtime/'audio'; clips_dir=runtime/'review_clips'; audio.mkdir(parents=True,exist_ok=True); clips_dir.mkdir(parents=True,exist_ok=True); output=out_root/safe; output.mkdir(parents=True,exist_ok=True)
    video=Path(cfg['video_root'])/item['filename']; wav=audio/'full_audio.wav'; t=time.perf_counter(); extract_audio(video,wav,Path(cfg['ffmpeg'])); waveform,meta=decode_audio(wav,int(cfg['sample_rate'])); stages['audio_extraction_and_decode']=round(time.perf_counter()-t,3)
    t=time.perf_counter(); vad=load_vad_and_regions(waveform,int(cfg['sample_rate']),cfg); stages['vad']=round(time.perf_counter()-t,3); write_json(output/'vad_segments.json',{'video_id':vid,'segments':vad}); write_json(output/'speech_segments_original.json',{'video_id':vid,'segments':[]});
    t=time.perf_counter(); model=whisper.load_model(cfg['whisper_model'],device='cpu',download_root=str(audio/'whisper_cache')); stages['whisper_model_load']=round(time.perf_counter()-t,3)
    t=time.perf_counter(); original=transcribe(model,waveform,int(cfg['sample_rate']),vad,cfg,language=None,task=None); stages['asr_original']=round(time.perf_counter()-t,3)
    t=time.perf_counter(); enhanced=transcribe(model,waveform,int(cfg['sample_rate']),vad,cfg,language=cfg['asr_language'],task=cfg['asr_task']); stages['asr_enhanced']=round(time.perf_counter()-t,3)
    fallback_regions_list=fallback_regions(enhanced,meta['duration_sec'],float(cfg['asr_fallback_context_sec'])); t=time.perf_counter(); fallback=transcribe(model,waveform,int(cfg['sample_rate']),fallback_regions_list,cfg,language='en',task='transcribe'); stages['asr_fallback']=round(time.perf_counter()-t,3)
    fallback_by={x['segment_id']:x for x in fallback}; records=[]
    for i,e in enumerate(enhanced):
        e=dict(e); e['original_transcript']=original[i].get('raw_transcript',''); e['fallback_used']=False
        if e['segment_id'] in fallback_by:
            f=dict(fallback_by[e['segment_id']]); f['fallback_used']=True; f['original_start_sec']=e['start_sec']; f['original_end_sec']=e['end_sec']; f['asr_status']='asr_uncertain' if not f['raw_transcript'] or f.get('detected_language') not in (None,'en') else 'fallback_ok'; e=f
        records.append(e)
    write_json(output/'speech_segments_original.json',{'video_id':vid,'segments':original}); write_json(output/'speech_segments_enhanced.json',{'video_id':vid,'segments':records}); write_json(output/'asr_fallback_records.json',{'video_id':vid,'records':fallback})
    (output/'full_transcript_original.txt').write_text('\n'.join(f"[{x['start_sec']:.3f}-{x['end_sec']:.3f}] {x['raw_transcript']}" for x in original)+'\n',encoding='utf-8'); (output/'full_transcript_enhanced.txt').write_text('\n'.join(f"[{x['start_sec']:.3f}-{x['end_sec']:.3f}] {x['raw_transcript']} [{x['asr_status']}]" for x in records)+'\n',encoding='utf-8')
    t=time.perf_counter(); windows,sedmeta=run_ast(waveform,int(cfg['sample_rate']),cfg,cfg['sed_model'],Path(cfg['runtime_root'])/'model_cache'); stages['sound_event_detection']=round(time.perf_counter()-t,3); (output/'sound_event_windows.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in windows)+'\n',encoding='utf-8'); candidates=[{'start_sec':w['start_sec'],'end_sec':w['end_sec'],'candidate_type':t,'score':w['target_event_scores'].get(t,0.0),'alternatives':w['top_labels'],'window_id':w['window_id']} for w in windows for t in w['candidate_types']]; write_json(output/'target_sound_candidates.json',{'video_id':vid,'candidates':candidates,'model':sedmeta})
    clip_records=[]; allclip=[]
    for r in records:
        if r.get('fallback_used') or r.get('asr_status')=='asr_uncertain': allclip.append((r['start_sec'],r['end_sec'],'asr_fallback' if r.get('fallback_used') else 'asr_uncertain'))
    for c in candidates: allclip.append((c['start_sec'],c['end_sec'],c['candidate_type']))
    t=time.perf_counter()
    for start,end,typ in allclip:
        s=max(0,start-float(cfg['clip_context_sec'])); e=min(meta['duration_sec'],end+float(cfg['clip_context_sec'])); name=f'{s:.3f}__{e:.3f}__{typ}.wav'; dest=clips_dir/name; export_clip(wav,dest,s,e,Path(cfg['ffmpeg'])); clip_records.append({'start_sec':s,'end_sec':e,'candidate_type':typ,'path':dest.as_posix()})
    stages['review_clip_export']=round(time.perf_counter()-t,3)
    write_json(output/'annotation_comparison.json',{'video_id':vid,'mode':'manual_review_only','automatic_ground_truth_scoring':False,'review_instructions':'Compare against the original video and human annotation manually; no Ground Truth fields were read or modified.'}); review_html(output/'audio_review.html',vid,meta['duration_sec'],vad,original,records,windows,clip_records,wav)
    elapsed=round(time.perf_counter()-t0,3); run_manifest={'experiment_id':cfg['experiment_id'],'video_id':vid,'video_path':video.as_posix(),'runtime_root':runtime.as_posix(),'output_root':output.as_posix(),'status':'completed','audio':meta,'stages':stages,'peak_rss_bytes':rss(),'mps_available':False,'device':'cpu','whisper_model':cfg['whisper_model'],'sed':sedmeta,'cache':{'whisper_cache':str(audio/'whisper_cache'),'sed_cache':str(Path(cfg['runtime_root'])/'model_cache'),'cache_policy':'external SSD'},'elapsed_sec':elapsed,'manual_review_only':True,'automatic_ground_truth_scoring':False,'annotation_files_read':False}; write_json(output/'run_manifest.json',run_manifest)
    report=f"# Audio enhancement diagnostic\n\n- Video: `{vid}`\n- Duration: {meta['duration_sec']:.3f}s\n- VAD segments: {len(vad)}\n- Original non-empty ASR: {sum(bool(x['raw_transcript']) for x in original)}/{len(original)}\n- Enhanced non-empty ASR: {sum(bool(x['raw_transcript']) for x in records)}/{len(records)}\n- Fallback records: {len(fallback)}\n- AST windows: {len(windows)}\n- Candidate clips: {len(clip_records)}\n- Elapsed seconds: {elapsed:.2f}\n- Peak RSS: {rss()} bytes\n\nManual review only; no automatic precision/recall or Ground Truth scoring was performed.\n"; (output/'REPORT.md').write_text(report,encoding='utf-8')
    return {'video_id':vid,'safe_video_id':safe,'duration_sec':meta['duration_sec'],'vad_segments':len(vad),'original_nonempty_asr':sum(bool(x['raw_transcript']) for x in original),'enhanced_nonempty_asr':sum(bool(x['raw_transcript']) for x in records),'fallback_count':len(fallback),'sed_windows':len(windows),'candidate_count':len(candidates),'clip_count':len(clip_records),'elapsed_sec':elapsed,'peak_rss_bytes':rss(),'device':'cpu','stage_times':stages}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/experiments/egopolice_audio_enhancement_diagnostic_v0_1.json'); args=ap.parse_args(); cfg=json.loads(Path(args.config).read_text()); pre=preflight(cfg); out=ROOT/cfg['output_root']; out.mkdir(parents=True,exist_ok=True); started=time.time(); summaries=[]
    for item in cfg['videos']: summaries.append(process(item,cfg,out))
    summary={'experiment_id':cfg['experiment_id'],'status':'completed','manual_review_only':True,'automatic_ground_truth_scoring':False,'preflight':pre,'model':cfg['sed_model'],'videos':summaries,'elapsed_sec':time.time()-started}; write_json(out/'summary.json',summary); write_json(out/'RUN_PROVENANCE.json',{'config':args.config,'runtime_root':cfg['runtime_root'],'videos':[x['video_id'] for x in summaries],'no_claude':True,'no_storyline':True,'no_annotation_read':True,'sed_model':cfg['sed_model']}); write_json(out/'FROZEN_CONFIG_SNAPSHOT.json',{'experiment_id':cfg['experiment_id'],'base':'egopolice-v0-naive-av-storyline-freeze','algorithm_change':'audio evidence extraction only','visual_and_answer_unchanged':True,'config':cfg});
    with (out/'per_video_summary.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=summaries[0].keys()); w.writeheader(); w.writerows(summaries)
    with (out/'speech_diagnostic.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=['video_id','vad_segments','original_nonempty_asr','enhanced_nonempty_asr','fallback_count','elapsed_sec','device']); w.writeheader(); w.writerows({k:x[k] for k in w.fieldnames} for x in summaries)
    with (out/'sound_event_diagnostic.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=['video_id','sed_windows','candidate_count','clip_count']); w.writeheader(); w.writerows({k:x[k] for k in w.fieldnames} for x in summaries)
    (out/'experiment_report.md').write_text('# Audio enhancement diagnostic\n\nManual-review case analysis for exactly the three user-specified videos. No Ground Truth scoring, Claude, storyline generation, or V0 modification.\n',encoding='utf-8')
if __name__=='__main__': main()
