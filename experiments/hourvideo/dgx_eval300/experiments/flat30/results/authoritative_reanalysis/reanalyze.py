#!/usr/bin/env python3
"""Offline-only Eval300 authoritative reanalysis. Reads frozen artifacts; writes beside itself."""
from pathlib import Path
import csv, glob, hashlib, json, math, random, re, statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone

HOME=Path('/home/naxucl'); OUT=Path(__file__).resolve().parent
FLAT1=HOME/'data/HourVideo/videoseal_original/runs_dgx_eval300_v1'
FLATR=HOME/'data/HourVideo/videoseal_original/eval300_timeout_retry_20260819T084545Z/original_retry'
RECON=HOME/'data/HourVideo/videoseal_original/eval300_reconciliation_audit_20260822T144251Z'
FORMAL=HOME/'data/HourVideo/experiments/hourvideo_v7_4_variant_c_budgets_v1/outputs/formal_eval300_20260821T095418Z'
UIDFILE=HOME/'data/HourVideo/benchmark/v1.0_release/hourvideo_eval300_v1_uids.txt'
SEED=20260826; STRICT=re.compile(r'^[A-E]$'); FINAL=re.compile(r'<final>\s*(.*?)\s*</final>',re.I|re.S)

def load(p): return json.loads(Path(p).read_text())
def norm(x): return str(x if x is not None else '').strip().upper()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pctile(xs,q):
    if not xs:return None
    z=sorted(float(x) for x in xs); pos=(len(z)-1)*q/100; lo=int(pos); hi=min(lo+1,len(z)-1); return z[lo]+(z[hi]-z[lo])*(pos-lo)
def desc(xs):
    xs=[float(x) for x in xs if x is not None]
    return {'n':len(xs),'mean':statistics.mean(xs) if xs else None,'median':statistics.median(xs) if xs else None,
            'p90':pctile(xs,90),'p95':pctile(xs,95)}
def index(root,kind):
    pat='*/metrics/*.json' if kind=='metric' else ('*/preds/*.json' if kind=='pred' else '*/*/trajectory.json')
    d=defaultdict(list)
    for p in sorted(root.glob(pat)):
        try: x=load(p); d[str(x.get('uid',''))].append((p,x))
        except Exception: pass
    return d
def stage(root): return {'root':root,'metric':index(root,'metric'),'pred':index(root,'pred'),'traj':index(root,'traj')}
def status(m):
    if not m:return 'missing'
    if str(m.get('status','')).lower()=='success':return 'success'
    if str(m.get('status','')).lower()=='timeout' or str(m.get('error_type','')).lower()=='timeout':return 'timeout'
    return 'error'
def one(d,u): return d.get(u,[None])[-1]

uids=[x for x in UIDFILE.read_text().splitlines() if x.strip()]
assert len(uids)==len(set(uids))==300
f1,fr=stage(FLAT1),stage(FLATR)
gt={}
for u in uids:
    z=one(f1['pred'],u); assert z
    gt[u]=norm(z[1].get('gt')); assert STRICT.fullmatch(gt[u])

def artifact(st,u):
    mp=one(st['metric'],u); pp=one(st['pred'],u); ts=st['traj'].get(u,[])
    m=mp[1] if mp else {}; p=pp[1] if pp else {}
    # Match finalizer behavior: require a completed trajectory with a nonempty answer for strict completion.
    good=[]
    for tp,t in ts:
        ans=norm(t.get('answer') or t.get('final_answer'))
        if t.get('finished_at') and ans: good.append((tp,t))
    tpair=good[-1] if good else (ts[-1] if ts else None)
    pred=norm(p.get('pred')); stt=status(m)
    strict=stt=='success' and bool(STRICT.fullmatch(pred)) and bool(good)
    return {'metric_path':str(mp[0]) if mp else '','pred_path':str(pp[0]) if pp else '',
            'trajectory_path':str(tpair[0]) if tpair else '', 'metric':m,'pred_obj':p,
            'trajectory':tpair[1] if tpair else {},'status':stt,'prediction':pred,
            'strict':strict,'correct':strict and pred==gt[u]}

flat={}; flat_source={}
for u in uids:
    a=artifact(f1,u); b=artifact(fr,u)
    if a['strict']: flat[u]=a; flat_source[u]='first_pass'
    elif b['strict']: flat[u]=b; flat_source[u]='retry'
    else: flat[u]=b if b['status']!='missing' else a; flat_source[u]='retry_unresolved' if b['status']!='missing' else 'first_pass_unresolved'

pre={'completed':sum(x['strict'] for x in flat.values()),'correct':sum(x['correct'] for x in flat.values()),
     'timeout':sum(x['status']=='timeout' for x in flat.values()),
     'success_strict_invalid':sum(x['status']=='success' and not x['strict'] for x in flat.values())}
expected={'completed':254,'correct':82,'timeout':36,'success_strict_invalid':10}
if pre!=expected:
    raise SystemExit('FLAT PRECONDITION FAILED '+json.dumps({'observed':pre,'expected':expected}))

groups={'flat':flat}
for name in ('h6','h15','h30'):
    man=load(FORMAL/name/'merged/per_question_manifest.json'); mm={x['uid']:x for x in man}
    assert set(mm)==set(uids)
    hs1,hsr=stage(FORMAL/name/'first_pass'),stage(FORMAL/name/'retry_1')
    d={}
    for u in uids:
        x=mm[u]; aa,bb=artifact(hs1,u),artifact(hsr,u)
        rec=aa if aa['strict'] else (bb if bb['status']!='missing' else aa)
        assert rec['strict']==bool(x.get('complete'))
        rec['merged']=x; rec['correct']=rec['strict'] and rec['prediction']==gt[u]
        d[u]=rec
    groups[name]=d

def events(rec):
    out=[]
    for i,s in enumerate(rec['trajectory'].get('steps',[]),1):
        resp=str(s.get('model_response') or '')
        action=(s.get('action') or {}).get('name')
        final=FINAL.search(resp)
        if not action and not (final and final.group(1).strip()): out.append((i,resp))
    return out
def traj_features(rec):
    t=rec['trajectory']; steps=t.get('steps',[]); acts=Counter((s.get('action') or {}).get('name') for s in steps)
    retrieval=[]; summaries=[]; inspectors=[]; frames=[]
    for s in steps:
        a=(s.get('action') or {}).get('name'); o=s.get('observation') or {}; oo=o.get('output') or {}
        if a=='visual_retrieve':
            retrieval.append((s.get('action') or {}).get('arguments',{})); summaries.append(oo.get('summary',''))
            md=o.get('metadata') or {}; prov=md.get('provenance') or []
            for q in prov:
                cand=q.get('candidate',{}); fine=q.get('fine',{}); frames.append(str(fine.get('frame_index') or fine.get('timestamp_sec') or cand.get('start_time','')))
        if a=='visual_inspect': inspectors.append({'arguments':(s.get('action') or {}).get('arguments',{}),'output':oo})
    calls=int(rec['metric'].get('planner_calls') or len(steps)); ev=events(rec)
    direct=(acts['visual_inspect']==0)
    fallback=bool(rec.get('merged',{}).get('fallback',False)) or any('fallback' in str(s).lower() for s in steps)
    return {'planner_calls':calls,'retrieval_calls':int(rec['metric'].get('visual_retrieve_calls') or acts['visual_retrieve']),'summarizer_calls':int(rec['metric'].get('visual_retrieve_calls') or acts['visual_retrieve']),
            'inspector_calls':int(rec['metric'].get('visual_inspect_calls') or acts['visual_inspect']),'frames':rec['metric'].get('sent_images') or rec.get('merged',{}).get('inspector_frames') or 0,
            'fallback':fallback,'direct':direct,'event_count':len(ev),'trajectory_length':len(steps),
            'retrieval_arguments':retrieval,'candidate_frame_ids':frames,'summaries':summaries,'inspectors':inspectors,
            'actions':[{'step':i+1,'name':(s.get('action') or {}).get('name'),'arguments':(s.get('action') or {}).get('arguments'),'response':s.get('model_response','')} for i,s in enumerate(steps)]}

features={g:{u:traj_features(r) for u,r in d.items()} for g,d in groups.items()}
summary={'generated_utc':datetime.now(timezone.utc).isoformat(),'frozen_uid_manifest':str(UIDFILE),'frozen_uid_sha256':sha(UIDFILE),
         'uid_count':300,'flat_precondition':pre,'definitions':{
         'strict_completed':'status=success; prediction full-matches ^[A-E]$; completed trajectory has finished_at and nonempty answer',
         'correct':'strict completed and normalized prediction equals artifact GT','timeout':'status=timeout OR error_type=timeout',
         'illegal_prediction':'final selected prediction does not full-match ^[A-E]$, regardless of status',
         'success_strict_invalid':'status=success but strict_completed is false',
         'e2e':'metric.elapsed_sec among strict completed only','overlap_warning':'timeout, illegal prediction, and strict invalid can overlap; do not sum them.'},'groups':{}}
for g,d in groups.items():
    f=features[g]; lat=[r['metric'].get('elapsed_sec') for r in d.values() if r['strict']]
    summary['groups'][g]={'strict_completed':sum(r['strict'] for r in d.values()),'correct':sum(r['correct'] for r in d.values()),
      'correct_over_300':sum(r['correct'] for r in d.values())/300,'correct_over_completed':sum(r['correct'] for r in d.values())/sum(r['strict'] for r in d.values()),
      'timeout':sum(r['status']=='timeout' for r in d.values()),'illegal_prediction':sum(not STRICT.fullmatch(r['prediction']) for r in d.values()),
      'success_strict_invalid':sum(r['status']=='success' and not r['strict'] for r in d.values()),'e2e_sec':desc(lat),
      'calls':{k:sum(x[k] for x in f.values()) for k in ('planner_calls','retrieval_calls','summarizer_calls','inspector_calls')},
      'visual_frames_sent':sum(int(x['frames'] or 0) for x in f.values()),'fallback_uid_count':sum(x['fallback'] for x in f.values()),
      'fallback_ratio':sum(x['fallback'] for x in f.values())/300,'direct_answer_uid_count':sum(x['direct'] for x in f.values()),
      'direct_answer_ratio':sum(x['direct'] for x in f.values())/300}

# Control-flow event records and associations. Recovery means a later valid action/final in the same stored attempt.
cf=[]; control={}
for g,d in groups.items():
    f=features[g]; evuids={u for u in uids if f[u]['event_count']}
    for u in uids:
        ev=events(d[u]); steps=d[u]['trajectory'].get('steps',[])
        for i,resp in ev:
            later=steps[i:]; recovered=d[u]['strict'] and any((s.get('action') or {}).get('name') or (lambda m: bool(m and m.group(1).strip()))(FINAL.search(str(s.get('model_response') or ''))) for s in later)
            cf.append({'config':g,'uid':u,'attempt_source':flat_source[u] if g=='flat' else ('retry' if '/retry_1/' in d[u]['trajectory_path'] else 'first_pass'),
                       'step_index':i,'same_attempt_recovered':int(recovered),'response_excerpt':resp.replace('\n',' ')[:500],'trajectory_path':d[u]['trajectory_path']})
    dist=Counter(f[u]['event_count'] for u in uids)
    def assoc(sub):
        ss=list(sub); return {'uids':len(ss),'timeout_rate':sum(d[u]['status']=='timeout' for u in ss)/len(ss) if ss else None,
          'trajectory_length':desc([f[u]['trajectory_length'] for u in ss]),'e2e_sec_all_available':desc([d[u]['metric'].get('elapsed_sec') for u in ss])}
    # Retry recovery: first attempt had event, selected retry is strict (Flat); for H groups inspect first-pass artifact explicitly.
    rr=0
    if g=='flat':
        for u in uids:
            if events(artifact(f1,u)) and flat_source[u]=='retry' and d[u]['strict']: rr+=1
    else:
        fst=stage(FORMAL/g/'first_pass')
        for u in uids:
            if events(artifact(fst,u)) and '/retry_1/' in d[u]['trajectory_path'] and d[u]['strict']: rr+=1
    control[g]={'planner_calls':sum(x['planner_calls'] for x in f.values()),'events':sum(x['event_count'] for x in f.values()),
      'events_per_100_planner_calls':100*sum(x['event_count'] for x in f.values())/sum(x['planner_calls'] for x in f.values()),
      'uids_with_event':len(evuids),'events_per_uid_distribution':dict(sorted(dist.items())),
      'same_attempt_recovered_uids':sum(any(row['uid']==u and row['same_attempt_recovered'] for row in cf if row['config']==g) for u in evuids),
      'retry_recovered_uids':rr,'parser_uid_association':assoc(evuids),'nonparser_uid_association':assoc(set(uids)-evuids),
      'interpretation':'descriptive association only; does not establish that parser events independently caused timeout or error.'}
summary['control_flow']=control

def mcnemar(b,c):
    n=b+c
    if not n:return 1.0
    k=min(b,c); return min(1.0,2*sum(math.comb(n,i) for i in range(k+1))/(2**n))
def bootstrap_diff(a,b,cluster=False,B=20000):
    rng=random.Random(SEED); a=list(map(float,a)); b=list(map(float,b)); vals=[]
    for _ in range(B):
        ix=[rng.randrange(len(a)) for _ in a]; vals.append(statistics.mean(b[i]-a[i] for i in ix))
    return [pctile(vals,2.5),pctile(vals,97.5)]
def cluster_boot(g, metric, B=20000):
    vids=sorted(set(u.split('_')[0] for u in uids)); rng=random.Random(SEED); vals=[]
    for _ in range(B):
        pick=[rng.choice(vids) for _ in vids]; rows=[u for v in pick for u in uids if u.split('_')[0]==v]
        if metric=='accuracy': vals.append(sum(groups[g][u]['correct']-groups['flat'][u]['correct'] for u in rows)/len(rows))
        elif metric=='completion': vals.append(sum(groups[g][u]['strict']-groups['flat'][u]['strict'] for u in rows)/len(rows))
        else:
            z=[groups[g][u]['metric'].get('elapsed_sec')-groups['flat'][u]['metric'].get('elapsed_sec') for u in rows if groups[g][u]['strict'] and groups['flat'][u]['strict']]
            if z: vals.append(statistics.mean(z))
    return [pctile(vals,2.5),pctile(vals,97.5)]
def wilcoxon_approx(ds):
    vals=[(abs(x),1 if x>0 else -1) for x in ds if x]
    if not vals:return None
    order=sorted(range(len(vals)),key=lambda i:vals[i][0]); ranks=[0.0]*len(vals); i=0
    while i<len(order):
        j=i+1
        while j<len(order) and vals[order[j]][0]==vals[order[i]][0]:j+=1
        rank=(i+1+j)/2
        for k in order[i:j]:ranks[k]=rank
        i=j
    wp=sum(r for r,(_,s) in zip(ranks,vals) if s>0); n=len(vals); mean=n*(n+1)/4
    counts=Counter(a for a,_ in vals); tie=sum(t*(t+1)*(2*t+1) for t in counts.values())
    var=(n*(n+1)*(2*n+1)-tie)/24
    z=(abs(wp-mean)-0.5)/math.sqrt(var) if var>0 else 0
    p=math.erfc(abs(z)/math.sqrt(2))
    return {'statistic':min(wp,n*(n+1)/2-wp),'pvalue_two_sided':p,'method':'normal approximation with tie correction and continuity correction'}
paired={}; rawps=[]
for g in ('h6','h15','h30'):
    a=groups['flat']; b=groups[g]
    def tab(field):
        av=[bool(a[u][field]) for u in uids]; bv=[bool(b[u][field]) for u in uids]
        return {'both_positive':sum(x and y for x,y in zip(av,bv)),'flat_positive_hier_negative':sum(x and not y for x,y in zip(av,bv)),
                'hier_positive_flat_negative':sum(y and not x for x,y in zip(av,bv)),'both_negative':sum(not x and not y for x,y in zip(av,bv))}
    acc=tab('correct'); comp=tab('strict'); acc['mcnemar_exact_two_sided_p']=mcnemar(acc['flat_positive_hier_negative'],acc['hier_positive_flat_negative']); comp['mcnemar_exact_two_sided_p']=mcnemar(comp['flat_positive_hier_negative'],comp['hier_positive_flat_negative'])
    diffs=[int(b[u]['correct'])-int(a[u]['correct']) for u in uids]
    rng=random.Random(SEED); boot=[statistics.mean(diffs[rng.randrange(300)] for _ in range(300)) for _ in range(20000)]
    acc['accuracy_difference_hier_minus_flat']=statistics.mean(diffs); acc['paired_uid_bootstrap_95ci']=[pctile(boot,2.5),pctile(boot,97.5)]; acc['video_cluster_bootstrap_95ci']=cluster_boot(g,'accuracy')
    comp['completion_difference_hier_minus_flat']=(sum(b[u]['strict'] for u in uids)-sum(a[u]['strict'] for u in uids))/300; comp['video_cluster_bootstrap_95ci']=cluster_boot(g,'completion')
    common=[u for u in uids if a[u]['strict'] and b[u]['strict']]; la=[a[u]['metric']['elapsed_sec'] for u in common]; lb=[b[u]['metric']['elapsed_sec'] for u in common]; dd=[y-x for x,y in zip(la,lb)]
    latency={'paired_n':len(common),'flat':desc(la),'hier':desc(lb),'hier_minus_flat':desc(dd),'paired_bootstrap_mean_difference_95ci':bootstrap_diff(la,lb),
             'video_cluster_bootstrap_mean_difference_95ci':cluster_boot(g,'latency'),'wilcoxon':wilcoxon_approx(dd)}
    paired[g]={'accuracy':acc,'completion':comp,'latency':latency}; rawps += [(g,'accuracy',acc['mcnemar_exact_two_sided_p']),(g,'completion',comp['mcnemar_exact_two_sided_p'])]
# Holm across the three planned comparisons separately for accuracy and completion.
for metric in ('accuracy','completion'):
    z=sorted([(g,p) for g,m,p in rawps if m==metric],key=lambda x:x[1]); running=0
    for rank,(g,p) in enumerate(z): running=max(running,min(1,(len(z)-rank)*p)); paired[g][metric]['holm_adjusted_p']=running

# Per-UID table.
rows=[]
for u in uids:
    r={'uid':u,'video_id':u.split('_')[0],'gt':gt[u]}
    for g,d in groups.items():
        x=d[u]; f=features[g][u]
        r.update({f'{g}_prediction':x['prediction'],f'{g}_strict_completed':int(x['strict']),f'{g}_correct':int(x['correct']),f'{g}_timeout':int(x['status']=='timeout'),
          f'{g}_illegal_prediction':int(not STRICT.fullmatch(x['prediction'])),f'{g}_success_strict_invalid':int(x['status']=='success' and not x['strict']),f'{g}_elapsed_sec':x['metric'].get('elapsed_sec'),
          f'{g}_planner_calls':f['planner_calls'],f'{g}_retrieval_calls':f['retrieval_calls'],f'{g}_summarizer_calls':f['summarizer_calls'],f'{g}_inspector_calls':f['inspector_calls'],f'{g}_frames':f['frames'],f'{g}_fallback':int(f['fallback']),f'{g}_direct':int(f['direct']),f'{g}_parser_events':f['event_count'],f'{g}_trajectory_length':f['trajectory_length']})
    rows.append(r)

def write_tsv(path,rows):
    keys=list(rows[0]) if rows else []
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys,delimiter='\t',extrasaction='ignore'); w.writeheader()
        for row in rows: w.writerow({k:(v.replace('\r','\\r').replace('\n','\\n') if isinstance(v,str) else v) for k,v in row.items()})
write_tsv(OUT/'per_uid.tsv',rows); write_tsv(OUT/'control_flow_events.tsv',cf)

# Discordant sets.
for g in ('h6','h15','h30'):
    sets={'flat_correct_hier_wrong':[u for u in uids if groups['flat'][u]['correct'] and not groups[g][u]['correct']],
          'hier_correct_flat_wrong':[u for u in uids if groups[g][u]['correct'] and not groups['flat'][u]['correct']],
          'flat_complete_hier_incomplete':[u for u in uids if groups['flat'][u]['strict'] and not groups[g][u]['strict']],
          'hier_complete_flat_incomplete':[u for u in uids if groups[g][u]['strict'] and not groups['flat'][u]['strict']],
          'both_complete_prediction_different':[u for u in uids if groups[g][u]['strict'] and groups['flat'][u]['strict'] and groups[g][u]['prediction']!=groups['flat'][u]['prediction']]}
    for k,v in sets.items(): (OUT/'discordant_uid_sets'/f'{g}_{k}.txt').write_text(''.join(x+'\n' for x in v))

# Per-video statistics.
pv=[]
for v in sorted(set(u.split('_')[0] for u in uids)):
    us=[u for u in uids if u.split('_')[0]==v]
    row={'video_id':v,'n_questions':len(us)}
    for g in groups:
        row.update({f'{g}_correct':sum(groups[g][u]['correct'] for u in us),f'{g}_completed':sum(groups[g][u]['strict'] for u in us),f'{g}_timeout':sum(groups[g][u]['status']=='timeout' for u in us)})
    for g in ('h6','h15','h30'):
        z=[groups[g][u]['metric'].get('elapsed_sec')-groups['flat'][u]['metric'].get('elapsed_sec') for u in us if groups[g][u]['strict'] and groups['flat'][u]['strict']]
        row[f'{g}_paired_latency_n']=len(z); row[f'{g}_paired_latency_hier_minus_flat_mean']=statistics.mean(z) if z else ''
    pv.append(row)
write_tsv(OUT/'per_video_statistics.tsv',pv)

# Deterministic H15 review packet: disjoint quotas, seed-recorded, conservative artifact-only labels.
rng=random.Random(SEED)
def sample(pool,n,used):
    z=sorted(set(pool)-used); rng.shuffle(z); take=z[:min(n,len(z))]; used.update(take); return take
used=set(); fc=[u for u in uids if groups['flat'][u]['correct'] and not groups['h15'][u]['correct']]; hc=[u for u in uids if groups['h15'][u]['correct'] and not groups['flat'][u]['correct']]
bad=[u for u in uids if groups['h15'][u]['status']=='timeout' or (groups['h15'][u]['status']=='success' and not groups['h15'][u]['strict'])]
diff=[u for u in uids if groups['flat'][u]['strict'] and groups['h15'][u]['strict'] and groups['flat'][u]['prediction']!=groups['h15'][u]['prediction']]
chosen=[('flat_correct_h15_wrong',u) for u in sample(fc,20,used)]+[('h15_correct_flat_wrong',u) for u in sample(hc,10,used)]+[('h15_timeout_or_invalid',u) for u in sample(bad,5,used)]+[('both_completed_behavior_different',u) for u in sample(diff,5,used)]
packet=[]
for bucket,u in chosen:
    a,b=groups['flat'][u],groups['h15'][u]; fa,fb=features['flat'][u],features['h15'][u]
    if b['status']=='timeout': cause='timeout_or_runtime_limit'; basis='H-15 terminal metric is timeout.'
    elif fb['event_count'] and not b['strict']: cause='parser_or_protocol_instability'; basis='Reconstructed protocol event(s) and incomplete H-15 artifact; causal sufficiency not established.'
    elif fb['event_count']: cause='multiple_possible_causes'; basis='Parser event observed but artifact does not establish it caused the answer difference.'
    elif fa['candidate_frame_ids']!=fb['candidate_frame_ids']: cause='retrieval_candidate_difference'; basis='Returned candidate identifiers/timestamps differ; this is not proof of recall failure without visual GT evidence.'
    else: cause='undetermined'; basis='Existing text artifacts do not isolate a cause.'
    packet.append({'sample_bucket':bucket,'uid':u,'video_id':u.split('_')[0],'question':b['trajectory'].get('question',''),'options':b['trajectory'].get('question',''),'gt':gt[u],
      'flat_prediction':a['prediction'],'h15_prediction':b['prediction'],'flat_retrieval_count':fa['retrieval_calls'],'h15_retrieval_count':fb['retrieval_calls'],
      'flat_candidate_frame_ids_timestamps':json.dumps(fa['candidate_frame_ids']),'h15_candidate_frame_ids_timestamps':json.dumps(fb['candidate_frame_ids']),
      'flat_summarizer_outputs':json.dumps(fa['summaries'],ensure_ascii=False),'h15_summarizer_outputs':json.dumps(fb['summaries'],ensure_ascii=False),
      'flat_planner_actions':json.dumps(fa['actions'],ensure_ascii=False),'h15_planner_actions':json.dumps(fb['actions'],ensure_ascii=False),
      'flat_inspector_artifacts':json.dumps(fa['inspectors'],ensure_ascii=False),'h15_inspector_artifacts':json.dumps(fb['inspectors'],ensure_ascii=False),
      'flat_parser_events':fa['event_count'],'h15_parser_events':fb['event_count'],'flat_timeout':int(a['status']=='timeout'),'h15_timeout':int(b['status']=='timeout'),
      'flat_fallback':int(fa['fallback']),'h15_fallback':int(fb['fallback']),'flat_e2e':a['metric'].get('elapsed_sec'),'h15_e2e':b['metric'].get('elapsed_sec'),
      'trajectory_difference_summary':f"steps {fa['trajectory_length']} vs {fb['trajectory_length']}; retrieval {fa['retrieval_calls']} vs {fb['retrieval_calls']}; inspector {fa['inspector_calls']} vs {fb['inspector_calls']}",
      'conservative_attribution':cause,'attribution_basis':basis,'flat_trajectory_path':a['trajectory_path'],'h15_trajectory_path':b['trajectory_path']})
write_tsv(OUT/'h15_error_review_packet.tsv',packet)

# Decision gate: automatic extension is blocked because the review identifies systematic artifact observability gaps;
# this does not assert a runtime implementation defect or retrieval design failure.
decision={'status':'BLOCK_FULL_EXTENSION','basis':['H-15 causal attribution cannot be completed from text artifacts alone for candidate differences; no ground-truth temporal evidence is present.','Required review packet fields are artifact-derived, but Flat/H15 candidate provenance is not schema-identical and visual recall failure cannot be proven.','Old paired_with_flat uses the stale 219-valid first pass, so extension decisions based on it must be withheld pending corrected review.'],
          'not_claimed':['No proof of systematic H-15 retrieval design failure.','No claim that parser events independently caused timeout or wrong answers.'],
          'scope':'Gate blocks approving full Flat/H-15 extension from this audit alone; it does not alter or invalidate frozen runs.'}

stale={'hierarchical_finalizer_flat_reference':'219-valid first-pass Flat, not 254-valid final merged Flat','old_paired_with_flat_status':'stale; original files not overwritten',
       'old_parser_failures_problem':'finalize_profile-style field infers events from trajectory-level invalid-response notes while iterating steps; it can repeat-count a single event and omit timeout attempts when no completed trajectory is selected. Reanalysis reconstructs each step from action and <final> content.'}
summary['stale_evidence']=stale; summary['paired_statistics']=paired; summary['review_packet']={'seed':SEED,'requested_quotas':{'flat_correct_h15_wrong':20,'h15_correct_flat_wrong':10,'h15_timeout_or_invalid':5,'both_completed_behavior_different':5},'actual_n':len(packet),'sampling':'sorted eligible UID set; Python random.Random(seed) shuffle; disjoint in quota order'}; summary['decision_gate']=decision
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
(OUT/'paired_statistics.json').write_text(json.dumps(paired,indent=2,ensure_ascii=False)+'\n')
(OUT/'decision_gate.json').write_text(json.dumps(decision,indent=2,ensure_ascii=False)+'\n')

report=f'''# HourVideo Eval300 authoritative reanalysis

Generated: {summary['generated_utc']}  
Frozen UID manifest: `{UIDFILE}`  
SHA-256: `{sha(UIDFILE)}`  
Coverage: 300 unique UIDs; Flat, H-6, H-15 and H-30 all cover the identical frozen set.

## Verified authoritative counts

| config | completed/300 | correct/300 | correct/completed | timeout | illegal pred | success strict-invalid | E2E mean / median / P90 / P95 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
for g,s in summary['groups'].items():
    e=s['e2e_sec']; report+=f"| {g} | {s['strict_completed']} | {s['correct']} ({100*s['correct_over_300']:.2f}%) | {100*s['correct_over_completed']:.2f}% | {s['timeout']} | {s['illegal_prediction']} | {s['success_strict_invalid']} | {e['mean']:.3f} / {e['median']:.3f} / {e['p90']:.3f} / {e['p95']:.3f} |\n"
report+='''
Timeout, illegal prediction, and strict invalid can overlap and must not be added to derive incompletion.

## Corrected paired results

All accuracy comparisons use all 300 UIDs, treating incomplete as not correct. Completion comparisons pair strict-completed state. Latency uses only jointly strict-completed UIDs. Exact values, Holm adjustment, paired bootstrap, Wilcoxon, and video-cluster bootstrap are in `paired_statistics.json`.

'''
for g,x in paired.items():
    a=x['accuracy']; c=x['completion']; l=x['latency']; report+=f"- {g}: accuracy discordance Flat-only={a['flat_positive_hier_negative']}, Hier-only={a['hier_positive_flat_negative']}, exact p={a['mcnemar_exact_two_sided_p']:.6g}, Holm p={a['holm_adjusted_p']:.6g}, difference={100*a['accuracy_difference_hier_minus_flat']:.2f} pp, CI={a['paired_uid_bootstrap_95ci']}; completion exact p={c['mcnemar_exact_two_sided_p']:.6g}, Holm p={c['holm_adjusted_p']:.6g}; latency n={l['paired_n']}, mean Hier-Flat={l['hier_minus_flat']['mean']:.3f}s, paired bootstrap CI={l['paired_bootstrap_mean_difference_95ci']}.\n"
report+='''

## Stale and corrected evidence

- The old hierarchical `paired_with_flat` reference used first-pass Flat (219 strict-valid), not authoritative merged Flat (254). Those results are stale; no source file was overwritten.
- The old `parser_failures` counter is not an event counter. It can apply a trajectory-level invalid-response note while iterating multiple steps, and timeout attempts may disappear when only completed trajectories are selected. This report reconstructs events step by step: no parsed action and no nonempty `<final>`.

## H-15 attribution limits

Directly observable: terminal status, predictions, action sequence, returned candidate metadata, summaries, Inspector inputs/outputs, reconstructed protocol events, frames recorded as sent, fallback flags present in artifacts, and elapsed time. Candidate sets often differ and timeout/protocol events are observable.

Not directly provable without ground-truth temporal visual evidence: that a candidate difference is a recall failure, that a summary omission alone caused a wrong answer, or that a parser event independently caused timeout/error. Automated labels are deliberately conservative; `retrieval_candidate_difference` means only an observed candidate difference.

The dataset contains 300 questions from 12 videos, not 300 independent video samples. Both UID bootstrap and 12-video cluster bootstrap are reported.

## Decision gate

**{decision['status']}**. This audit does not provide enough artifact-only evidence to approve full extension, nor enough evidence to declare a systematic retrieval redesign requirement. Review the deterministic packet (seed {SEED}) and resolve the stated observability/stale-comparison issues before authorizing remaining Flat/H-15 execution.

## Execution boundary

This was an offline filesystem analysis. It did not start a model, GPU workload, API call, retry, rescoring run, or remaining-question run. Original results, runtime, indexer, planner, inspector, configurations, trajectories, predictions, and formal manifests were not modified.
'''
(OUT/'FINAL_AUDIT_REPORT.md').write_text(report)

# Manifest last, excluding itself to avoid self-reference.
files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='MANIFEST.sha256')
(OUT/'MANIFEST.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(OUT)}\n' for p in files))
print(json.dumps({'out':str(OUT),'flat_precondition':pre,'groups':summary['groups'],'decision':decision['status'],'packet_n':len(packet)},indent=2))
