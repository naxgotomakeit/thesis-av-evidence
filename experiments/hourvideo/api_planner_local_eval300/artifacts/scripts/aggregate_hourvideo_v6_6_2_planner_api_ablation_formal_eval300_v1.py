#!/usr/bin/env python3
"""Read-only canonical aggregation for the completed Planner-only API Eval300 run.

This program deliberately reads only the formal namespace and frozen benchmark
annotation.  It writes derived analysis solely to canonical_summary_v1, which is
excluded from the immutable raw-closure hash.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FORMAL = ROOT / "outputs/experiments/hourvideo_v6_6_2_planner_api_ablation_formal_eval300_v1"
OUT = FORMAL / "canonical_summary_v1"
MANIFEST = FORMAL / "manifests/formal_eval300_manifest.json"
ATTEMPTS = FORMAL / "telemetry/api_attempts.jsonl"
STARTS = FORMAL / "telemetry/api_request_starts.jsonl"
LEDGER = FORMAL / "telemetry/budget_ledger.jsonl"
ANNOTATION = Path("${PROJECT_MSC_ROOT}/HourVideo/benchmark/v1.0_release/json/dev_v1.0_annotations.json")
EXPECTED_MANIFEST_SHA = "1adb56fcc90b8fbe2b6e9b41de62967c668b9e8b84336555708deaaf62beff5f"
SIDES = ("r1_av", "r3_2")
SEED = 20260904


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def loadl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def dump(name: str, data: Any) -> None:
    (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def freeze_raw() -> dict[str, Any]:
    rows = []
    for p in sorted(FORMAL.rglob("*")):
        if not p.is_file() or OUT in p.parents:
            continue
        rows.append({"path": str(p.relative_to(FORMAL)), "bytes": p.stat().st_size, "sha256": sha_file(p)})
    h = hashlib.sha256("".join(f"{r['sha256']}  {r['path']}\n" for r in rows).encode()).hexdigest()
    return {"tree_sha256": h, "file_count": len(rows), "total_bytes": sum(r["bytes"] for r in rows), "files": rows}


def dist(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None, "p95": None, "sum": 0.0}
    a = sorted(values)
    def pct(q: float) -> float:
        i = (len(a) - 1) * q; lo, hi = math.floor(i), math.ceil(i)
        return a[lo] if lo == hi else a[lo] + (a[hi] - a[lo]) * (i - lo)
    return {"n": len(a), "sum": sum(a), "mean": mean(a), "median": median(a), "p90": pct(.9), "p95": pct(.95)}


def wilson(k: int, n: int) -> list[float] | None:
    if not n: return None
    z = 1.959963984540054; p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d; r = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return [c-r, c+r]


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if not n: return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c)+1)) / (2**n)
    return min(1.0, 2*tail)


def bootstrap(diffs: list[float], reps: int = 20000) -> dict[str, Any]:
    rng = random.Random(SEED); n = len(diffs)
    if not n: return {"seed": SEED, "reps": reps, "n": 0, "ci95": None}
    vals = sorted(sum(diffs[rng.randrange(n)] for _ in range(n))/n for _ in range(reps))
    return {"seed": SEED, "reps": reps, "n": n, "mean_difference": mean(diffs), "ci95": [vals[int(.025*(reps-1))], vals[int(.975*(reps-1))]]}


def status_for(qid: str, side: str) -> tuple[dict[str, Any], Path]:
    p = FORMAL / "live/cases" / qid / side / "route_status.json"
    return load(p), p


def final_for(qid: str, side: str) -> dict[str, Any] | None:
    p = FORMAL / "live/cases" / qid / side / "final_answer.json"
    return load(p) if p.is_file() else None


def route_events(qid: str, side: str) -> list[dict[str, Any]]:
    p = FORMAL / "live/cases" / qid / "route_events.jsonl"
    return [x for x in loadl(p) if x.get("route") == side] if p.is_file() else []


def local_attempts(qid: str, side: str) -> list[dict[str, Any]]:
    p = FORMAL / "live/cases" / qid / "model_attempts.jsonl"
    return [x for x in loadl(p) if x.get("route") == side] if p.is_file() else []


def failure_category(reason: str | None) -> str | None:
    s = (reason or "").lower()
    if not s: return None
    if "planner_provider_exhausted" in s: return "planner_provider_exhausted"
    if "maximum context length" in s or "context_overflow" in s: return "downstream_local_context_overflow"
    if "visual structured response reached max_tokens" in s: return "fine_visual_max_tokens_or_validation_exhaustion"
    if "direct final contract exhausted" in s: return "final_contract_exhaustion"
    if "shared" in s and "contract exhausted" in s: return "shared_contract_exhaustion"
    if "provider" in s or "http" in s or "service" in s: return "provider_service_runtime_error"
    return "other_evidenced_failure"


def all_gold() -> dict[str, str]:
    raw = load(ANNOTATION); result = {}
    for video in raw.values():
        for q in video.get("benchmark_dataset", []): result[q["qid"]] = q["correct_answer_label"].strip().upper()
    return result


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n=len(rows); correct=sum(x["correct"] for x in rows); pred=sum(x["prediction_present"] for x in rows)
    normal=sum(x["execution_status"] == "normal_success" for x in rows)
    downgraded=sum(x["execution_status"] == "downgraded_recovery" for x in rows); failed=sum(x["execution_status"] == "failed" for x in rows)
    return {"denominator":n,"correct":correct,"accuracy":correct/n,"accuracy_percent":100*correct/n,"wilson_95":wilson(correct,n),"predictions_present":pred,"prediction_completion_rate":pred/n,"strict_completion_definition":"execution_status == normal_success","strict_completion":normal,"strict_completion_rate":normal/n,"missing_predictions":n-pred,"normal_success":normal,"downgraded_recovery":downgraded,"failed":failed,"reasoning_termination":dict(sorted(Counter(str(x["reasoning_termination"] or "null") for x in rows).items()))}


def main() -> None:
    # Sidecar files are derived only.  Rebuilding them is safe; raw closure is
    # independently frozen before and after each build.
    OUT.mkdir(parents=True, exist_ok=True)
    raw_before = freeze_raw()
    manifest_bytes = MANIFEST.read_bytes(); manifest = json.loads(manifest_bytes)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    errors=[]; qids=[q["question_id"] for q in manifest["questions"]]
    expected=[(q,s) for q in qids for s in SIDES]
    if manifest_sha != EXPECTED_MANIFEST_SHA: errors.append("formal manifest SHA mismatch")
    if len(qids)!=300 or len(set(qids))!=300: errors.append("manifest is not 300 unique questions")
    if manifest.get("routes") != list(SIDES): errors.append("manifest routes mismatch")
    capacities={q["question_id"]:q["previous_r3_capacity"] for q in manifest["questions"]}
    statuses={}; records=[]
    for qid, side in expected:
        try: st, sp = status_for(qid,side)
        except FileNotFoundError: errors.append(f"missing status {qid}::{side}"); continue
        if st.get("question_id")!=qid or st.get("side")!=side: errors.append(f"status identity mismatch {qid}::{side}")
        if st.get("execution_status") not in {"normal_success","downgraded_recovery","failed"}: errors.append(f"nonterminal status {qid}::{side}")
        final=final_for(qid,side); selected=((final or {}).get("answer") or {}).get("selected_option_id")
        planner=FORMAL/"planner/cases"/qid/side/"planner.json"; ev=route_events(qid,side)
        ends=[e for e in ev if e.get("event")=="route_end"]; starts=[e for e in ev if e.get("event")=="route_start"]
        records.append({"question_id":qid,"route":side,"status":st,"status_path":sp,"planner_path":planner,"planner_present":planner.is_file(),"prediction":str(selected).upper() if selected else None,"prediction_present":selected is not None,"events":ev,"route_start_count":len(starts),"route_end_count":len(ends),"e2e_sec":ends[-1].get("e2e_sec") if ends else None,"local_attempts":local_attempts(qid,side),"capacity":capacities[qid]})
    identities=[(x["question_id"],x["route"]) for x in records]
    if identities != expected or len(set(identities))!=600: errors.append("route identities differ from manifest")
    starts=loadl(STARTS); attempts=loadl(ATTEMPTS); start_ids=[x.get("request_id") for x in starts]; end_ids=[x.get("request_id") for x in attempts]
    if Counter(start_ids)!=Counter(end_ids): errors.append("API request-start/attempt-end mismatch")
    logical=defaultdict(list)
    for a in attempts: logical[a.get("logical_call_id")].append(a)
    for key, rows in logical.items():
        if len(rows)>3: errors.append(f"retry limit exceeded {key}")
    plan_keys=[]
    for r in records:
        if r["planner_present"]:
            p=load(r["planner_path"])
            key=(p.get("question_id"),p.get("route")); plan_keys.append(key)
            if key != (r["question_id"],r["route"]): errors.append(f"planner identity mismatch {r['question_id']}::{r['route']}")
    if len(plan_keys)!=len(set(plan_keys)): errors.append("duplicate accepted Planner identity")
    cost=sum(float(a.get("estimated_cost_usd") or 0) for a in attempts)
    ledger=loadl(LEDGER); ledger_cost=sum(float(x.get("actual_cost_usd") or 0) for x in ledger if x.get("event")=="reconcile")
    if abs(cost-ledger_cost)>1e-9: errors.append("attempt/ledger cost mismatch")
    # Exhausted Planner route is a deliberate durable failed route with no planner artifact.
    failed_no_plan=[r for r in records if r["status"].get("execution_status")=="failed" and not r["planner_present"]]
    if not failed_no_plan: errors.append("failed Planner routes were silently dropped")
    structural={"status":"PASS" if not errors else "FAIL","gold_loaded":False,"errors":errors,"manifest_sha256":manifest_sha,"expected_manifest_sha256":EXPECTED_MANIFEST_SHA,"population":{"routes":len(records),"unique_route_identities":len(set(identities)),"r1":sum(r["route"]=="r1_av" for r in records),"r3":sum(r["route"]=="r3_2" for r in records)},"terminal_routes":len(records),"planner_artifacts":len(plan_keys),"failed_routes_without_planner_artifact":len(failed_no_plan),"api_request_starts":len(starts),"api_attempt_ends":len(attempts),"api_pairing_valid":Counter(start_ids)==Counter(end_ids),"attempt_cost_usd":cost,"ledger_reconciled_cost_usd":ledger_cost,"cost_reconciles":abs(cost-ledger_cost)<=1e-9,"formal_only_paths":{"formal":str(FORMAL),"gate_or_pilot_included":False},"raw_closure_before":raw_before}
    dump("structural_validation.json", structural)
    if errors: raise RuntimeError("structural audit failed; gold was not loaded: " + "; ".join(errors))
    # Hard boundary: only now may gold be read.
    gold=all_gold()
    for r in records:
        r["gold"] = gold.get(r["question_id"])
        if r["gold"] is None: raise RuntimeError(f"missing gold for {r['question_id']}")
        r["correct"] = r["prediction"] == r["gold"]
        r["execution_status"] = r["status"]["execution_status"]; r["reasoning_termination"] = r["status"].get("reasoning_termination")
    byside={s:[r for r in records if r["route"]==s] for s in SIDES}
    accuracy={"schema":"planner_api_ablation_accuracy_v1","fixed_denominator":300,"r1_av":summary(byside["r1_av"]),"r3_2":summary(byside["r3_2"])}
    completion={"definition":{"prediction_completion":"final_answer prediction present","strict_completion":"normal_success only; downgraded recovery remains separately reported"},"r1_av":summary(byside["r1_av"]),"r3_2":summary(byside["r3_2"])}
    dump("accuracy.json",accuracy); dump("completion.json",completion)
    paired=[]
    for qid in qids:
        r1=next(r for r in byside["r1_av"] if r["question_id"]==qid); r3=next(r for r in byside["r3_2"] if r["question_id"]==qid); paired.append((r1,r3))
    a=sum(x[0]["correct"] and x[1]["correct"] for x in paired); b=sum(x[0]["correct"] and not x[1]["correct"] for x in paired); c=sum(not x[0]["correct"] and x[1]["correct"] for x in paired); d=300-a-b-c
    ca=sum(x[0]["prediction_present"] and x[1]["prediction_present"] for x in paired); cb=sum(x[0]["prediction_present"] and not x[1]["prediction_present"] for x in paired); cc=sum(not x[0]["prediction_present"] and x[1]["prediction_present"] for x in paired); cd=300-ca-cb-cc
    paired_accuracy={"correctness_table":{"both_correct":a,"r1_correct_r3_wrong":b,"r1_wrong_r3_correct":c,"both_wrong":d},"r3_minus_r1_accuracy_difference":(c-b)/300,"r3_minus_r1_accuracy_percentage_points":100*(c-b)/300,"exact_two_sided_mcnemar_p":mcnemar_exact(b,c),"paired_bootstrap":bootstrap([int(y["correct"])-int(x["correct"]) for x,y in paired]),"completion_definition":"prediction present","completion_table":{"both_complete":ca,"r1_only_complete":cb,"r3_only_complete":cc,"neither_complete":cd},"r3_minus_r1_completion_difference":(cc-cb)/300,"exact_completion_mcnemar_p":mcnemar_exact(cb,cc),"completion_paired_bootstrap":bootstrap([int(y["prediction_present"])-int(x["prediction_present"]) for x,y in paired])}
    dump("paired_accuracy.json",paired_accuracy)
    # Planner accounting is actual API attempts including failed/retry calls.
    planner={}
    for side in SIDES:
        aa=[x for x in attempts if x.get("logical_call_id","").split("::")[1:2]==[side]]
        success=[x for x in aa if x.get("status")=="success"]
        planner[side]={"logical_planner_calls":len({x.get("logical_call_id") for x in aa}),"actual_api_attempts":len(aa),"retries":sum(bool(x.get("is_retry")) for x in aa),"status_counts":dict(Counter(x.get("status") for x in aa)),"provider_or_validation_failures":sum(x.get("status")!="success" for x in aa),"input_tokens":sum(int(x.get("input_tokens") or 0) for x in aa),"output_tokens":sum(int(x.get("output_tokens") or 0) for x in aa),"total_tokens":sum(int(x.get("input_tokens") or 0)+int(x.get("output_tokens") or 0) for x in aa),"total_usd":sum(float(x.get("estimated_cost_usd") or 0) for x in aa),"mean_cost_per_question":sum(float(x.get("estimated_cost_usd") or 0) for x in aa)/300,"mean_input_tokens_per_question":sum(int(x.get("input_tokens") or 0) for x in aa)/300,"mean_output_tokens_per_question":sum(int(x.get("output_tokens") or 0) for x in aa)/300,"latency_sec":dist([float(x["latency_sec"]) for x in aa if x.get("latency_sec") is not None]),"accepted_path_only":{"attempts":len(success),"total_usd":sum(float(x.get("estimated_cost_usd") or 0) for x in success)}}
    planner["r3_minus_r1"]={"input_tokens":planner["r3_2"]["input_tokens"]-planner["r1_av"]["input_tokens"],"output_tokens":planner["r3_2"]["output_tokens"]-planner["r1_av"]["output_tokens"],"total_tokens":planner["r3_2"]["total_tokens"]-planner["r1_av"]["total_tokens"],"usd":planner["r3_2"]["total_usd"]-planner["r1_av"]["total_usd"]}; planner["formal_total_usd"]=cost
    dump("planner_cost.json",planner)
    # Downstream aggregate and paired operational metrics.
    downstream={}; latency={}
    for side, rows in byside.items():
        aa=[x for r in rows for x in r["local_attempts"]]
        stages={}
        for stage in sorted({x.get("stage", "unknown") for x in aa}):
            z=[x for x in aa if x.get("stage")==stage]; stages[stage]={"attempts":len(z),"input_tokens":sum(int(x.get("input_tokens") or 0) for x in z),"output_tokens":sum(int(x.get("output_tokens") or 0) for x in z),"total_tokens":sum(int(x.get("input_tokens") or 0)+int(x.get("output_tokens") or 0) for x in z),"latency_sec":dist([float(x["latency_sec"]) for x in z if x.get("latency_sec") is not None]),"status_counts":dict(Counter(x.get("status") for x in z))}
        e2e=[float(r["e2e_sec"]) for r in rows if r["e2e_sec"] is not None]
        downstream[side]={"model_attempts_including_retries":len(aa),"stage_breakdown":stages,"input_tokens":sum(int(x.get("input_tokens") or 0) for x in aa),"output_tokens":sum(int(x.get("output_tokens") or 0) for x in aa),"total_tokens":sum(int(x.get("input_tokens") or 0)+int(x.get("output_tokens") or 0) for x in aa),"fine_physical_image_transmissions":sum(int(x.get("physical_image_transmissions") or 0) for x in aa),"downstream_model_attempt_latency_sec":dist([float(x["latency_sec"]) for x in aa if x.get("latency_sec") is not None]),"post_planner_route_e2e_sec":dist(e2e),"per_question":{"model_attempts":len(aa)/300,"fine_images":sum(int(x.get("physical_image_transmissions") or 0) for x in aa)/300,"input_tokens":sum(int(x.get("input_tokens") or 0) for x in aa)/300,"output_tokens":sum(int(x.get("output_tokens") or 0) for x in aa)/300}}
        latency[side]={"planner_latency_sec":planner[side]["latency_sec"],"post_planner_e2e_sec":dist(e2e),"total_modeled_latency_sec":dist([float(r["e2e_sec"])+next((float(a.get("latency_sec") or 0) for a in attempts if a.get("logical_call_id")==f"{r['question_id']}::{side}::planner"),0.0) for r in rows if r["e2e_sec"] is not None])}
    dump("downstream_cost.json",downstream); dump("latency.json",latency)
    total={s:{"total_model_api_requests":planner[s]["actual_api_attempts"]+downstream[s]["model_attempts_including_retries"],"total_tokens":planner[s]["total_tokens"]+downstream[s]["total_tokens"],"fine_images":downstream[s]["fine_physical_image_transmissions"],"planner_api_usd":planner[s]["total_usd"],"local_gpu_monetary_cost":"not estimated; no authoritative local GPU/electricity price recorded","modeled_latency_definition":"Planner API attempt latency plus post-Planner route E2E where route E2E exists"} for s in SIDES}
    dump("total_cost.json",total)
    # old local overflow is attribution only.
    recovery={}
    for label in ("context_overflow_pre_model","eligible_pre_model"):
        rows=[r for r in byside["r3_2"] if r["capacity"]==label]; aa=[x for r in rows for x in attempts if x.get("logical_call_id")==f"{r['question_id']}::r3_2::planner"]
        recovery[label]={"question_ids":[r["question_id"] for r in rows],"denominator":len(rows),"api_planner_attempted":len({x.get("logical_call_id") for x in aa}),"api_planner_accepted":sum(r["planner_present"] for r in rows),"planner_retries":sum(bool(x.get("is_retry")) for x in aa),"planner_failures":sum(x.get("status")!="success" for x in aa),"terminal_routes":len(rows),"predictions_present":sum(r["prediction_present"] for r in rows),"correct":sum(r["correct"] for r in rows),"completion":sum(r["prediction_present"] for r in rows),"normal_success":sum(r["execution_status"]=="normal_success" for r in rows),"downgraded_recovery":sum(r["execution_status"]=="downgraded_recovery" for r in rows),"failed":sum(r["execution_status"]=="failed" for r in rows),"failure_categories":dict(Counter(failure_category(r["status"].get("failure_reason")) for r in rows if r["execution_status"]=="failed"))}
    dump("local_overflow_recovery.json",recovery)
    failures=[]
    for r in records:
        if r["execution_status"]=="failed":
            key=f"{r['question_id']}::{r['route']}::planner"; failures.append({"question_id":r["question_id"],"route":r["route"],"stage":"planner" if failure_category(r["status"].get("failure_reason"))=="planner_provider_exhausted" else "downstream","execution_status":"failed","failure_category":failure_category(r["status"].get("failure_reason")),"failure_reason":r["status"].get("failure_reason"),"prediction_present":r["prediction_present"],"planner_attempts":len(logical.get(key,[])),"downstream_attempts":len(r["local_attempts"])})
    dump("failure_classification.json",{"count":len(failures),"by_category":dict(Counter(x["failure_category"] for x in failures)),"routes":failures})
    # Resume audit: no terminal output has duplicate lifecycle; attempt contract remains bounded.
    # A start without an end followed by a later start/end is an interrupted
    # *incomplete* route resumption, not a rerun of a terminal route.  The
    # durable terminal state is the final route_end/route_status pair.  There
    # must never be a new start after that terminal end.
    preterminal_resumptions=[]; starts_after_terminal=[]
    for r in records:
        starts=[e for e in r["events"] if e.get("event")=="route_start"]
        ends=[e for e in r["events"] if e.get("event")=="route_end"]
        if len(starts)>1 and len(ends)==1:
            preterminal_resumptions.append({"route_key":f"{r['question_id']}::{r['route']}","start_count":len(starts),"end_count":len(ends),"terminal_route_uuid":ends[0].get("route_uuid")})
        if ends:
            terminal_time=max(str(e.get("written_at_utc") or "") for e in ends)
            if any(str(e.get("written_at_utc") or "") > terminal_time for e in starts):
                starts_after_terminal.append(f"{r['question_id']}::{r['route']}")
    exhausted=[k for k,v in logical.items() if len(v)==3 and all(x.get("status")!="success" for x in v)]
    resume={"status":"PASS" if not starts_after_terminal else "FAIL","operational_interruptions":"cream tmux/tunnel interruptions are operational and are not classified as route scientific failures absent a route artifact.","terminal_routes":600,"terminal_route_rerun_evidence":{"route_starts_after_a_route_end":starts_after_terminal,"result":"no evidence of terminal-route rerun" if not starts_after_terminal else "requires review"},"preterminal_interrupted_route_resumptions":preterminal_resumptions,"accepted_planner_artifact_regeneration":{"unique_artifact_identities":len(set(plan_keys)),"artifact_count":len(plan_keys),"result":"no duplicate artifact identity"},"retry_limit":{"logical_calls_with_more_than_3_attempts":[k for k,v in logical.items() if len(v)>3],"exhausted_logical_calls":exhausted},"terminal_without_final_answer_skipped":sum(r["execution_status"]=="failed" and not r["prediction_present"] for r in records),"costs_counted_once":{"attempt_cost_usd":cost,"ledger_cost_usd":ledger_cost,"valid":abs(cost-ledger_cost)<=1e-9}}
    dump("resume_audit.json",resume)
    # paired efficiency only uses questions where the needed metric exists in both sides; no zero imputation.
    pe={}
    metrics={"planner_input_tokens":lambda x: sum(int(a.get("input_tokens") or 0) for a in attempts if a.get("logical_call_id")==f"{x['question_id']}::{x['route']}::planner"),"planner_cost_usd":lambda x: sum(float(a.get("estimated_cost_usd") or 0) for a in attempts if a.get("logical_call_id")==f"{x['question_id']}::{x['route']}::planner"),"post_planner_e2e_sec":lambda x:x["e2e_sec"],"fine_images":lambda x:sum(int(a.get("physical_image_transmissions") or 0) for a in x["local_attempts"]),"downstream_calls":lambda x:len(x["local_attempts"])}
    for name, fn in metrics.items():
        vals=[(fn(x),fn(y)) for x,y in paired]; vals=[(float(x),float(y)) for x,y in vals if x is not None and y is not None]
        ds=[y-x for x,y in vals]; pe[name]={"paired_count":len(vals),"r1":dist([x for x,_ in vals]),"r3":dist([y for _,y in vals]),"r3_minus_r1":dist(ds),"bootstrap":bootstrap(ds)}
    dump("paired_efficiency.json",pe)
    canonical={"schema":"hourvideo_v6_6_2_planner_api_ablation_formal_canonical_summary_v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"formal_manifest_sha256":manifest_sha,"raw_closure_before":raw_before,"analysis_scope":"formal namespace only; gate/pilot excluded","gold_loading_boundary":"gold loaded only after structural_validation PASS","api_calls_during_aggregation":0,"raw_artifacts_modified":False,"output_files":["structural_validation.json","accuracy.json","completion.json","paired_accuracy.json","local_overflow_recovery.json","planner_cost.json","downstream_cost.json","total_cost.json","latency.json","failure_classification.json","resume_audit.json","paired_efficiency.json","thesis_data_report.md"]}
    dump("canonical_manifest.json",canonical)
    md=["# Planner-only API ablation — formal Eval300 canonical report", "", "All values are derived read-only from the formal namespace. Gate/pilot artifacts and costs are excluded.", "", "## Structural audit", f"- Formal manifest SHA-256: `{manifest_sha}`", f"- Raw closure SHA-256 (before/after): `{raw_before['tree_sha256']}` / pending", f"- 600 unique routes: 300 R1 and 300 R3; API starts/ends: {len(starts)}/{len(attempts)}; formal API spend: ${cost:.6f}.", "", "## Accuracy and completion", f"- R1: {accuracy['r1_av']['correct']}/300 ({accuracy['r1_av']['accuracy_percent']:.2f}%), predictions {accuracy['r1_av']['predictions_present']}/300, strict normal-success {accuracy['r1_av']['strict_completion']}/300.", f"- R3: {accuracy['r3_2']['correct']}/300 ({accuracy['r3_2']['accuracy_percent']:.2f}%), predictions {accuracy['r3_2']['predictions_present']}/300, strict normal-success {accuracy['r3_2']['strict_completion']}/300.", "", "## Paired comparison", f"- Correctness table: both correct {a}; R1-only {b}; R3-only {c}; both wrong {d}.", f"- R3 − R1 accuracy: {100*(c-b)/300:.2f} pp; exact two-sided McNemar p={paired_accuracy['exact_two_sided_mcnemar_p']:.6g}.", "", "## Capacity attribution", f"- Old local-overflow R3 subset: {recovery['context_overflow_pre_model']['api_planner_accepted']}/150 accepted Planner, {recovery['context_overflow_pre_model']['terminal_routes']}/150 terminal, {recovery['context_overflow_pre_model']['predictions_present']}/150 predictions.", "", "## Cost", f"- Planner API: R1 ${planner['r1_av']['total_usd']:.6f}, R3 ${planner['r3_2']['total_usd']:.6f}, total ${cost:.6f}. Local GPU monetary cost is not estimated.", "", "## Failures", f"- {len(failures)} durable failed routes; detailed artifact: `failure_classification.json`."]
    (OUT/"thesis_data_report.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    raw_after=freeze_raw()
    if raw_after["tree_sha256"] != raw_before["tree_sha256"]: raise RuntimeError("immutable raw closure changed during aggregation")
    canonical["raw_closure_after"]=raw_after; dump("canonical_manifest.json",canonical)
    # Update report only after proven equality (the report itself remains outside raw closure).
    report=(OUT/"thesis_data_report.md").read_text(); report=report.replace("/ pending", f"/ `{raw_after['tree_sha256']}`")
    (OUT/"thesis_data_report.md").write_text(report)


if __name__ == "__main__": main()
