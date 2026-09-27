#!/usr/bin/env python3
"""Approved one-incident recovery: reference 27 v8 routes, run remaining 148."""
from __future__ import annotations
import argparse, hashlib, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("${PROJECT_MSC_ROOT}/direct_r1_visual_only_correction_v8_formal_candidate")
SOURCE_NS = SOURCE / "runs/direct_r1_visual_only_correction_v8_formal_175"
SOURCE_MANIFEST = SOURCE_NS / "formal_manifest_visual_only_correction_v8.json"
SOURCE_MANIFEST_SHA = "d556d817cc0ba6b49f0dfa7b6e4cf291387f9a5f55aa86ec02d7cb0b80517c61"
SOURCE_LEDGER = SOURCE_NS / "budget_ledger.json"
SOURCE_LEDGER_SHA = "c86f944e666f5699493df912a249ceb54179ebd28d07efa82c9ad20a6dcb927c"
SOURCE_STOP_REPORT = SOURCE / "R1_DIRECT_VISUAL_ONLY_V8_FORMAL_STOP_STATUS.md"
SOURCE_STOP_REPORT_SHA = "1b2a65918943108c448791b1e25f9625ca2e6e1bd912e02939bf639f6de3c37a"
INCIDENT_ROUTE = "R1VO8:4572b198-2c1c-4920-bcf0-95fcebe12261_4_23"
INCIDENT_ATTEMPT = INCIDENT_ROUTE + ":attempt:2"
EXPERIMENT = "direct_r1_visual_only_correction_v9_recovery_remaining_148"
NAMESPACE = ROOT / "runs" / EXPERIMENT
MANIFEST_PATH = NAMESPACE / "recovery_manifest_v9.json"
LOCK_PATH = NAMESPACE / "recovery_launch_lock_v9.json"
RECOVERY_RECORD = ROOT / "recovery/route27_qa_success_with_unknown_cost.json"
TOTAL_CAP = 15.0

sys.path.insert(0, str(ROOT / "src"))
from direct_api_v1.anthropic_provider import AnthropicDirectAgent
from direct_api_v1.formal_runtime import FormalOrchestrator
from direct_api_v1.frame_resolver import FrozenFrameResolver
from direct_api_v1.maps import FORBIDDEN_QUESTION_KEYS

def sha(p: Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p: Path): return json.loads(p.read_text(encoding="utf-8"))
def rows(p: Path): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def stable(path: Path, value):
    payload=json.dumps(value,indent=2,ensure_ascii=False)+"\n"
    if path.exists():
        if path.read_text()!=payload: raise RuntimeError(f"frozen asset drift: {path}")
    else:
        path.parent.mkdir(parents=True,exist_ok=True); path.write_text(payload)

def verify_incident():
    artifact_path=SOURCE_NS/"route_artifacts"/f"{INCIDENT_ROUTE}.json"
    status_path=SOURCE_NS/"route_status"/f"{INCIDENT_ROUTE}.json"
    artifact,status,ledger=load(artifact_path),load(status_path),load(SOURCE_LEDGER)
    starts=[x for x in rows(SOURCE_NS/"journals/request_start.jsonl") if x["route_id"]==INCIDENT_ROUTE]
    ends=[x for x in rows(SOURCE_NS/"journals/attempt_end.jsonl") if x["route_id"]==INCIDENT_ROUTE]
    controls=[x for x in rows(SOURCE_NS/"journals/controller_result.jsonl") if x["route_id"]==INCIDENT_ROUTE]
    if len(starts)!=8 or len(ends)!=8 or len(controls)!=7: raise RuntimeError("incident journal cardinality mismatch")
    by_id={x["attempt_id"]:x for x in ends}
    if set(by_id)!=set(x["attempt_id"] for x in starts): raise RuntimeError("incident attempt identity mismatch")
    unknown=by_id[INCIDENT_ATTEMPT]
    if unknown.get("response_received") or unknown.get("http_status") is not None: raise RuntimeError("unknown attempt misclassified")
    if unknown.get("response_metadata",{}).get("failure_stage")!="send_or_wait_for_response": raise RuntimeError("incident failure-stage drift")
    received=[x for x in ends if x.get("response_received")]
    if len(received)!=7: raise RuntimeError("expected seven durable responses")
    for start in starts:
        digest=hashlib.sha256(start["attempt_id"].encode()).hexdigest()
        if not (SOURCE_NS/"request_payloads"/f"{digest}.json").is_file(): raise RuntimeError("missing request payload")
        response=SOURCE_NS/"provider_responses"/f"{digest}.json"
        if (start["attempt_id"]!=INCIDENT_ATTEMPT) != response.is_file(): raise RuntimeError("response persistence mismatch")
    if artifact.get("terminal_status")!="final_answer" or artifact.get("final_prediction") not in "ABCDE": raise RuntimeError("incident answer/parser incomplete")
    if artifact.get("rounds")!=7 or artifact.get("total_api_attempts")!=8 or len(artifact.get("provider_attempts",[]))!=8: raise RuntimeError("incident artifact incomplete")
    if len(artifact.get("turns",[]))!=7 or artifact["turns"][-1].get("action_type")!="final_answer": raise RuntimeError("final parsed turn missing")
    frames=[f for turn in artifact["turns"] for f in turn.get("resolved_frames",[]) if not f.get("duplicate_of_seen_frame")]
    if artifact.get("unique_images_transmitted")!=16 or len(frames)!=16: raise RuntimeError("incident frame history incomplete")
    if any(f["expected_sha256"]!=f["observed_sha256"] or sha(Path(f["frame_path"]))!=f["expected_sha256"] for f in frames): raise RuntimeError("incident frame SHA mismatch")
    if abs(sum(float(x.get("cache_aware_usd") or 0) for x in ends)-float(artifact["total_usd"]))>1e-12: raise RuntimeError("incident known-cost mismatch")
    liability=ledger["attempts"].get(INCIDENT_ATTEMPT,{})
    if liability.get("state")!="uncertain_liability" or abs(float(liability.get("reserved_upper_bound_usd",0))-.25256)>1e-12: raise RuntimeError("incident liability missing")
    if status.get("state")!="blocked_unknown_provider_outcome" or status.get("unresolved_attempt_ids")!=[INCIDENT_ATTEMPT]: raise RuntimeError("incident blocked status drift")
    return {
      "schema_version":"route27_offline_qa_recovery_v1","gold_read":False,"provider_sends":0,
      "route_id":INCIDENT_ROUTE,"question_id":artifact["question_id"],
      "qa_result":"success_complete","cost_result":"one_provider_attempt_unknown",
      "source_artifact":str(artifact_path),"source_artifact_sha256":sha(artifact_path),
      "source_status":str(status_path),"source_status_sha256":sha(status_path),
      "request_count":8,"durable_response_count":7,"controller_turns":7,"unique_images":16,
      "parser_terminal":"final_answer","prediction_present":True,
      "unknown_attempt_id":INCIDENT_ATTEMPT,"unknown_reserved_usd":.25256,
      "original_blocked_state_preserved":True,"artifact_overwritten":False,
    }

def materialize():
    for p,d in ((SOURCE_MANIFEST,SOURCE_MANIFEST_SHA),(SOURCE_LEDGER,SOURCE_LEDGER_SHA),(SOURCE_STOP_REPORT,SOURCE_STOP_REPORT_SHA)):
        if sha(p)!=d: raise RuntimeError(f"source SHA drift: {p}")
    for p in (ROOT/"src/direct_api_v1").glob("*.py"):
        if sha(p)!=sha(SOURCE/p.relative_to(ROOT)): raise RuntimeError(f"QA code differs from v8: {p}")
    incident=verify_incident(); stable(RECOVERY_RECORD,incident)
    source=load(SOURCE_MANIFEST); statuses={p.stem:load(p) for p in (SOURCE_NS/"route_status").glob("*.json")}
    ordered=source["routes"]
    completed=ordered[:26]; blocked=ordered[26]; remaining=ordered[27:]
    if blocked["route_id"]!=INCIDENT_ROUTE or len(remaining)!=148: raise RuntimeError("v8 stop boundary/order mismatch")
    if any(statuses.get(x["route_id"],{}).get("state")!="terminal_success" for x in completed): raise RuntimeError("first 26 not terminal success")
    if any(x["route_id"] in statuses for x in remaining): raise RuntimeError("remaining route already started")
    refs=[]
    for route in completed:
        ap=SOURCE_NS/"route_artifacts"/f"{route['route_id']}.json"; sp=SOURCE_NS/"route_status"/f"{route['route_id']}.json"
        refs.append({"question_id":route["question_id"],"source_kind":"v8_terminal_success","source_route_id":route["route_id"],"source_artifact":str(ap),"source_artifact_sha256":sha(ap),"source_status":str(sp),"source_status_sha256":sha(sp)})
    ap=Path(incident["source_artifact"])
    refs.append({"question_id":blocked["question_id"],"source_kind":"v8_offline_recovered_qa_success_unknown_cost","source_route_id":blocked["route_id"],"source_artifact":str(ap),"source_artifact_sha256":sha(ap),"recovery_record":str(RECOVERY_RECORD),"recovery_record_sha256":sha(RECOVERY_RECORD)})
    videos=[]
    for old in source["videos"]:
        vid=old["video_id"]; v=dict(old)
        v["r1_visual_only_map"]=str(ROOT/f"maps/r1_visual_only/{vid}/r1_visual_only_navigation_map.json")
        v["frame_sha_manifest"]=str(ROOT/f"manifests/frame_sha256_current_snapshot/{vid}.json")
        if sha(Path(v["r1_visual_only_map"]))!=v["r1_visual_only_sha256"] or sha(Path(v["frame_sha_manifest"]))!=v["frame_sha_manifest_sha256"]: raise RuntimeError("map/frame manifest drift")
        videos.append(v)
    vb={x["video_id"]:x for x in videos}; routes=[]
    for old in remaining:
        r=dict(old); r["route_id"]="R1VO9:"+r["question_id"]; r["map_path"]=vb[r["video_id"]]["r1_visual_only_map"]; routes.append(r)
    config=load(ROOT/"config/direct_v1_anthropic_smoke.json"); cred=Path(config["credential_env_path"])
    manifest={"schema_version":"r1_visual_only_v9_network_recovery_candidate_v1","experiment_id":EXPERIMENT,"gold_loaded":False,"api_calls_at_freeze":0,"total_eval_routes":175,"referenced_completed_routes":26,"offline_recovered_routes":1,"new_routes":148,"concurrency":1,"hard_budget_usd":15.0,"source_v8_manifest":str(SOURCE_MANIFEST),"source_v8_manifest_sha256":SOURCE_MANIFEST_SHA,"source_v8_ledger":str(SOURCE_LEDGER),"source_v8_ledger_sha256":SOURCE_LEDGER_SHA,"approved_bypass_attempt_id":INCIDENT_ATTEMPT,"approved_bypass_policy":"retain_full_reservation_no_resend_only_this_attempt","credential_file_path":str(cred),"credential_file_sha256":sha(cred),"credential_resolution":"file_only_sha_locked_no_environment_override","required_transport":"urllib_fallback","protocol_identity":source["protocol_identity"],"completed_references":refs,"routes":routes,"videos":videos}
    return manifest

def create_namespace(manifest):
    if NAMESPACE.exists(): return
    NAMESPACE.mkdir(parents=True)
    for d in ("inputs","journals","route_status","route_artifacts","route_checkpoints","provider_responses","request_payloads","failure_diagnostics"): (NAMESPACE/d).mkdir()
    for f in ("request_start.jsonl","attempt_end.jsonl","controller_result.jsonl","lifecycle.jsonl"): (NAMESPACE/"journals"/f).write_bytes(b"")
    source_inputs=SOURCE_NS/"inputs"
    for r in manifest["routes"]:
        p=source_inputs/f"{r['question_id']}.json"
        if sha(p)!=r["question_sha256"] or set(load(p))&FORBIDDEN_QUESTION_KEYS: raise RuntimeError("remaining input drift/gold")
        shutil.copy2(p,NAMESPACE/"inputs"/p.name)
    stable(MANIFEST_PATH,manifest)
    old=load(SOURCE_LEDGER); attempts={}
    for aid,row in old["attempts"].items():
        x={**row,"inherited_from_v8_ledger":str(SOURCE_LEDGER),"source_ledger_sha256":SOURCE_LEDGER_SHA}
        if aid==INCIDENT_ATTEMPT: x["state"]="inherited_approved_unknown_liability"
        elif row["state"] in ("uncertain_liability","inherited_uncertain_liability"): x["state"]="inherited_uncertain_liability"
        else: x["state"]="inherited_settled_attempt"
        attempts[aid]=x
    stable(NAMESPACE/"budget_ledger.json",{"schema_version":"reserved_budget_ledger_v2","cap_usd":15.0,"spent_usd":old["spent_usd"],"reserved_usd":old["reserved_usd"],"attempt_ids":old["attempt_ids"],"attempts":attempts,"charges":[{**x,"inherited_from_v8_ledger":str(SOURCE_LEDGER)} for x in old["charges"]],"inherited_accounting":{"source_ledger":str(SOURCE_LEDGER),"source_ledger_sha256":SOURCE_LEDGER_SHA,"approved_unknown_attempt":INCIDENT_ATTEMPT}})
    stable(LOCK_PATH,{"schema_version":"r1_visual_only_v9_recovery_launch_lock_v1","experiment_id":EXPERIMENT,"manifest_path":str(MANIFEST_PATH),"manifest_sha256":sha(MANIFEST_PATH),"runner_sha256":sha(Path(__file__)),"route_count":148,"total_coverage":175,"hard_budget_usd":15.0,"spent_usd_at_freeze":1.6287881,"reserved_usd_at_freeze":1.51536,"available_usd_at_freeze":11.8558519,"approved_unknown_attempt":INCIDENT_ATTEMPT,"required_transport":"urllib_fallback","credential_file_sha256":manifest["credential_file_sha256"],"real_execution_allowed":True})

def prepare():
    manifest=materialize(); create_namespace(manifest)
    if load(MANIFEST_PATH)!=manifest: raise RuntimeError("recovery manifest drift")
    lock=load(LOCK_PATH)
    if lock["manifest_sha256"]!=sha(MANIFEST_PATH) or lock["runner_sha256"]!=sha(Path(__file__)): raise RuntimeError("recovery lock drift")
    if sha(Path(manifest["credential_file_path"]))!=manifest["credential_file_sha256"]: raise RuntimeError("credential identity drift")
    ledger=load(NAMESPACE/"budget_ledger.json"); available=ledger["cap_usd"]-ledger["spent_usd"]-ledger["reserved_usd"]
    if abs(available-11.8558519)>1e-9: raise RuntimeError("inherited budget drift")
    return manifest,ledger

def build(manifest,agent_class=AnthropicDirectAgent,orchestrator_class=FormalOrchestrator):
    config=load(ROOT/"config/direct_v1_anthropic_smoke.json"); vb={x["video_id"]:x for x in manifest["videos"]}
    def resolver(route):
        v=vb[route["video_id"]]; return FrozenFrameResolver.from_hierarchy(load(Path(v["hierarchy"])),load(Path(v["frame_sha_manifest"])))
    def agent(route,lifecycle,state):
        ledger=load(NAMESPACE/"budget_ledger.json"); available=ledger["cap_usd"]-ledger["spent_usd"]-ledger["reserved_usd"]
        return agent_class(credential_env_path=Path(config["credential_env_path"]),model=config["model"],pricing=config["anthropic_cache_aware_pricing_usd_per_million_tokens"],max_output_tokens=config["max_output_tokens"],timeout_sec=config["timeout_sec"],max_retries=config["max_retries"],max_total_usd=available,max_billable_input_tokens=200000,lifecycle=lifecycle,transport_mode="urllib_fallback",credential_file_sha256=manifest["credential_file_sha256"],require_file_credential=True)
    return orchestrator_class(manifest_path=MANIFEST_PATH,namespace=NAMESPACE,agent_factory=agent,resolver_factory=resolver,cap_usd=15.0)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--execute-real-recovery",action="store_true"); p.add_argument("--confirm-remaining-routes",type=int); p.add_argument("--confirm-total-budget-usd",type=float); a=p.parse_args(); m,l=prepare()
    if not a.execute_real_recovery:
        print(json.dumps({"valid":True,"mode":"offline_no_api","real_api_calls":0,"references":27,"remaining_routes":148,"spent":l["spent_usd"],"reserved":l["reserved_usd"],"available":l["cap_usd"]-l["spent_usd"]-l["reserved_usd"],"namespace":str(NAMESPACE)},indent=2)); return
    if a.confirm_remaining_routes!=148 or a.confirm_total_budget_usd!=15.0: raise SystemExit("exact 148/$15 confirmation required")
    print(json.dumps(build(m).execute(),indent=2))
if __name__=="__main__": main()
