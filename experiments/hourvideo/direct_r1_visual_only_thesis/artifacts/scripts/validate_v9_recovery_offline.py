#!/usr/bin/env python3
"""Offline-only delta validation for the single-incident v9 recovery."""
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
import run_direct_r1_visual_only_correction_v9_recovery as runner
from direct_api_v1.formal_runtime import FormalStore,NamespaceWriterLock,NamespaceLockError

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text())
class Capture:
    kwargs=None
    def __init__(self,**kwargs): Capture.kwargs=kwargs

def main():
    m,l=runner.prepare(); source=load(runner.SOURCE_MANIFEST); recovery=load(runner.RECOVERY_RECORD)
    source_q=[x["question_id"] for x in source["routes"]]; ref_q=[x["question_id"] for x in m["completed_references"]]; new_q=[x["question_id"] for x in m["routes"]]
    code_equal=all(sha(p)==sha(runner.SOURCE/p.relative_to(ROOT)) for p in (ROOT/"src/direct_api_v1").glob("*.py"))
    maps_ok=all(sha(v["r1_visual_only_map"])==v["r1_visual_only_sha256"] and all(x.get("audio_channel")==[] for x in load(v["r1_visual_only_map"])["coarse_regions"]) for v in m["videos"])
    frame_manifests_ok=all(sha(v["frame_sha_manifest"])==v["frame_sha_manifest_sha256"] for v in m["videos"])
    inputs_ok=all(sha(runner.NAMESPACE/"inputs"/f"{r['question_id']}.json")==r["question_sha256"] for r in m["routes"])
    clean={d:len(list((runner.NAMESPACE/d).glob("*.json"))) for d in ("route_status","route_artifacts","request_payloads","provider_responses")}
    journals=sum(len(p.read_text().splitlines()) for p in (runner.NAMESPACE/"journals").glob("*.jsonl"))
    with tempfile.TemporaryDirectory(prefix="r1vo9-recovery-test-") as td:
        td=Path(td); rejected=False
        with NamespaceWriterLock(td):
            try:
                with NamespaceWriterLock(td): pass
            except NamespaceLockError: rejected=True
        store=FormalStore(td,"offline-test",15.0); store.initialise(); store.reserve_attempt("new-unknown",.25,route_id="future"); store.mark_uncertain("new-unknown","offline_injected")
        new_unknown_stops=bool(store.uncertain_liabilities())
    orch=runner.build(m,agent_class=Capture); orch.agent_factory(m["routes"][0],object(),object()); kw=Capture.kwargs
    inherited_states={}
    for x in l["attempts"].values(): inherited_states[x["state"]]=inherited_states.get(x["state"],0)+1
    available=l["cap_usd"]-l["spent_usd"]-l["reserved_usd"]
    checks={
      "route27_QA_complete":recovery["qa_result"]=="success_complete" and recovery["request_count"]==8 and recovery["durable_response_count"]==7 and recovery["controller_turns"]==7 and recovery["unique_images"]==16 and recovery["parser_terminal"]=="final_answer",
      "route27_unknown_cost_preserved":recovery["cost_result"]=="one_provider_attempt_unknown" and recovery["unknown_attempt_id"]==runner.INCIDENT_ATTEMPT and recovery["unknown_reserved_usd"]==.25256,
      "route27_recovery_zero_sends":recovery["provider_sends"]==0 and runner.INCIDENT_ROUTE not in {x["route_id"] for x in m["routes"]},
      "first27_not_rerun":len(ref_q)==27 and new_q==source_q[27:] and not set(ref_q)&set(new_q),
      "full_175_order_and_uniqueness":ref_q+new_q==source_q and len(set(ref_q+new_q))==175,
      "remaining_exact_148":len(new_q)==148,
      "QA_code_unchanged":code_equal,
      "normal_QA_parameters_unchanged":kw["model"]=="claude-haiku-4-5-20251001" and kw["max_output_tokens"]==512 and kw["timeout_sec"]==120 and kw["max_retries"]==1 and kw["transport_mode"]=="urllib_fallback" and kw["require_file_credential"] is True and m["protocol_identity"]["max_new_images_per_turn"]==3 and m["protocol_identity"]["max_unique_images_per_question"]==16 and m["protocol_identity"]["max_route_turns"]==32,
      "normal_payload_inputs_equal_v8":inputs_ok and maps_ok and frame_manifests_ok,
      "credential_locked":sha(m["credential_file_path"])==m["credential_file_sha256"] and kw["credential_file_sha256"]==m["credential_file_sha256"],
      "budget_exact":abs(l["spent_usd"]-1.6287881)<1e-9 and abs(l["reserved_usd"]-1.51536)<1e-9 and abs(available-11.8558519)<1e-9 and abs(kw["max_total_usd"]-available)<1e-9,
      "specific_bypass_only":l["attempts"][runner.INCIDENT_ATTEMPT]["state"]=="inherited_approved_unknown_liability" and inherited_states=={"inherited_uncertain_liability":5,"inherited_settled_attempt":194,"inherited_approved_unknown_liability":1},
      "new_unknown_still_stops":new_unknown_stops,
      "writer_lock_effective":rejected,
      "new_namespace_clean":all(v==0 for v in clean.values()) and journals==0,
      "manifest_runner_locked":load(runner.LOCK_PATH)["manifest_sha256"]==sha(runner.MANIFEST_PATH) and load(runner.LOCK_PATH)["runner_sha256"]==sha(runner.__file__),
    }
    out={"schema_version":"r1_visual_only_v9_recovery_offline_validation_v1","real_api_calls":0,"gold_read":False,"checks":checks,"route27_recovery":recovery,"coverage":{"references":27,"new":148,"total":175},"namespace_counts":{**clean,"journal_rows":journals},"budget":{"cap":l["cap_usd"],"spent":l["spent_usd"],"reserved":l["reserved_usd"],"available":available,"states":inherited_states},"agent":{"model":kw["model"],"max_output_tokens":kw["max_output_tokens"],"timeout_sec":kw["timeout_sec"],"max_retries":kw["max_retries"],"transport":kw["transport_mode"],"max_total_usd":kw["max_total_usd"]},"pass":all(checks.values()),"acceptance":"PASS" if all(checks.values()) else "FAIL"}
    p=ROOT/"validation/v9_recovery_offline_validation.json"; p.write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n"); print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
