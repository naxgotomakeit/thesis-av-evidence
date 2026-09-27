from __future__ import annotations

import hashlib, json
from pathlib import Path
from typing import Any

from experiments.hourvideo_v6_6_2_local_context_limited_eval300_v1.prompt_contract import PLANNER_R1_SYSTEM, PLANNER_R3_SYSTEM
from experiments.hourvideo_v6_6_2_local_context_limited_eval300_v1 import formal_runtime as frozen
from . import formal_runtime as gate
from .anthropic_transport import AnthropicPlannerTransport
from .budget import BudgetGuard
from .completion import wait_for_terminal_route
from .pricing import PricingCatalog
from .providers import PlannerProviderAdapter, ProviderRequest
from .providers import ProviderCallFailed
from .telemetry import AttemptTelemetryJournal

ROUTES=("r1_av","r3_2")

def _digest(v: Any) -> str: return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def route_is_terminal(output_root: Path, question_id: str, route: str) -> bool:
    """Execution resume rule: terminal status, not prediction presence, is authoritative."""
    path=output_root/'live'/'cases'/question_id/route/'route_status.json'
    if not path.is_file(): return False
    try: row=json.loads(path.read_text())
    except (OSError,json.JSONDecodeError): return False
    return row.get('question_id')==question_id and row.get('side')==route and row.get('execution_status') in {'normal_success','downgraded_recovery','failed'}

def routes_to_execute(output_root: Path, question_id: str) -> tuple[str,...]:
    return tuple(route for route in ROUTES if not route_is_terminal(output_root,question_id,route))

def planner_is_exhausted(telemetry_root: Path, logical_call_id: str) -> bool:
    path=telemetry_root/'api_attempts.jsonl'
    if not path.is_file(): return False
    rows=[json.loads(x) for x in path.read_text().splitlines() if x]
    rows=[x for x in rows if x.get('logical_call_id')==logical_call_id]
    return len(rows)>=3 and not any(x.get('status')=='success' for x in rows)

def materialize_planner_exhausted(output_root: Path, question_id: str, route: str, logical_call_id: str) -> None:
    path=output_root/'live'/'cases'/question_id/route/'route_status.json'
    if path.is_file(): return
    gate.write_json(path,{'video_uid':question_id.rsplit('_',2)[0],'question_id':question_id,'side':route,'execution_status':'failed','reasoning_termination':None,'failure_reason':'planner_provider_exhausted','terminal':True,'planner_attempts_exhausted':True,'planner_logical_call_id':logical_call_id,'prediction_present':False,'written_at_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()})

def load_manifest(root: Path, config: dict[str,Any]) -> tuple[dict[str,Any],str]:
    p=root/config["pilot_manifest"]; raw=p.read_bytes(); m=json.loads(raw)
    qs=m.get("questions",[])
    expected=int(config.get("expected_questions",5))
    if len(qs)!=expected or m.get("routes")!=list(ROUTES) or len({q["question_id"] for q in qs})!=expected: raise RuntimeError("invalid paired/formal manifest")
    if expected==5 and ([q["previous_r3_capacity"] for q in qs].count("context_overflow_pre_model")!=3 or [q["previous_r3_capacity"] for q in qs].count("eligible_pre_model")!=2): raise RuntimeError("manifest capacity composition invalid")
    return m,hashlib.sha256(raw).hexdigest()

def _request(root:Path, base:dict[str,Any], qid:str, route:str)->tuple[ProviderRequest,dict[str,Any]]:
    source=root/base["frozen_inputs"]["source_experiment"]; question=gate.load_json(source/"cases"/qid/"question_input.json"); case=gate.load_json(source/"case_configs"/(qid+".json")); mp=Path(case["asset_paths"]["r1_av_navigation_map.json" if route=="r1_av" else "r3_2_navigation_map.json"]); doc=gate.load_json(mp); ids=[x["coarse_id"] for x in doc["coarse_regions"]]; reqs=gate.v661.option_requirements(question); canonical={"question":question,"requirements":reqs,"navigation_map":gate.v661._base._planner_map(doc)}; payload,pids=gate.v661._base._local_indexed_planner_contract(canonical,ids); schema=frozen._selected_schema(question,reqs,pids); prof=gate._planner_profile(base); system=PLANNER_R1_SYSTEM if route=="r1_av" else PLANNER_R3_SYSTEM
    return ProviderRequest(logical_call_id=f"{qid}::{route}::planner",role="planner",provider=prof["provider"],model=prof["model"],system_prompt=system,payload=payload,schema=schema,max_output_tokens=8192,estimated_input_tokens=200000,temperature=0.0,retry_feedback_instruction="Regenerate the complete JSON for the same question and unchanged navigation map; fix only this contract error."), {"question":question,"requirements":reqs,"coarse_ids":ids,"map":mp,"payload_sha":_digest(payload),"prompt_sha":hashlib.sha256(system.encode()).hexdigest()}

def run(root:Path, config_path:Path)->dict[str,Any]:
    cfg=gate.load_json(config_path); base_path=root/cfg["base_config"]; base=gate.load_json(base_path); gate.validate_phase2_config(root,base_path); manifest,msha=load_manifest(root,cfg); out=root/cfg["output_root"]; (out/"manifests").mkdir(parents=True,exist_ok=True); gate.write_json(out/"manifests"/"pilot_manifest.json",{"manifest":manifest,"sha256":msha,"route_count":10})
    profile=gate._planner_profile(base); pricing=PricingCatalog.load(root/base["pricing_catalog"]); adapter=PlannerProviderAdapter(transports={"anthropic":AnthropicPlannerTransport(timeout_sec=profile["timeout_sec"])},credential_env_by_provider={"anthropic":profile["credential_env"]},pricing=pricing,telemetry=AttemptTelemetryJournal(out/"telemetry"),budget=BudgetGuard(out/"telemetry"/"budget_ledger.jsonl",cfg["api_budget"]["max_cost_usd"],True))
    for q in manifest["questions"]:
      qid=q["question_id"]
      active=routes_to_execute(out,qid)
      for route in active:
       req,ctx=_request(root,base,qid,route); dest=out/"planner"/"cases"/qid/route/"planner.json"
       if planner_is_exhausted(out/'telemetry',req.logical_call_id):
        materialize_planner_exhausted(out,qid,route,req.logical_call_id); continue
       if not dest.is_file():
        try: result=adapter.call(req,max_attempts=3,schema_validator=lambda raw,s: frozen._validate_and_project_selected(raw,ctx["question"],ctx["requirements"],ctx["coarse_ids"]))
        except ProviderCallFailed:
         materialize_planner_exhausted(out,qid,route,req.logical_call_id); continue
        plan=frozen._validate_and_project_selected(result.parsed_output,ctx["question"],ctx["requirements"],ctx["coarse_ids"])
        gate.write_json(dest,{"schema_version":"v6_6_2_planner_api_ablation_frozen_planner_v1","question_id":qid,"video_uid":qid.rsplit("_",2)[0],"route":route,"input_asset_path":str(ctx["map"]),"input_asset_sha256":gate.sha256_file(ctx["map"]),"input_payload_sha256":ctx["payload_sha"],"planner_model":req.model,"provider":req.provider,"temperature":0.0,"max_tokens":8192,"prompt_sha256":ctx["prompt_sha"],"output":plan,"usage":{"input_tokens":result.input_tokens,"output_tokens":result.output_tokens,"latency_sec":result.latency_sec,"estimated_cost_usd":str(result.estimated_cost_usd),"attempt_index":result.attempt_index},"gold_loaded":False})
      active=routes_to_execute(out,qid)
      if active:
       downstream=gate.load_json(root/base["frozen_baseline_config"]); downstream.update({"experiment":cfg["experiment"],"planner_source_experiment":str((out/"planner").relative_to(root)),"output_root":str((out/"live").relative_to(root))}); dcfg=out/"manifests"/"downstream.json"; gate.write_json(dcfg,downstream)
       oldpaths,oldsides=gate.v661._case_paths,gate.v661._configured_sides; gate.v661._case_paths=lambda c,r,f:[x for x in oldpaths(c,r,None) if x[0]["question_id"]==qid]; gate.v661._configured_sides=lambda c:active
       try: gate.v662.run_live(root,dcfg,video_uid=qid)
       finally: gate.v661._case_paths,gate.v661._configured_sides=oldpaths,oldsides
       for route in active: wait_for_terminal_route(out/"live",qid,route)
    return {"status":"RUN_COMPLETE","manifest_sha256":msha,"routes":len(manifest["questions"])*2}
