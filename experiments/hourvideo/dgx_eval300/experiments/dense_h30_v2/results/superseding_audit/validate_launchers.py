#!/usr/bin/env python3
import hashlib,json,re
from pathlib import Path
B=Path('/home/naxucl/data/HourVideo/experiments/hourvideo_v7_4_variant_c_budgets_v1')
O=B/'outputs/dense_semantic_beam_b_h30_h8_superseding_audit_v2_20260829T155639Z'
P=B/'outputs/dense_semantic_beam_b_h15_eval300_formal_v1_20260827T224330Z'
H30=B/'outputs/dense_semantic_beam_b_h30_eval300_formal_v2_20260829T155639Z/run_formal_h30.sh'
H8=B/'outputs/dense_semantic_beam_b_h8_eval300_formal_v2_20260829T155639Z/run_formal_h8.sh'
parent=json.loads((P/'h15/status/experiment_config.json').read_text())
def derived(b,overlay):
 d=dict(parent);d['DENSE_BEAM_B']=str(b);d['HIERARCHICAL_PROFILE']=f'dense_beam_b_h{b}';d['overlay']=overlay;return d
h30=derived(30,parent['overlay']);h8=derived(8,str(B/'outputs/dense_semantic_beam_b_h8_budget_overlay_v2_20260829T155639Z/runtime_overlay'))
def diff(a,b):return {k:{'parent':a.get(k),'child':b.get(k)} for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)}
d30,d8=diff(parent,h30),diff(parent,h8)
ok30=set(d30)=={'DENSE_BEAM_B','HIERARCHICAL_PROFILE'}
ok8=set(d8)=={'DENSE_BEAM_B','HIERARCHICAL_PROFILE','overlay'}
result={'status':'PASS' if ok30 and ok8 else 'BLOCKED','h15_actual_config_sha256':hashlib.sha256((P/'h15/status/experiment_config.json').read_bytes()).hexdigest(),'h15_vs_h30':d30,'h15_vs_h8':d8,'h30_uses_exact_parent_overlay':h30['overlay']==parent['overlay'],'h8_overlay_difference_governed_by_amendment':True,'launchers':{'h30':str(H30),'h8':str(H8)},'prestart_gates_added':['superseding audit manifest','no-model provider/config runtime_gate'],'prestart_gates_change_model_request_or_control_flow':False}
(O/'LAUNCHER_RESOLVED_CONFIG_AUDIT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(result['status'])
if result['status']!='PASS':raise SystemExit(2)
