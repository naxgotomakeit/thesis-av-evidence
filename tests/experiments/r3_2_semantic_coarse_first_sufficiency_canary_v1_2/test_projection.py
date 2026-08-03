from experiments.r3_2_semantic_coarse_first_sufficiency_canary_v1_2.core import project
def test_stage2_uncertain_projects_to_fine_review_without_status_change():
 rows={'q':[{'requirement_id':'r','status':'uncertain','next_stage':'answer_ready','supporting_coarse_ids':['C01'],'detail_query':'x','rationale':'y'}]};stage2={'q':{'output':{'assessments':[{'requirement_id':'r'}]}}};out=project(rows,stage2)['q'][0];assert out['status']=='uncertain';assert out['next_stage']=='fine_visual_review';assert out['rationale']=='y'
