from experiments.reviewed_visual_evidence_cache_v1.conflict import resolve_exact_scope
from experiments.r1_av_r3_2_review_cache_integration_canary_v1.core import apply_scope_records,validate_batch

def test_reviewed_visual_wins_only_exact_scope():
 rows=[{"evidence_id":"caption","evidence_type":"visual_caption","image_sha256":"h","fine_id":"F","timestamp_sec":1,"fact_scope":"weapon_visibility","assertion":"confirmed"},{"evidence_id":"review","evidence_type":"reviewed_visual_frame","image_sha256":"h","fine_id":"F","timestamp_sec":1,"fact_scope":"weapon_visibility","assertion":"not_supported"},{"evidence_id":"other_time","evidence_type":"visual_caption","image_sha256":"x","fine_id":"X","timestamp_sec":2,"fact_scope":"weapon_visibility","assertion":"confirmed"}]
 out=resolve_exact_scope(rows);exact=next(x for x in out["resolutions"] if x["scope_key"]["fine_id"]=="F");assert exact["preferred_evidence_ids"]==["review"] and exact["superseded_evidence_ids"]==["caption"]
 assert any(x["scope_key"]["fine_id"]=="X" and not x["superseded_evidence_ids"] for x in out["resolutions"])

def test_batch_requires_exact_fine_coverage():
 value={"reviews":[{"fine_id":"F1","status":"confirmed","requirement_effect":"supports_requirement","direct_visual_support":True,"finding":"visible","confidence":"high"}]}
 assert validate_batch(value,["F1"])==[] and validate_batch(value,["F1","F2"])

def test_scope_update_does_not_touch_other_requirement():
 rows=[{"requirement_id":"r1","status":"uncertain","direct_support":False},{"requirement_id":"r2","status":"not_found","direct_support":False}]
 records=[{"fine_id":"F1","requirement_effect":"supports_requirement","direct_visual_support":True,"finding":"visible"}]
 out=apply_scope_records(rows,["r1"],"weapon_visibility",records);assert out[0]["status"]=="supported" and out[1]==rows[1]
