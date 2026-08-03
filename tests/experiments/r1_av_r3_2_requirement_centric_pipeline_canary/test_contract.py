from experiments.r1_av_r3_2_requirement_centric_pipeline_canary.core import planner_schema, validate_plan


REQ=[{"requirement_id":"q::a"},{"requirement_id":"q::b"}]


def test_duplicate_or_reordered_requirements_fail():
    result={"question_id":"q","requirement_plans":[{"requirement_id":"q::b","query_variants":["b"],"suggested_coarse_ids":["C01"]},{"requirement_id":"q::a","query_variants":["a"],"suggested_coarse_ids":["C01"]}],"hard_filtering_allowed":False}
    assert validate_plan(result,"q",REQ)


def test_exact_requirements_pass():
    result={"question_id":"q","requirement_plans":[{"requirement_id":"q::a","query_variants":["a"],"suggested_coarse_ids":["C01"]},{"requirement_id":"q::b","query_variants":["b"],"suggested_coarse_ids":["C01"]}],"hard_filtering_allowed":False}
    assert validate_plan(result,"q",REQ)==[]


def test_schema_freezes_false_hard_filtering():
    assert planner_schema(REQ,["C01"])["properties"]["hard_filtering_allowed"]["const"] is False
