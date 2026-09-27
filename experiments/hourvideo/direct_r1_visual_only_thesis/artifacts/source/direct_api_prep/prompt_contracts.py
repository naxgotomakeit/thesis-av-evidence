"""Frozen text templates only; these make no model/provider call."""

MAX_IMAGES_CLAUSE = "You may select no more than 16 original video frames for final visual evidence. Return ranked frame selections only; the final answering request must contain at most 16 physical images."

R1_NAVIGATION_PROMPT = """Use the supplied frozen R1 structural/ASR navigation map to identify temporal evidence for the question. Do not infer an answer and do not use prior Shared, Fine, or Final traces. Select original frames with a stable best-first ranking. {cap}""".format(cap=MAX_IMAGES_CLAUSE)

R3_NAVIGATION_PROMPT = """Use the supplied frozen R3 semantic hierarchy and captioned regions to navigate to temporal evidence for the question. Do not infer an answer and do not use prior Shared, Fine, or Final traces. Select original frames with a stable best-first ranking. {cap}""".format(cap=MAX_IMAGES_CLAUSE)
