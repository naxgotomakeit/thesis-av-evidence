"""Cost-boundary contract without a provider or tokenizer dependency."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CostBoundaries:
    offline_construction_cost: str
    question_time_navigation_retrieval_cost: str
    direct_answering_api_cost: str
    total_operational_cost: str
    map_text_input_tokens: int | None
    map_text_tokenizer: str | None
    embedding_query_encode_count: int | None
    image_count_transmitted: int | None

    def as_dict(self) -> dict:
        return asdict(self)


R1_R3_TOKEN_REQUIREMENT = "Before a formal Direct run, record actual map/text input tokens using the pinned navigation tokenizer; do not substitute character counts."
GENS_COST_DISCLOSURE = "Native frozen selection is reused without a GenS rerun; its selected-frame count and answering input remain separately visible."
