from __future__ import annotations

from dataclasses import dataclass

from src.diagnostics.cradio_v4.dino_comparison import (
    _plain_mapping,
    boundary_agreement,
    distribution_summary,
)


@dataclass
class _MappingWrapper:
    """Minimal stand-in for Transformers' non-JSON-native SizeDict."""

    height: int
    width: int


def test_plain_mapping_converts_transformers_mapping_wrapper() -> None:
    value = _MappingWrapper(height=224, width=224)
    converted = _plain_mapping(value)
    assert type(converted) is dict
    assert converted == {"height": 224, "width": 224}


def test_boundary_agreement_is_one_to_one_and_deterministic() -> None:
    result = boundary_agreement([10.0, 20.0, 30.0], [9.0, 10.5, 22.0, 40.0], 2.0)
    assert result["matched_count"] == 2
    assert result["matches"] == [
        {"left": 10.0, "right": 10.5, "distance_sec": 0.5},
        {"left": 20.0, "right": 22.0, "distance_sec": 2.0},
    ]
    assert result["unmatched_left"] == [30.0]
    assert result["unmatched_right"] == [9.0, 40.0]


def test_distribution_summary_reports_scale_statistics() -> None:
    result = distribution_summary([0.8, 0.9, 1.0])
    assert result["n"] == 3
    assert result["median"] == 0.9
    assert result["min"] == 0.8
    assert result["max"] == 1.0
