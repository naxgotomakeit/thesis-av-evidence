"""Deliberately simple full-video uniform-frame EgoPolice baseline."""

from .core import build_mcq_prompt, load_mcq_cases, parse_prediction, uniform_timestamps
from .runner import resolve_runtime_config

__all__ = [
    "build_mcq_prompt", "load_mcq_cases", "parse_prediction", "resolve_runtime_config",
    "uniform_timestamps",
]
