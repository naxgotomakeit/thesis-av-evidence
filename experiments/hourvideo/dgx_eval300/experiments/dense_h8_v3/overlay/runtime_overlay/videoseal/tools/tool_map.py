from __future__ import annotations

from typing import Dict, Type

from videoseal.tools.base import Tool
from videoseal.tools.retrieval_adapter import build_retrieve_tool
from videoseal.tools.visual_tools import VisualInspectAliasTool


def build_tool_map() -> Dict[str, Type[Tool]]:
    return {
        "visual_retrieve": build_retrieve_tool(),
        "visual_inspect": VisualInspectAliasTool,
    }
