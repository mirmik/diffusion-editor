"""Filename contract shared by Qwen front-sector reconstruction scripts."""

from __future__ import annotations

import re


VIEW_PATTERN = re.compile(
    r"^(?:mv-)?(low|eye|elevated)-(000|045|315)"
    r"(?:-multiple-angles-on)?\.png$"
)
