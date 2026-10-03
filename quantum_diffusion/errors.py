"""Normalize execution failures into trace text for guidance and diagnostics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def format_execution_error(
    error: Mapping[str, Any] | None, stderr: str = ""
) -> str:
    """Format a kernel error payload, using stderr when details are unavailable."""
    if not isinstance(error, Mapping) or not error:
        error = {}
    traceback = error.get("traceback", ())
    if not isinstance(traceback, Iterable) or isinstance(traceback, (str, bytes)):
        traceback = ()
    return "\n".join(
        [
            str(error.get("ename", "ExecutionError")),
            str(error.get("evalue", stderr)),
            *(str(line) for line in traceback),
        ]
    )
