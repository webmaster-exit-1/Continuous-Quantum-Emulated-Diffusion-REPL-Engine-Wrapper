"""Normalize execution failures into trace text for guidance and diagnostics."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class TracebackFrame:
    filename: str
    line: int | None
    function: str = ""


@dataclass(frozen=True)
class ErrorContext:
    exception_type: str
    message: str
    traceback: tuple[str, ...]
    frames: tuple[TracebackFrame, ...]
    failing_line: int
    region_start: int
    region_end: int
    failing_region: str
    context_before: str
    context_after: str


_FRAME_PATTERNS = (
    re.compile(
        r'File ["\'](?P<filename>[^"\']+)["\'], line (?P<line>\d+)'
        r'(?:, in (?P<function>.+))?'
    ),
    re.compile(
        r"File (?P<filename>[^\s\"']+):(?P<line>\d+)(?:, in (?P<function>.+))?"
    ),
    re.compile(
        r"Cell In\[[^\]]+\], line (?P<line>\d+)"
        r"(?:, in (?P<function>.+))?"
    ),
)
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def error_parts(error: Mapping[str, Any] | None) -> tuple[str, str, tuple[str, ...]]:
    if not isinstance(error, Mapping):
        error = {}
    traceback = error.get("traceback", ())
    if not isinstance(traceback, Iterable) or isinstance(traceback, (str, bytes)):
        traceback = ()
    return (
        str(error.get("ename", "ExecutionError")),
        str(error.get("evalue", "")),
        tuple(str(line) for line in traceback),
    )


def traceback_frames(traceback: tuple[str, ...]) -> tuple[TracebackFrame, ...]:
    frames = []
    for text in traceback:
        for line in _ANSI_ESCAPE.sub("", text).splitlines():
            for pattern in _FRAME_PATTERNS:
                match = pattern.search(line)
                if match:
                    frames.append(
                        TracebackFrame(
                            filename=match.groupdict().get("filename") or "<repl>",
                            line=int(match.group("line")),
                            function=match.groupdict().get("function") or "",
                        )
                    )
                    break
    return tuple(frames)


def extract_error_context(
    code: str, error: Mapping[str, Any] | None, stderr: str = ""
) -> ErrorContext:
    """Extract traceback features and the smallest statement containing the failure."""
    exception_type, message, traceback = error_parts(error)
    message = message or stderr
    frames = traceback_frames(traceback)
    lines = code.splitlines(keepends=True)
    explicit_line = None
    if isinstance(error, Mapping):
        for key in ("lineno", "line_number", "line"):
            value = error.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                explicit_line = value
                break
    failing_line = explicit_line or next(
        (frame.line for frame in reversed(frames) if frame.line is not None), 1
    )
    failing_line = max(1, min(failing_line, max(1, len(lines))))
    region_start = region_end = failing_line
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        tree = None
    if tree is not None:
        candidates = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.stmt)
            and node.lineno <= failing_line <= getattr(node, "end_lineno", node.lineno)
        ]
        if candidates:
            node = min(
                candidates,
                key=lambda item: (
                    getattr(item, "end_lineno", item.lineno) - item.lineno,
                    -item.lineno,
                ),
            )
            region_start = node.lineno
            region_end = getattr(node, "end_lineno", node.lineno)
    region_start = max(1, min(region_start, max(1, len(lines))))
    region_end = max(region_start, min(region_end, max(1, len(lines))))
    return ErrorContext(
        exception_type=exception_type,
        message=message,
        traceback=traceback,
        frames=frames,
        failing_line=failing_line,
        region_start=region_start,
        region_end=region_end,
        failing_region="".join(lines[region_start - 1 : region_end]).rstrip("\r\n"),
        context_before="".join(lines[: region_start - 1]).rstrip("\r\n"),
        context_after="".join(lines[region_end:]).rstrip("\r\n"),
    )


def apply_region_repair(code: str, context: ErrorContext, replacement: str) -> str:
    """Replace only the extracted failing statement while preserving its line ending."""
    lines = code.splitlines(keepends=True)
    start = context.region_start - 1
    end = context.region_end
    original = "".join(lines[start:end])
    if (
        original.endswith(("\n", "\r"))
        and replacement
        and not replacement.endswith(("\n", "\r"))
    ):
        replacement += "\n"
    return "".join(lines[:start]) + replacement + "".join(lines[end:])


def format_execution_error(
    error: Mapping[str, Any] | None, stderr: str = ""
) -> str:
    """Format a kernel error payload, using stderr when details are unavailable."""
    exception_type, message, traceback = error_parts(error)
    return "\n".join(
        [
            exception_type,
            message or stderr,
            *traceback,
        ]
    )
