"""Bounded execute-and-correct feedback loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .canvas import ContinuousCanvas
from .guidance import DiffusionSampler, ErrorEntropyGuidance
from .repl import ExecutionResult


class ExecutableKernel(Protocol):
    async def execute(self, code: str, *, timeout: float = 60.0) -> ExecutionResult: ...


@dataclass(frozen=True)
class FeedbackResult:
    code: str
    execution: ExecutionResult
    attempts: int
    corrected: bool


class REPLFeedbackLoop:
    """Run code and ask a diffusion sampler to revise failures, up to a limit."""

    def __init__(
        self,
        kernel: ExecutableKernel,
        sampler: DiffusionSampler,
        *,
        canvas: ContinuousCanvas | None = None,
        guidance: ErrorEntropyGuidance | None = None,
    ) -> None:
        self.kernel = kernel
        self.sampler = sampler
        self.canvas = canvas or ContinuousCanvas()
        self.guidance = guidance or ErrorEntropyGuidance()

    async def run(
        self, code: str, *, max_attempts: int = 3, timeout: float = 60.0
    ) -> FeedbackResult:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        candidate = code
        corrected = False
        for attempt in range(1, max_attempts + 1):
            self.canvas.append(candidate + "\n")
            execution = await self.kernel.execute(candidate, timeout=timeout)
            if execution.status == "ok":
                self.canvas.append(execution.stdout + execution.stderr)
                return FeedbackResult(candidate, execution, attempt, corrected)
            error = execution.error or {
                "ename": "ExecutionError",
                "evalue": execution.stderr,
                "traceback": [],
            }
            trace = "\n".join(
                [
                    str(error.get("ename", "ExecutionError")),
                    str(error.get("evalue", "")),
                    *[str(line) for line in error.get("traceback", [])],
                ]
            )
            self.canvas.append(trace + "\n")
            if attempt == max_attempts:
                return FeedbackResult(candidate, execution, attempt, corrected)
            candidate = self.guidance.apply(
                self.sampler, self.canvas.text, trace
            )
            if not isinstance(candidate, str):
                raise TypeError("diffusion sampler must return corrected code as text")
            corrected = True
        raise AssertionError("feedback loop exited without a result")
