"""Coordinate kernel execution, guidance, and bounded correction attempts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .canvas import ContinuousCanvas
from .errors import format_execution_error
from .guidance import DiffusionSampler, ErrorEntropyGuidance
from .repl import ExecutionResult


class ExecutableKernel(Protocol):
    """Kernel interface required by the feedback loop."""

    async def execute(self, code: str, *, timeout: float = 60.0) -> ExecutionResult: ...


class RetryPolicy(Protocol):
    """Decide whether a failed execution should receive another correction."""

    def should_retry(
        self, execution: ExecutionResult, *, attempt: int, max_attempts: int
    ) -> bool: ...


class AttemptLimitRetryPolicy:
    """Retry failures until the configured attempt limit is reached."""

    def should_retry(
        self, execution: ExecutionResult, *, attempt: int, max_attempts: int
    ) -> bool:
        return execution.status != "ok" and attempt < max_attempts


@dataclass(frozen=True)
class FeedbackResult:
    """Final candidate and execution outcome produced by a feedback run."""

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
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.kernel = kernel
        self.sampler = sampler
        self.canvas = canvas or ContinuousCanvas()
        self.guidance = guidance or ErrorEntropyGuidance()
        self.retry_policy = retry_policy or AttemptLimitRetryPolicy()

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
            trace = format_execution_error(execution.error, execution.stderr)
            self.canvas.append(trace + "\n")
            if not self.retry_policy.should_retry(
                execution, attempt=attempt, max_attempts=max_attempts
            ):
                return FeedbackResult(candidate, execution, attempt, corrected)
            candidate = self.guidance.apply(
                self.sampler, self.canvas.text, trace
            )
            if not isinstance(candidate, str):
                raise TypeError("diffusion sampler must return corrected code as text")
            corrected = True
        raise AssertionError("feedback loop exited without a result")
