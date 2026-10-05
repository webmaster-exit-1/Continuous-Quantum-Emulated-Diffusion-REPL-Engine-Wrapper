"""Coordinate kernel execution, guidance, and bounded correction attempts."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Protocol

from .canvas import ContinuousCanvas
from .errors import apply_region_repair, extract_error_context, format_execution_error
from .guidance import (
    DiffusionSampler,
    ErrorEntropyGuidance,
    RepairRequest,
    RepairSampler,
)
from .repl import ExecutionResult


class ExecutableKernel(Protocol):
    """Asynchronous kernel contract used by the feedback loop.

    Implementations accept source code and the timeout keyword, then return an
    ``ExecutionResult`` describing the execution.
    """

    async def execute(self, code: str, *, timeout: float = 60.0) -> ExecutionResult: ...


class RetryPolicy(Protocol):
    """Decide whether a failed execution should receive another correction.

    ``attempt`` is one-based; ``max_attempts`` is the run's upper bound.
    """

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
        sampler: DiffusionSampler | RepairSampler,
        *,
        canvas: ContinuousCanvas | None = None,
        guidance: ErrorEntropyGuidance | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.kernel = kernel
        self.sampler = sampler
        self.canvas = canvas or ContinuousCanvas()
        self.guidance = guidance or ErrorEntropyGuidance()
        self.retry_policy = (
            retry_policy if retry_policy is not None else AttemptLimitRetryPolicy()
        )

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
            failure = extract_error_context(
                candidate, execution.error, execution.stderr
            )
            if attempt >= max_attempts or not self.retry_policy.should_retry(
                execution, attempt=attempt, max_attempts=max_attempts
            ):
                return FeedbackResult(candidate, execution, attempt, corrected)
            repair = getattr(self.sampler, "repair", None)
            if repair is not None:
                guidance = self.guidance.from_error(failure)
                replacement = repair(
                    RepairRequest(
                        code=candidate,
                        failure=failure,
                        guidance=guidance.state,
                        entropy=guidance.entropy,
                    )
                )
                if inspect.isawaitable(replacement):
                    replacement = await replacement
                if not isinstance(replacement, str):
                    raise TypeError("code repair sampler must return replacement text")
                candidate = apply_region_repair(candidate, failure, replacement)
            else:
                candidate = self.guidance.apply(
                    self.sampler, self.canvas.text, trace, error=failure
                )
                if not isinstance(candidate, str):
                    raise TypeError("diffusion sampler must return corrected code as text")
            corrected = True
        raise AssertionError("feedback loop exited without a result")
