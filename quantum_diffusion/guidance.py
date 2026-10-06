"""Convert execution errors into sampler guidance and adapt streaming inference."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

from numpy.typing import NDArray
import numpy as np

from .canvas import ContinuousCanvas
from .errors import ErrorContext, error_parts
from .quantum import QuantumTwinEvaluator

T = TypeVar("T")


class DiffusionSampler(Protocol):
    """Produce corrected source from canvas context and error guidance."""

    def guide(
        self, canvas: str, guidance: NDArray[np.complex128], entropy: float
    ) -> str: ...


class RepairSampler(Protocol):
    def repair(self, request: RepairRequest) -> str | Awaitable[str]: ...


class CodeRepairBackend(Protocol):
    """Synchronous or asynchronous adapter for a generative code repair model."""

    def repair(self, request: RepairRequest) -> str | Awaitable[str]: ...


@dataclass(frozen=True)
class Guidance:
    state: NDArray[np.complex128]
    entropy: float


@dataclass(frozen=True)
class RepairRequest:
    """Model input for replacing one failing statement in the source."""

    code: str
    failure: ErrorContext
    guidance: NDArray[np.complex128]
    entropy: float


class BackendRepairSampler:
    """Adapt a pluggable repair backend to the feedback loop's sampler interface."""

    def __init__(self, backend: CodeRepairBackend) -> None:
        self.backend = backend

    async def repair(self, request: RepairRequest) -> str:
        result = self.backend.repair(request)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, str):
            raise TypeError("code repair backend must return replacement text")
        return result


class ErrorEntropyGuidance:
    """Encode structured failures as a normalized deterministic state and entropy."""

    def __init__(self, evaluator: QuantumTwinEvaluator | None = None) -> None:
        self.evaluator = evaluator or QuantumTwinEvaluator()

    def from_error(self, error: str | ErrorContext) -> Guidance:
        """Entropy is the noise on the exit-1 state; guidance is its exit-0 twin."""
        encoded = (
            self.evaluator.encode_trace(error)
            if isinstance(error, str)
            else self.evaluator.encode_error(error)
        )
        exit_one = self.evaluator.as_exit(encoded, 1)
        return Guidance(
            state=self.evaluator.exit_twin(exit_one),
            entropy=self.evaluator.entropy(exit_one),
        )

    def apply(
        self,
        sampler: DiffusionSampler,
        canvas: str,
        error_trace: str,
        *,
        error: ErrorContext | None = None,
    ) -> str:
        guidance = self.from_error(error if error is not None else error_trace)
        return sampler.guide(canvas, guidance.state, guidance.entropy)


Inference = Callable[[str], AsyncIterable[T] | Awaitable[AsyncIterable[T]]]


class ModelAPIWrapper:
    """Append model inputs and streamed text updates to a bounded canvas."""

    def __init__(
        self, inference: Inference, canvas: ContinuousCanvas | None = None
    ) -> None:
        self.inference = inference
        self.canvas = canvas or ContinuousCanvas()

    async def stream(self, update: str) -> AsyncIterator[str]:
        current = self.canvas.append(update)
        result = self.inference(current)
        if inspect.isawaitable(result):
            result = await result
        async for chunk in result:
            if not isinstance(chunk, str):
                raise TypeError("model inference must stream string updates")
            yield self.canvas.append(chunk)
