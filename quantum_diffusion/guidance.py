"""Convert execution errors into sampler guidance and adapt streaming inference."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

from numpy.typing import NDArray
import numpy as np

from .canvas import ContinuousCanvas
from .quantum import QuantumTwinEvaluator

T = TypeVar("T")


class DiffusionSampler(Protocol):
    def guide(
        self, canvas: str, guidance: NDArray[np.complex128], entropy: float
    ) -> str: ...


@dataclass(frozen=True)
class Guidance:
    state: NDArray[np.complex128]
    entropy: float


class ErrorEntropyGuidance:
    """Encode an error trace as a normalized deterministic state and entropy."""

    def __init__(self, evaluator: QuantumTwinEvaluator | None = None) -> None:
        self.evaluator = evaluator or QuantumTwinEvaluator()

    def from_error(self, error_trace: str) -> Guidance:
        state = self.evaluator.twin_state(error_trace)
        return Guidance(state=state, entropy=self.evaluator.entropy(state))

    def apply(self, sampler: DiffusionSampler, canvas: str, error_trace: str) -> str:
        guidance = self.from_error(error_trace)
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
