"""Two-qubit matrix operators and deterministic error-state inversion."""

from __future__ import annotations

import hashlib
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]

_H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
HADAMARD = np.kron(_H, np.eye(2, dtype=np.complex128))
CNOT = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ],
    dtype=np.complex128,
)


def _state_vector(state: Sequence[complex] | NDArray[np.complex128]) -> ComplexArray:
    vector = np.asarray(state, dtype=np.complex128)
    if vector.shape != (4,):
        raise ValueError("a two-qubit state must have exactly four amplitudes")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("state amplitudes must have a finite, non-zero norm")
    return vector / norm


def density_matrix(
    state: Sequence[complex] | NDArray[np.complex128],
) -> ComplexArray:
    """Return the normalized pure-state density matrix for a two-qubit state."""
    vector = _state_vector(state)
    return np.outer(vector, vector.conj())


def state_fidelity(
    left: Sequence[complex] | NDArray[np.complex128],
    right: Sequence[complex] | NDArray[np.complex128],
) -> float:
    """Return pure-state fidelity, invariant to global phase."""
    overlap = np.vdot(_state_vector(left), _state_vector(right))
    return float(np.clip(abs(overlap) ** 2, 0.0, 1.0))


class QuantumTwinEvaluator:
    """Encode an error trace and invert it through a fixed unitary operator."""

    def __init__(self, operator: NDArray[np.complex128] | None = None) -> None:
        self.operator = np.asarray(
            np.kron(HADAMARD, CNOT) if operator is None else operator,
            dtype=np.complex128,
        )
        if self.operator.shape != (4, 4):
            raise ValueError("the twin operator must be a 4x4 matrix")
        if not np.allclose(self.operator.conj().T @ self.operator, np.eye(4)):
            raise ValueError("the twin operator must be unitary")

    def encode_trace(self, trace: str) -> ComplexArray:
        """Map arbitrary trace text to a stable normalized four-amplitude state."""
        if not isinstance(trace, str):
            raise TypeError("error trace must be a string")
        digest = hashlib.sha256(trace.encode("utf-8", errors="replace")).digest()
        values = np.frombuffer(digest[:32], dtype=np.uint8).astype(np.float64)
        amplitudes = (values[:4] - 127.5) + 1j * (values[4:8] - 127.5)
        return _state_vector(amplitudes)

    def twin_state(
        self, trace_or_state: str | Sequence[complex] | NDArray[np.complex128]
    ) -> ComplexArray:
        """Return E-dagger applied to an encoded trace or supplied state."""
        state = (
            self.encode_trace(trace_or_state)
            if isinstance(trace_or_state, str)
            else _state_vector(trace_or_state)
        )
        return _state_vector(self.operator.conj().T @ state)

    def forward_state(
        self, state: Sequence[complex] | NDArray[np.complex128]
    ) -> ComplexArray:
        """Apply E, the inverse operation of :meth:`twin_state`."""
        return _state_vector(self.operator @ _state_vector(state))

    def inversion_fidelity(self, trace: str) -> float:
        """Measure fidelity after applying the twin and forward transformations."""
        original = self.encode_trace(trace)
        restored = self.forward_state(self.twin_state(original))
        return state_fidelity(original, restored)

    def entropy(self, state: Sequence[complex]) -> float:
        """Return Shannon entropy of the four measurement probabilities."""
        probabilities = np.abs(_state_vector(state)) ** 2
        nonzero = probabilities[probabilities > 0]
        return float(-np.sum(nonzero * np.log2(nonzero)))
