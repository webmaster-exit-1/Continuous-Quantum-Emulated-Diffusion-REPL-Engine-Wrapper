"""Two-qubit matrix operators and deterministic error-feature transforms."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .errors import ErrorContext, traceback_frames

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

# Computational basis: |exit, feature>. Exit 0 is success, exit 1 is failure.
EXIT_0 = np.array([1, 0, 0, 0], dtype=np.complex128)
EXIT_1 = np.array([0, 0, 1, 0], dtype=np.complex128)
# Pauli X on the exit qubit. The twin of exit 1 is exit 0.
_STATUS_FLIP = np.array(
    [
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
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
    """Encode structured failure features and apply a fixed unitary transform."""

    def __init__(self, operator: NDArray[np.complex128] | None = None) -> None:
        self.operator = np.asarray(
            CNOT @ HADAMARD if operator is None else operator,
            dtype=np.complex128,
        )
        if self.operator.shape != (4, 4):
            raise ValueError("the twin operator must be a 4x4 matrix")
        if not np.allclose(self.operator.conj().T @ self.operator, np.eye(4)):
            raise ValueError("the twin operator must be unitary")

    def encode_trace(self, trace: str) -> ComplexArray:
        """Encode exception, message, and traceback features into a stable state."""
        if not isinstance(trace, str):
            raise TypeError("error trace must be a string")
        lines = trace.splitlines()
        first = lines[0] if lines else "ExecutionError"
        match = re.match(r"^([\w.]+)(?::\s*(.*))?$", first)
        if match:
            exception_type = match.group(1)
            message = match.group(2) or (lines[1] if len(lines) > 1 else "")
        else:
            exception_type = "ExecutionError"
            message = first
        frames = traceback_frames(tuple(lines))
        return self._encode_features(
            exception_type,
            message,
            tuple((frame.filename, frame.function) for frame in frames),
            "",
            (frames[-1].line, frames[-1].line, frames[-1].line)
            if frames and frames[-1].line is not None
            else None,
        )

    def encode_error(self, error: ErrorContext) -> ComplexArray:
        """Encode structured execution failure and the affected source region."""
        return self._encode_features(
            error.exception_type,
            error.message,
            tuple((frame.filename, frame.function) for frame in error.frames),
            error.failing_region,
            (error.failing_line, error.region_start, error.region_end),
        )

    @staticmethod
    def _encode_features(
        exception_type: str,
        message: str,
        frames: tuple[tuple[str, str], ...],
        failing_region: str,
        location: tuple[int | None, int | None, int | None] | None = None,
    ) -> ComplexArray:
        vector = np.zeros(8, dtype=np.float64)

        def add_tokens(text: str, start: int, width: int, weight: float) -> None:
            tokens = re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*|\d+", text.lower())
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=2).digest()
                index = start + digest[0] % width
                sign = 1.0 if digest[1] & 1 else -1.0
                vector[index] += sign * weight

        if exception_type:
            vector[0] += 1.0
            add_tokens(exception_type, 0, 2, 1.0)
        if message:
            vector[2] += 1.0
            add_tokens(message, 2, 3, 0.65)
        frame_text = " ".join(
            f"{filename} {function}" for filename, function in frames
        )
        if location is not None:
            frame_text += " " + " ".join(
                str(value) for value in location if value is not None
            )
        if frame_text:
            vector[5] += 1.0
            add_tokens(frame_text, 5, 2, 0.5)
        if failing_region:
            vector[7] += 1.0
            add_tokens(failing_region, 7, 1, 0.35)
        amplitudes = vector[::2] + 1j * vector[1::2]
        return _state_vector(amplitudes)


    def as_exit(self, state: Sequence[complex] | NDArray[np.complex128], code: int) -> ComplexArray:
        """Place a feature state on exit 0 or exit 1. 1 is the observed failure."""
        if code not in (0, 1):
            raise ValueError("exit code must be 0 or 1")
        vector = _state_vector(state)
        feature = vector[:2] + vector[2:]
        placed = np.zeros(4, dtype=np.complex128)
        if code == 0:
            placed[:2] = feature
        else:
            placed[2:] = feature
        if float(np.linalg.norm(placed)) == 0:
            placed = EXIT_0 if code == 0 else EXIT_1
        return _state_vector(placed)

    def exit_probability(self, state: Sequence[complex] | NDArray[np.complex128], code: int) -> float:
        """Return the probability of the requested exit qubit."""
        vector = _state_vector(state)
        span = vector[:2] if code == 0 else vector[2:]
        return float(np.sum(np.abs(span) ** 2))

    def exit_twin(self, state: Sequence[complex] | NDArray[np.complex128]) -> ComplexArray:
        """Return the exit-0 twin of an exit-1 error state.

        The observed failure is exit 1. Its entropy is the diffusion noise.
        The adjoint twin operator mixes the feature register, then the exit
        qubit is forced to 0 so the denoiser is guided at the success twin
        rather than at another failure.
        """
        failure = self.as_exit(state, 1)
        mixed = _STATUS_FLIP @ (self.operator.conj().T @ failure)
        mixed[2:] = 0
        if float(np.linalg.norm(mixed)) == 0:
            mixed = EXIT_0.copy()
        return _state_vector(mixed)

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
