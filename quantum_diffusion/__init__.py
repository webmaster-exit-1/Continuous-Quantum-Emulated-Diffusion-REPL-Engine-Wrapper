"""Continuous quantum-emulated diffusion REPL components."""

from .canvas import ContinuousCanvas
from .quantum import (
    CNOT,
    HADAMARD,
    QuantumTwinEvaluator,
    density_matrix,
    state_fidelity,
)

__all__ = [
    "CNOT",
    "HADAMARD",
    "ContinuousCanvas",
    "QuantumTwinEvaluator",
    "density_matrix",
    "state_fidelity",
]
