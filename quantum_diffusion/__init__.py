"""Continuous quantum-emulated diffusion REPL components."""

from .canvas import ContinuousCanvas
from .engine import FeedbackResult, REPLFeedbackLoop
from .quantum import (
    CNOT,
    HADAMARD,
    QuantumTwinEvaluator,
    density_matrix,
    state_fidelity,
)
from .repl import ExecutionResult, KernelSession, SandboxProfile

__all__ = [
    "CNOT",
    "HADAMARD",
    "ContinuousCanvas",
    "ExecutionResult",
    "FeedbackResult",
    "KernelSession",
    "QuantumTwinEvaluator",
    "REPLFeedbackLoop",
    "SandboxProfile",
    "density_matrix",
    "state_fidelity",
]
