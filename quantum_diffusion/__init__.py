"""Public API for the persistent REPL feedback library.

The names in ``__all__`` are the package-root import surface; implementation
helpers and integration protocols remain available from their defining modules.
"""

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
