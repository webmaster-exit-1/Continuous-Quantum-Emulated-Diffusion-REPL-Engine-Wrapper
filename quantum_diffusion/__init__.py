"""Public API for the persistent REPL feedback library.

The names in ``__all__`` are the supported package-root import surface.
"""

from .canvas import ContinuousCanvas
from .engine import (
    AttemptLimitRetryPolicy,
    ExecutableKernel,
    FeedbackResult,
    REPLFeedbackLoop,
    RetryPolicy,
)
from .guidance import DiffusionSampler, Inference
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
    "AttemptLimitRetryPolicy",
    "DiffusionSampler",
    "ExecutableKernel",
    "HADAMARD",
    "ContinuousCanvas",
    "ExecutionResult",
    "FeedbackResult",
    "Inference",
    "KernelSession",
    "QuantumTwinEvaluator",
    "REPLFeedbackLoop",
    "RetryPolicy",
    "SandboxProfile",
    "density_matrix",
    "state_fidelity",
]
