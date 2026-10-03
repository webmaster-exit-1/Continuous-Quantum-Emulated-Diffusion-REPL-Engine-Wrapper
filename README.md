# Continuous Quantum-Emulated Diffusion REPL Engine

A Python library foundation for a persistent Jupyter REPL feedback loop around
trusted or semi-trusted code. It includes two-qubit matrix operators,
deterministic error-trace encoding and inversion, a bounded text canvas, a
diffusion-sampler interface, and a streaming model adapter. The sampler and
inference API are interfaces: this package does not ship or train a diffusion
model, provide a complete hostile-code sandbox, or perform physical quantum
computation.

## Architecture

`KernelSession` owns a persistent IPython kernel and returns structured
`ExecutionResult` values. `REPLFeedbackLoop` records candidates and output in a
`ContinuousCanvas`, formats execution errors, obtains state/entropy guidance,
and asks a `DiffusionSampler` for a revision. An injectable retry policy decides
whether another correction should be attempted; the existing `max_attempts`
argument remains the hard per-run bound. `ErrorEntropyGuidance` and
`QuantumTwinEvaluator` keep error encoding separate from loop orchestration.
`ModelAPIWrapper` is a separate adapter for asynchronous inference streams.

The main flow is source → kernel result → normalized error trace → deterministic
guidance → sampler candidate → bounded retry. The canvas retains the newest
text by character count, including appended source, errors, output, and model
updates; it is not a structured history or token-aware prompt manager.

The package-root names listed in `quantum_diffusion.__all__` are the supported
convenience imports. Extension interfaces and helpers can be imported from
their defining modules, such as `quantum_diffusion.engine` and
`quantum_diffusion.guidance`.

## Install

```sh
python -m pip install .
```

Kernel filesystem isolation uses
[bubblewrap](https://github.com/containers/bubblewrap) by default. Install it
using your operating system's package manager before starting a kernel. The
writable execution directory is private to the session; the rest of the
filesystem is mounted read-only inside the kernel. This is partial filesystem
isolation only, not a security boundary for hostile code.

## Quick start

```python
import asyncio

from quantum_diffusion import KernelSession, SandboxProfile
from quantum_diffusion.engine import REPLFeedbackLoop


class MySampler:
    def guide(self, canvas, guidance, entropy):
        # Replace this with a sampler that consumes the guidance vector.
        return "print('corrected')"


async def main():
    async with KernelSession(SandboxProfile()) as kernel:
        loop = REPLFeedbackLoop(kernel, MySampler())
        result = await loop.run("prin('typo')", max_attempts=2)
        print(result.code, result.execution.stdout, result.execution.status)


asyncio.run(main())
```

To run a kernel without bubblewrap, explicitly pass
`SandboxProfile(use_bubblewrap=False)`. This is intended only for trusted code
and development; it removes the filesystem isolation.

The kernel keeps variables between executions. `ExecutionResult` includes
`stdout`, `stderr`, rich MIME outputs, execution status, and exception details.
`REPLFeedbackLoop` uses the exception trace to calculate a normalized twin
state and entropy, asks the supplied sampler for revised source, and retries
up to the configured attempt limit. `ContinuousCanvas` bounds the retained
text context by character count. `ModelAPIWrapper` accepts an async streaming
inference callable and appends its updates to the same canvas.

## Matrix and error-state model

`HADAMARD` is the two-qubit operator H ⊗ I; `CNOT` is the controlled-NOT
operator. `QuantumTwinEvaluator` hashes an error trace into a normalized
four-amplitude complex vector and applies the adjoint of a composed unitary.
This is a deterministic error-feature transform, not a physical quantum
computation or a semantic guarantee that generated code will be corrected.

## Security and scope

Bubblewrap gives the kernel a read-only view of the host filesystem and a
writable session workspace. It does not disable networking, install seccomp
filters, limit CPU/memory/process use, or provide identity separation. The
kernel runs under the current user's identity, so filesystem permissions and
other host-level access may still matter. Do not run arbitrary hostile code
where containment is required; use a separately hardened container or VM with
network, syscall, resource, and identity controls. Explicitly disabling
bubblewrap is intended only for trusted code and development.

## Tests

```sh
python -m unittest discover -s tests -v
```
