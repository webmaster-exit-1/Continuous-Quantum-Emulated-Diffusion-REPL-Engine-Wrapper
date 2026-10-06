# Continuous Quantum-Emulated Diffusion REPL Engine

A Python library foundation for a persistent Jupyter REPL feedback loop around
trusted or semi-trusted code. It includes two-qubit matrix operators,
deterministic error-trace encoding and inversion, a bounded text canvas, a
diffusion-sampler interface, and a streaming model adapter. The sampler and
inference API are interfaces: this package does not ship or train a diffusion
model, provide a complete hostile-code sandbox, or perform physical quantum
computation.

## Architecture

The project is a library, not a service or CLI. A persistent kernel executes
code, a feedback loop turns failures into deterministic guidance, and a
user-supplied `DiffusionSampler` proposes revised source. Each module has one
responsibility:

| Module | Responsibility |
| --- | --- |
| `quantum_diffusion/repl.py` | Kernel lifecycle and execution. `SandboxProfile` prepares the private workspace and wraps the kernel command with bubblewrap (or not, if disabled). `KernelSession` owns the persistent IPython kernel, serializes executions, and returns structured `ExecutionResult` values (`stdout`, `stderr`, MIME outputs, `status`, `error`). |
| `quantum_diffusion/engine.py` | Orchestration. `REPLFeedbackLoop` runs a candidate, records text on the canvas, asks a `RetryPolicy` whether to retry, and returns a `FeedbackResult`. It depends only on the `ExecutableKernel`, `DiffusionSampler` and `RetryPolicy` protocols. |
| `quantum_diffusion/errors.py` | Error extraction. `format_execution_error` turns a kernel error payload (or stderr) into trace text. |
| `quantum_diffusion/guidance.py` | Guidance and adapters. `ErrorEntropyGuidance` converts a trace into a `Guidance` (state vector and entropy) and calls the `DiffusionSampler`; `ModelAPIWrapper` streams an async inference callable into a canvas. |
| `quantum_diffusion/quantum.py` | Deterministic error-state transform: `HADAMARD`, `CNOT`, `QuantumTwinEvaluator`, `density_matrix`, `state_fidelity`. Pure NumPy; no physical quantum computation. |
| `quantum_diffusion/canvas.py` | `ContinuousCanvas`, a thread-safe, character-bounded text window. |
| `quantum_diffusion/__init__.py` | Declares the supported package-root imports in `__all__`. Extension interfaces (`DiffusionSampler`, `ExecutableKernel`, `RetryPolicy`, `ErrorEntropyGuidance`, `ModelAPIWrapper`) are imported from their defining modules. |

### Execution flow

1. **Source input** – `REPLFeedbackLoop.run(code, max_attempts=..., timeout=...)`
   appends the candidate to the canvas.
2. **Kernel execution** – the candidate runs in the persistent kernel, so
   variables survive between attempts. Success returns immediately with
   `corrected=True` only if an earlier attempt was revised.
3. **Error extraction** – on failure, `format_execution_error` builds trace text
   from the error name, value and traceback (falling back to stderr), and the
   trace is appended to the canvas.
4. **Guidance generation** – `ErrorEntropyGuidance` encodes the trace as a
   normalized state vector and entropy via `QuantumTwinEvaluator`.
5. **Candidate revision** – `DiffusionSampler.guide(canvas, state, entropy)`
   returns replacement source; a non-string result raises `TypeError`.
6. **Retry loop** – the revised candidate is executed again. The loop stops on
   success, when `max_attempts` is reached (a hard bound), when the
   `RetryPolicy` declines, or when a repair returns text that does not change
   the candidate. An unchanged repair is not re-executed.

### Context retention

`ContinuousCanvas` holds source, execution output, error traces and model
updates as one string and keeps only the newest `max_chars` characters
(default 32,768); the oldest text is evicted first. The bound keeps the context
passed to the sampler and inference callable at a predictable size no matter how
long a session runs, and the newest text is the most relevant for the next
revision. Consequences: earlier context can be lost, there is no structure or
token accounting, and a single update longer than the budget is truncated to
its suffix. `ModelAPIWrapper` appends inputs and streamed chunks to its canvas
using the same policy; pass the same canvas to the loop and wrapper to share
context.

### How to extend this project

- **New correction strategy**: implement `DiffusionSampler.guide(canvas,
  guidance, entropy) -> str` and pass it to `REPLFeedbackLoop`.
- **Different retry behavior**: implement `RetryPolicy.should_retry(execution,
  *, attempt, max_attempts)`; `max_attempts` still bounds the run.
- **Different kernel backend**: implement `ExecutableKernel.execute(code, *,
  timeout)` returning an `ExecutionResult`.
- **Different error encoding**: pass a custom `QuantumTwinEvaluator` to
  `ErrorEntropyGuidance`, or change trace formatting in `errors.py`.
- **Different context policy**: change `canvas.py` and keep its `append`/`text`
  interface; add tests next to the existing `unittest` tests.
- **New public names**: add them to `quantum_diffusion/__init__.py` `__all__`
  only if they are intended as stable, and update `tests/test_api.py`.

Keep terminology consistent: "kernel session" for execution, "canvas" for the
bounded context, "guidance" for the deterministic state and entropy, and
"sampler" for the user-supplied revision component.

## Install

```sh
python -m pip install .
```

The diffusion sampler loads a masked language model, not a chat model. The
checkpoint that fits the loader is `microsoft/codebert-base-mlm`
(`quantum_diffusion.RECOMMENDED_MODEL`). `microsoft/codebert-base` is the
replaced-token model and will be rejected because it has no mask token.

```sh
python -m pip install ".[diffusion]"
python examples/repair_with_codebert.py
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
operator. An execution failure is exit 1. `QuantumTwinEvaluator` hashes that
failure into a normalized four-amplitude state, and the entropy of the exit-1
state is the diffusion noise. `exit_twin` applies the adjoint unitary and
forces the exit qubit to 0. The guidance command is that exit-0 twin minus
the exit-1 state: where it isn't, minus where it is. Entropy on where it is
remains the noise. This is a deterministic error-feature
transform, not a physical quantum computation or a guarantee that generated
code will be corrected.

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
