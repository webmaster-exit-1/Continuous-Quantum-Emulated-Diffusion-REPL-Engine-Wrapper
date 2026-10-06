"""Run the feedback loop with the CodeBERT masked model.

Install the optional weights first::

    pip install ".[diffusion]"

The checkpoint is microsoft/codebert-base-mlm. microsoft/codebert-base is the
replaced-token model and has no mask token this sampler can fill.
"""

import asyncio

from quantum_diffusion import RECOMMENDED_MODEL, REPLFeedbackLoop
from quantum_diffusion.repl import KernelSession, SandboxProfile
from quantum_diffusion.samplers import DiscreteDiffusionSampler


async def main() -> None:
    sampler = DiscreteDiffusionSampler(model_path=RECOMMENDED_MODEL, steps=8)
    async with KernelSession(SandboxProfile(use_bubblewrap=False)) as kernel:
        result = await REPLFeedbackLoop(kernel, sampler).run(
            "print(missing_name)", max_attempts=2
        )
    print(result.code)
    print(result.execution.status)
    print(result.execution.stdout or result.execution.stderr)


if __name__ == "__main__":
    asyncio.run(main())
