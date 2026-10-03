import tempfile
import unittest
from pathlib import Path

from quantum_diffusion.repl import KernelSession, SandboxProfile


class KernelSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_kernel_captures_streams_and_errors(self):
        with tempfile.TemporaryDirectory() as workspace:
            async with KernelSession(
                SandboxProfile(workspace=workspace, use_bubblewrap=False)
            ) as kernel:
                result = await kernel.execute(
                    "print('out')\nimport sys\nprint('err', file=sys.stderr)\n2 + 3"
                )
                self.assertEqual(result.status, "ok")
                self.assertIn("out", result.stdout)
                self.assertIn("err", result.stderr)
                self.assertTrue(
                    any(
                        output.get("data", {}).get("text/plain") == "5"
                        for output in result.outputs
                    )
                )

                failure = await kernel.execute("raise ValueError('bad')")
                self.assertEqual(failure.status, "error")
                self.assertEqual(failure.error["ename"], "ValueError")

    async def test_kernel_context_persists_between_executions(self):
        async with KernelSession(
            SandboxProfile(use_bubblewrap=False)
        ) as kernel:
            await kernel.execute("value = 42")
            result = await kernel.execute("value")
            self.assertTrue(
                any(
                    output.get("data", {}).get("text/plain") == "42"
                    for output in result.outputs
                )
            )
