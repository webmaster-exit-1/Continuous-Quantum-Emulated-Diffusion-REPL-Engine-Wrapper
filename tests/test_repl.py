import asyncio
import tempfile
import unittest
from unittest.mock import patch

from quantum_diffusion.repl import KernelSession, SandboxProfile


class KernelSessionTests(unittest.IsolatedAsyncioTestCase):
    def test_bubblewrap_command_mounts_workspace_writable_and_root_read_only(self):
        with tempfile.TemporaryDirectory() as workspace:
            profile = SandboxProfile(workspace=workspace)
            with patch("quantum_diffusion.repl.shutil.which", return_value="/usr/bin/bwrap"):
                profile.prepare()
                command = profile.wrap_command(["python", "kernel.py"], "connection.json")

        self.assertEqual(
            command,
            [
                "/usr/bin/bwrap",
                "--die-with-parent",
                "--new-session",
                "--ro-bind",
                "/",
                "/",
                "--bind",
                workspace,
                workspace,
                "--chdir",
                workspace,
                "python",
                "kernel.py",
            ],
        )

    def test_bubblewrap_requires_installed_binary(self):
        with tempfile.TemporaryDirectory() as workspace:
            profile = SandboxProfile(workspace=workspace)
            with patch("quantum_diffusion.repl.shutil.which", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "bubblewrap is required"):
                    profile.prepare()

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

    async def test_concurrent_executions_are_serialized(self):
        async with KernelSession(SandboxProfile(use_bubblewrap=False)) as kernel:
            first, second = await asyncio.gather(
                kernel.execute("import time; time.sleep(0.2); print('first')"),
                kernel.execute("print('second')"),
            )
        self.assertIn("first", first.stdout)
        self.assertNotIn("second", first.stdout)
        self.assertIn("second", second.stdout)
        self.assertNotIn("first", second.stdout)

    async def test_input_without_stdin_support_returns_error(self):
        async with KernelSession(SandboxProfile(use_bubblewrap=False)) as kernel:
            result = await kernel.execute("input('prompt: ')", timeout=5)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error["ename"], "StdinNotImplementedError")

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

    async def test_kernel_does_not_inherit_arbitrary_environment_secrets(self):
        with patch.dict("os.environ", {"REPL_TEST_SECRET": "not-for-kernel"}):
            async with KernelSession(
                SandboxProfile(use_bubblewrap=False)
            ) as kernel:
                result = await kernel.execute("'REPL_TEST_SECRET' in __import__('os').environ")
                self.assertTrue(
                    any(
                        output.get("data", {}).get("text/plain") == "False"
                        for output in result.outputs
                    )
                )
