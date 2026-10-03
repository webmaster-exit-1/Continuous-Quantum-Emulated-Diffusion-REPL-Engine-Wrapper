import unittest

from quantum_diffusion.engine import REPLFeedbackLoop
from quantum_diffusion.repl import ExecutionResult


class FakeKernel:
    def __init__(self):
        self.calls = []

    async def execute(self, code, *, timeout=60.0):
        self.calls.append(code)
        if code == "bad":
            return ExecutionResult(
                status="error",
                error={"ename": "SyntaxError", "evalue": "invalid syntax", "traceback": []},
            )
        return ExecutionResult(stdout="ok\n")


class CorrectingSampler:
    def __init__(self):
        self.guidance = None

    def guide(self, canvas, guidance, entropy):
        self.guidance = (guidance, entropy)
        return "good"


class FeedbackLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_syntax_error_is_guided_and_corrected(self):
        kernel = FakeKernel()
        sampler = CorrectingSampler()
        result = await REPLFeedbackLoop(kernel, sampler).run("bad")
        self.assertEqual(kernel.calls, ["bad", "good"])
        self.assertEqual(result.code, "good")
        self.assertTrue(result.corrected)
        self.assertEqual(result.attempts, 2)
        self.assertIsNotNone(sampler.guidance)

    async def test_attempt_limit_is_respected(self):
        kernel = FakeKernel()
        result = await REPLFeedbackLoop(kernel, CorrectingSampler()).run(
            "bad", max_attempts=1
        )
        self.assertEqual(kernel.calls, ["bad"])
        self.assertEqual(result.execution.status, "error")
        with self.assertRaises(ValueError):
            await REPLFeedbackLoop(kernel, CorrectingSampler()).run(
                "bad", max_attempts=0
            )
