import unittest

from quantum_diffusion.engine import AttemptLimitRetryPolicy, REPLFeedbackLoop
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

    async def test_retry_policy_can_stop_after_first_failure(self):
        class StopAfterFirstFailure:
            def should_retry(self, execution, *, attempt, max_attempts):
                return False

        kernel = FakeKernel()
        result = await REPLFeedbackLoop(
            kernel, CorrectingSampler(), retry_policy=StopAfterFirstFailure()
        ).run("bad")
        self.assertEqual(kernel.calls, ["bad"])
        self.assertEqual(result.attempts, 1)
        self.assertFalse(result.corrected)


class RetryPolicyTests(unittest.TestCase):
    def test_attempt_limit_policy_retries_only_with_attempts_remaining(self):
        policy = AttemptLimitRetryPolicy()
        failure = ExecutionResult(status="error")
        success = ExecutionResult(status="ok")
        self.assertTrue(policy.should_retry(failure, attempt=1, max_attempts=2))
        self.assertFalse(policy.should_retry(failure, attempt=2, max_attempts=2))
        self.assertFalse(policy.should_retry(success, attempt=1, max_attempts=2))
