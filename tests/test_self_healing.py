import tempfile
import unittest

from quantum_diffusion.engine import REPLFeedbackLoop
from quantum_diffusion.guidance import BackendRepairSampler
from quantum_diffusion.repl import KernelSession, SandboxProfile


class EvaluationRepairBackend:
    repairs = {
        "print(missing_name)": "print('repaired')",
        "value = 4 / 0": "value = 4",
        "print(items[2])": "print(items[0])",
    }

    def __init__(self):
        self.requests = []

    async def repair(self, request):
        self.requests.append(request)
        return self.repairs[request.failure.failing_region]


class SelfHealingEvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_loop_repairs_known_failures_in_a_persistent_kernel(self):
        snippets = (
            ("print(missing_name)", "repaired", "print('repaired')"),
            ("value = 4 / 0\nprint(value)", "4", "value = 4\nprint(value)"),
            ("items = [1]\nprint(items[2])", "1", "items = [1]\nprint(items[0])"),
        )
        backend = EvaluationRepairBackend()

        with tempfile.TemporaryDirectory() as workspace:
            async with KernelSession(
                SandboxProfile(workspace=workspace, use_bubblewrap=False)
            ) as kernel:
                loop = REPLFeedbackLoop(kernel, BackendRepairSampler(backend))
                for code, expected_output, expected_code in snippets:
                    with self.subTest(code=code):
                        result = await loop.run(code)
                        self.assertEqual(result.execution.status, "ok")
                        self.assertIn(expected_output, result.execution.stdout)
                        self.assertTrue(result.corrected)
                        self.assertEqual(result.code, expected_code)

        self.assertEqual(len(backend.requests), len(snippets))
        self.assertTrue(
            all(request.guidance.shape == (4,) for request in backend.requests)
        )
        self.assertTrue(all(request.entropy >= 0 for request in backend.requests))
        self.assertEqual(backend.requests[1].failure.context_after, "print(value)")
        self.assertEqual(backend.requests[2].failure.context_before, "items = [1]")
