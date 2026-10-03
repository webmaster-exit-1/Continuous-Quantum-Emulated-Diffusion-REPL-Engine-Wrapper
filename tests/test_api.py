import unittest

import quantum_diffusion


class PublicAPITests(unittest.TestCase):
    def test_package_root_exports_are_explicit_and_importable(self):
        self.assertEqual(
            set(quantum_diffusion.__all__),
            {
                "AttemptLimitRetryPolicy",
                "CNOT",
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
            },
        )
        for name in quantum_diffusion.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(quantum_diffusion, name))
