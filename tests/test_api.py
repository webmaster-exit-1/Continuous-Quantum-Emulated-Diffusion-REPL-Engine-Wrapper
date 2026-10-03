import unittest

import quantum_diffusion


class PublicAPITests(unittest.TestCase):
    def test_package_root_exports_are_explicit_and_importable(self):
        self.assertEqual(
            set(quantum_diffusion.__all__),
            {
                "CNOT",
                "HADAMARD",
                "ContinuousCanvas",
                "ExecutionResult",
                "FeedbackResult",
                "KernelSession",
                "QuantumTwinEvaluator",
                "REPLFeedbackLoop",
                "SandboxProfile",
                "density_matrix",
                "state_fidelity",
            },
        )
        for name in quantum_diffusion.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(quantum_diffusion, name))
