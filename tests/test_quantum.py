import unittest

import numpy as np

from quantum_diffusion.quantum import (
    CNOT,
    HADAMARD,
    QuantumTwinEvaluator,
    density_matrix,
    state_fidelity,
)


class QuantumOperatorTests(unittest.TestCase):
    def test_gate_operators_are_unitary(self):
        identity = np.eye(4)
        self.assertTrue(np.allclose(HADAMARD.conj().T @ HADAMARD, identity))
        self.assertTrue(np.allclose(CNOT.conj().T @ CNOT, identity))

    def test_density_matrix_is_normalized_and_pure(self):
        matrix = density_matrix([1, 1j, 0, 0])
        self.assertTrue(np.allclose(matrix, matrix.conj().T))
        self.assertAlmostEqual(float(np.trace(matrix).real), 1.0)
        self.assertAlmostEqual(float(np.trace(matrix @ matrix).real), 1.0)

    def test_twin_inversion_preserves_state(self):
        evaluator = QuantumTwinEvaluator()
        self.assertAlmostEqual(evaluator.inversion_fidelity("SyntaxError"), 1.0)
        state = evaluator.encode_trace("ValueError: invalid value")
        twin = evaluator.twin_state(state)
        self.assertAlmostEqual(state_fidelity(state, evaluator.forward_state(twin)), 1.0)
        self.assertGreaterEqual(evaluator.entropy(twin), 0.0)
        self.assertLessEqual(evaluator.entropy(twin), 2.0)

    def test_invalid_matrices_and_states_are_rejected(self):
        with self.assertRaises(ValueError):
            density_matrix([0, 0, 0, 0])
        with self.assertRaises(ValueError):
            QuantumTwinEvaluator(np.zeros((4, 4), dtype=np.complex128))
