import unittest

from quantum_diffusion.errors import format_execution_error


class ExecutionErrorFormattingTests(unittest.TestCase):
    def test_formats_kernel_error_fields(self):
        self.assertEqual(
            format_execution_error(
                {
                    "ename": "NameError",
                    "evalue": "unknown name",
                    "traceback": ["line one", "line two"],
                }
            ),
            "NameError\nunknown name\nline one\nline two",
        )

    def test_falls_back_to_stderr_for_missing_or_malformed_details(self):
        self.assertEqual(
            format_execution_error(None, "kernel failed"),
            "ExecutionError\nkernel failed",
        )
        self.assertEqual(
            format_execution_error({"traceback": None}, "kernel failed"),
            "ExecutionError\nkernel failed",
        )
