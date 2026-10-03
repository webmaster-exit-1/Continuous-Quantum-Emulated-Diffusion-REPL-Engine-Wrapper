import unittest

from quantum_diffusion.errors import (
    apply_region_repair,
    extract_error_context,
    format_execution_error,
)


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

    def test_extracts_failing_statement_and_surrounding_context(self):
        code = "before = 1\nresult = compute(\n    missing,\n)\nafter = 2\n"
        error = {
            "ename": "NameError",
            "evalue": "name 'missing' is not defined",
            "traceback": ["Cell In[4], line 3", "    missing,"],
        }

        context = extract_error_context(code, error)

        self.assertEqual(context.failing_line, 3)
        self.assertEqual((context.region_start, context.region_end), (2, 4))
        self.assertEqual(context.failing_region, "result = compute(\n    missing,\n)")
        self.assertEqual(context.context_before, "before = 1")
        self.assertEqual(context.context_after, "after = 2")
        repaired = apply_region_repair(code, context, "result = 1")
        self.assertEqual(repaired, "before = 1\nresult = 1\nafter = 2\n")

    def test_uses_structured_line_number_when_traceback_is_unavailable(self):
        context = extract_error_context(
            "first = 1\nsecond = 2\n",
            {"ename": "SyntaxError", "evalue": "invalid syntax", "lineno": 2},
        )
        self.assertEqual(context.failing_line, 2)
        self.assertEqual(context.failing_region, "second = 2")

    def test_region_repair_preserves_crlf_line_endings(self):
        code = "before = 1\r\nvalue = 1 / 0\r\nafter = 2\r\n"
        context = extract_error_context(
            code,
            {
                "ename": "ZeroDivisionError",
                "evalue": "division by zero",
                "traceback": ["Cell In[1], line 2"],
            },
        )

        repaired = apply_region_repair(code, context, "value = 1")

        self.assertEqual(repaired, "before = 1\r\nvalue = 1\r\nafter = 2\r\n")

    def test_uses_stderr_when_kernel_error_has_no_message(self):
        context = extract_error_context(
            "raise RuntimeError()",
            {"ename": "RuntimeError"},
            "runtime failure details",
        )
        self.assertEqual(context.message, "runtime failure details")
