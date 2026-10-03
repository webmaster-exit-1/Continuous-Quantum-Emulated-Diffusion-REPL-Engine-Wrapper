import unittest

from quantum_diffusion.canvas import ContinuousCanvas
from quantum_diffusion.guidance import ErrorEntropyGuidance, ModelAPIWrapper


class CanvasTests(unittest.TestCase):
    def test_canvas_retains_bounded_suffix(self):
        canvas = ContinuousCanvas(max_chars=5)
        self.assertEqual(canvas.append("abc"), "abc")
        self.assertEqual(canvas.append("def"), "bcdef")
        self.assertEqual(canvas.append("123456"), "23456")
        self.assertEqual(canvas.version, 3)

    def test_canvas_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            ContinuousCanvas(0)
        with self.assertRaises(TypeError):
            ContinuousCanvas().append(None)


class GuidanceTests(unittest.TestCase):
    def test_error_is_converted_to_normalized_guidance(self):
        guidance = ErrorEntropyGuidance().from_error("NameError: x")
        self.assertAlmostEqual(float(abs(guidance.state @ guidance.state.conj()) ** 0.5), 1.0)
        self.assertGreaterEqual(guidance.entropy, 0)


class ModelWrapperTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_updates_accumulate_in_canvas(self):
        async def inference(context):
            async def chunks():
                yield f"\n# generated after {context}"

            return chunks()

        wrapper = ModelAPIWrapper(inference, ContinuousCanvas(max_chars=200))
        updates = [item async for item in wrapper.stream("print('hello')")]
        self.assertEqual(len(updates), 1)
        self.assertIn("print('hello')", wrapper.canvas.text)
        self.assertIn("# generated after", wrapper.canvas.text)
