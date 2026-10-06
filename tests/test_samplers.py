import re
import subprocess
import sys
import tempfile
import unittest
import warnings

import numpy as np

from quantum_diffusion import DiscreteDiffusionSampler, REPLFeedbackLoop, TokenScorer
from quantum_diffusion.errors import extract_error_context
from quantum_diffusion.guidance import RepairRequest
from quantum_diffusion.repl import ExecutionResult, KernelSession, SandboxProfile


def request(region, before="", after="", guidance=None, entropy=0.5):
    context = extract_error_context(
        "\n".join(part for part in (before, region, after) if part),
        {"ename": "Error", "evalue": "x", "traceback": []},
    )
    context = type(context)(
        **{
            **context.__dict__,
            "failing_region": region,
            "context_before": before,
            "context_after": after,
        }
    )
    if guidance is None:
        guidance = np.array([0.5, 0.5j, -0.5, 0.5], dtype=np.complex128)
    return RepairRequest(
        code=before + region + after, failure=context, guidance=guidance, entropy=entropy
    )


class FakeScorer:
    """Tiny vocabulary scorer; logits come from a fixed random table per token id."""

    mask_token_id = 0

    def __init__(self, vocab_size=12):
        self.vocab = ["[MASK]"]
        self.vocab_size = vocab_size
        self.calls = 0

    def encode(self, tokens):
        ids = []
        for token in tokens:
            if token not in self.vocab:
                self.vocab.append(token)
            ids.append(self.vocab.index(token))
        return ids

    def decode(self, token_id):
        return self.vocab[token_id] if token_id < len(self.vocab) else f"t{token_id}"

    def score(self, token_ids, mask_positions, temperature):
        self.calls += 1
        rng = np.random.default_rng(len(mask_positions))
        rows = rng.normal(size=(len(mask_positions), max(self.vocab_size, len(self.vocab))))
        for i, token in enumerate(self.vocab):
            if token.isspace():
                rows[:, i] = -50.0
        return rows / temperature


class FixScorer:
    """Context-driven scorer: fills a call statement with a variable defined earlier."""

    mask_token_id = 0

    def __init__(self):
        self.vocab = ["[MASK]", "print", "(", ")", "[", "]", "0", "1"]
        self.plan = None

    def encode(self, tokens):
        self.plan = None
        ids = []
        for token in tokens:
            if token not in self.vocab:
                self.vocab.append(token)
            ids.append(self.vocab.index(token))
        return ids

    def decode(self, token_id):
        return self.vocab[token_id]

    def score(self, token_ids, mask_positions, temperature):
        words = [
            self.vocab[i]
            for i in token_ids
            if i != self.mask_token_id and not self.vocab[i].isspace()
        ]
        defined = [
            words[i - 1] for i in range(1, len(words)) if words[i] == "="
        ]
        name = self.vocab.index(defined[-1]) if defined else 1
        v = self.vocab.index
        if self.plan is None:
            run = len(mask_positions)
            template = (
                [v("print"), v("("), name, v(")")]
                if run == 4
                else [v("print"), v("("), name, v("["), v("0"), v("]"), v(")")]
            )
            self.plan = dict(zip(mask_positions, template))
        rows = np.zeros((len(mask_positions), len(self.vocab)))
        for row, position in zip(rows, mask_positions):
            row[self.plan[position]] = 20.0
        return rows / temperature


class SamplerTests(unittest.TestCase):
    def test_scorer_protocol(self):
        self.assertIsInstance(FakeScorer(), TokenScorer)

    def test_only_masked_span_changes_and_context_is_preserved(self):
        before, after = "a = 1\n  b = 2", "c = 3\n"
        sampler = DiscreteDiffusionSampler(scorer=FakeScorer(), steps=4)
        result = sampler.repair(request("  x = y + z", before, after))
        self.assertTrue(result.startswith("  "))
        self.assertEqual(
            [t.isspace() for t in re.findall(r"\s+|\w+|[^\w\s]", result)],
            [t.isspace() for t in re.findall(r"\s+|\w+|[^\w\s]", "  x = y + z")],
        )
        self.assertNotEqual(result, "  x = y + z")
        req = request("  x = y + z", before, after)
        self.assertEqual(req.failure.context_before, before)
        self.assertEqual(req.failure.context_after, after)

    def test_determinism_and_guidance_bias(self):
        g1 = np.array([1, 0, 0, 0], dtype=np.complex128)
        g2 = np.array([0, 1j, 0, 0], dtype=np.complex128)
        run = lambda g: DiscreteDiffusionSampler(scorer=FakeScorer(), seed=3).repair(
            request("x = y + z", guidance=g)
        )
        self.assertEqual(run(g1), run(g1))
        bias = DiscreteDiffusionSampler.logit_bias
        np.testing.assert_array_equal(bias(g1, 12), bias(g1, 12))
        self.assertFalse(np.allclose(bias(g1, 12), bias(g2, 12)))
        self.assertAlmostEqual(DiscreteDiffusionSampler.temperature(2.0), 1.2)

    def test_steps_bound_scorer_calls(self):
        for steps in (1, 2, 3):
            scorer = FakeScorer()
            DiscreteDiffusionSampler(scorer=scorer, steps=steps).repair(
                request("a = b + c * d")
            )
            self.assertLessEqual(scorer.calls, steps)
            self.assertGreaterEqual(scorer.calls, 1)

    def test_no_model_returns_region_and_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = DiscreteDiffusionSampler().repair(request("x = 1 / 0", "a = 1"))
        self.assertEqual(result, "x = 1 / 0")
        self.assertTrue(any("no diffusion model" in str(w.message).lower() for w in caught))

    def test_model_path_without_extras_raises_install_hint(self):
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            with self.assertRaises(ImportError) as ctx:
                DiscreteDiffusionSampler(model_path="missing").repair(request("x"))
            self.assertIn("pip install .[diffusion]", str(ctx.exception))

    def test_import_does_not_import_torch(self):
        code = "import sys, quantum_diffusion; sys.exit('torch' in sys.modules)"
        self.assertEqual(subprocess.run([sys.executable, "-c", code]).returncode, 0)


class FakeKernel:
    async def execute(self, code, *, timeout=60.0):
        return ExecutionResult(
            status="error",
            stdout="",
            stderr="",
            error={"ename": "NameError", "evalue": "boom", "traceback": []},
        )


class LoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_sampler_stops_when_repair_does_not_change_code(self):
        kernel = FakeKernel()
        loop = REPLFeedbackLoop(kernel, None)
        self.assertIsInstance(loop.sampler, DiscreteDiffusionSampler)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = await loop.run("print(x)", max_attempts=3)
        self.assertFalse(result.corrected)
        self.assertEqual(result.code, "print(x)")
        self.assertEqual(result.attempts, 1)

    async def test_end_to_end_repair_by_token_logits(self):
        snippets = (
            ("total = 5\nprint(totl)", "5"),
            ("items = [9]\nprint(items[3])", "9"),
        )
        with tempfile.TemporaryDirectory() as workspace:
            async with KernelSession(
                SandboxProfile(workspace=workspace, use_bubblewrap=False)
            ) as kernel:
                loop = REPLFeedbackLoop(
                    kernel, DiscreteDiffusionSampler(scorer=FixScorer(), steps=3)
                )
                for code, output in snippets:
                    result = await loop.run(code)
                    self.assertEqual(result.execution.status, "ok", result.code)
                    self.assertIn(output, result.execution.stdout)
                    self.assertTrue(result.corrected)


class RealTracebackTests(unittest.IsolatedAsyncioTestCase):
    async def test_failing_line_from_real_ipykernel_traceback(self):
        code = "x = 1\ny = 2\nz = y / 0\nprint(z)"
        async with KernelSession(SandboxProfile(use_bubblewrap=False)) as kernel:
            result = await kernel.execute(code)
        context = extract_error_context(code, result.error, result.stderr)
        self.assertEqual(context.failing_line, 3)
        self.assertEqual(context.failing_region, "z = y / 0")


if __name__ == "__main__":
    unittest.main()
