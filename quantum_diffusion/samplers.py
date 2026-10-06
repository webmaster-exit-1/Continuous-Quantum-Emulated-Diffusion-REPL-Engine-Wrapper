"""Masked discrete diffusion sampler for region-based code repair.

The sampler treats the failing statement as a span of mask tokens, keeps the
code before and after it as fixed context, and runs a fixed number of denoising
steps. Each step scores every still-masked position, samples a candidate token
for each, commits the most confident candidates and leaves the rest masked
(confidence-based iterative unmasking in the style of MaskGIT/MDLM).

``torch`` and ``transformers`` are optional and only imported when a model is
loaded from ``model_path``.
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .guidance import RepairRequest

_TOKEN = re.compile(r"\s+|\w+|[^\w\s]")
_GOLDEN = 0.6180339887498949
_BIAS_SCALE = 0.25


@runtime_checkable
class TokenScorer(Protocol):
    """Scores masked positions of a token sequence.

    ``encode`` maps string tokens to ids (one id per token), ``decode`` maps an
    id back to token text, and ``mask_token_id`` marks masked positions in the
    sequence passed to ``score``.
    """

    mask_token_id: int

    def encode(self, tokens: Sequence[str]) -> list[int]: ...

    def decode(self, token_id: int) -> str: ...

    def score(
        self,
        token_ids: Sequence[int],
        mask_positions: Sequence[int],
        temperature: float,
    ) -> Sequence[Sequence[float]]:
        """Return one logit row (over the vocabulary) per mask position.

        The scorer is responsible for applying ``temperature`` (logits divided
        by it) to the rows it returns.
        """
        ...


class _MaskedLMScorer:
    """Scorer backed by a pretrained masked language model (fill-mask style)."""

    def __init__(self, model_path: str) -> None:
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "DiscreteDiffusionSampler(model_path=...) needs the optional "
                "diffusion dependencies; run `pip install .[diffusion]`"
            ) from exc
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        self.model.eval()
        if self.tokenizer.mask_token_id is None:
            raise ValueError(f"{model_path!r} is not a masked language model")
        self.mask_token_id = int(self.tokenizer.mask_token_id)
        limit = getattr(self.model.config, "max_position_embeddings", 512)
        self._window = max(8, min(int(limit), 512) - 4)

    def encode(self, tokens: Sequence[str]) -> list[int]:
        unk = self.tokenizer.unk_token_id
        ids = []
        for token in tokens:
            pieces = self.tokenizer(token, add_special_tokens=False)["input_ids"]
            ids.append(int(pieces[0]) if pieces else int(unk or 0))
        return ids

    def decode(self, token_id: int) -> str:
        return str(self.tokenizer.decode([token_id])).strip()

    def score(
        self,
        token_ids: Sequence[int],
        mask_positions: Sequence[int],
        temperature: float,
    ) -> Sequence[Sequence[float]]:
        torch = self._torch
        ids = list(token_ids)
        centre = sum(mask_positions) // max(1, len(mask_positions))
        start = max(0, min(centre - self._window // 2, len(ids) - self._window))
        window = ids[start : start + self._window]
        head = [self.tokenizer.cls_token_id] if self.tokenizer.cls_token_id is not None else []
        tail = [self.tokenizer.sep_token_id] if self.tokenizer.sep_token_id is not None else []
        batch = torch.tensor([head + window + tail])
        with torch.no_grad():
            logits = self.model(input_ids=batch).logits[0]
        rows = [
            logits[len(head) + position - start] / temperature
            for position in mask_positions
            if start <= position < start + len(window)
        ]
        if len(rows) != len(mask_positions):
            raise ValueError("masked span does not fit the model context window")
        return torch.stack(rows).float().numpy()


class DiscreteDiffusionSampler:
    """Repair a failing statement by iteratively unmasking it.

    Conditioning from the quantum guidance state is a heuristic, not a
    score-entropy gradient:

    * temperature is ``1.0 + entropy * 0.1``; it is handed to the scorer, which
      divides its logits by it, so higher-entropy failures sample more
      diversely;
    * a deterministic logit bias is added to every masked position. For vocab id
      ``v`` it is ``0.25 * sum_k Re(g_k * exp(2*pi*i * frac((k + 1) * v * phi)))``
      where ``g_k`` are the four guidance amplitudes and ``phi`` is the golden
      ratio conjugate. Equal guidance vectors give equal biases and different
      vectors shift which tokens are favoured.

    Whitespace tokens inside the failing region stay fixed, so indentation and
    layout are preserved; every other token of the region starts masked.

    Without ``scorer`` or ``model_path`` no model is available: ``repair``
    warns and returns the failing region unchanged.
    """

    def __init__(
        self,
        model_path: str | None = None,
        steps: int = 8,
        scorer: TokenScorer | None = None,
        *,
        seed: int | None = 0,
    ) -> None:
        if steps < 1:
            raise ValueError("steps must be at least one")
        self.model_path = model_path
        self.steps = steps
        self.seed = seed
        self._scorer = scorer

    @staticmethod
    def temperature(entropy: float) -> float:
        return 1.0 + float(entropy) * 0.1

    @staticmethod
    def noise_scale(entropy: float) -> float:
        """Map exit-1 Shannon entropy to a diffusion noise level in [0, 1].

        Two qubits have at most 2 bits of entropy. Zero noise is already exit 0.
        """
        return float(min(1.0, max(0.0, float(entropy) / 2.0)))

    @staticmethod
    def logit_bias(guidance: NDArray[np.complex128], vocab_size: int) -> NDArray[np.float64]:
        """Deterministic per-token logit bias derived from the guidance vector."""
        amplitudes = np.asarray(guidance, dtype=np.complex128).reshape(-1)
        ids = np.arange(vocab_size, dtype=np.float64)
        bias = np.zeros(vocab_size, dtype=np.float64)
        for k, amplitude in enumerate(amplitudes):
            phase = np.mod((k + 1) * ids * _GOLDEN, 1.0)
            bias += (amplitude * np.exp(2j * np.pi * phase)).real
        return _BIAS_SCALE * bias

    def _get_scorer(self) -> TokenScorer | None:
        if self._scorer is None and self.model_path is not None:
            self._scorer = _MaskedLMScorer(self.model_path)
        return self._scorer

    def repair(self, request: RepairRequest) -> str:
        failure = request.failure
        region = failure.failing_region
        scorer = self._get_scorer()
        if scorer is None:
            warnings.warn(
                "No diffusion model is configured (no model_path or scorer); "
                "returning the failing region unchanged.",
                stacklevel=2,
            )
            return region

        before = _TOKEN.findall(failure.context_before)
        span = _TOKEN.findall(region)
        after = _TOKEN.findall(failure.context_after)
        masked = [i for i, token in enumerate(span) if not token.isspace()]
        if not masked:
            return region

        ids = scorer.encode(before + span + after)
        offset = len(before)
        positions = [offset + i for i in masked]
        mask_id = scorer.mask_token_id
        for position in positions:
            ids[position] = mask_id
        committed: dict[int, int] = {}
        temperature = self.temperature(request.entropy)
        rng = np.random.default_rng(self.seed)

        for step in range(self.steps):
            remaining = [p for p in positions if p not in committed]
            if not remaining:
                break
            logits = np.asarray(
                scorer.score(ids, remaining, temperature), dtype=np.float64
            )
            if logits.ndim != 2 or logits.shape[0] != len(remaining):
                raise ValueError("scorer must return one logit row per mask position")
            logits = logits + self.logit_bias(request.guidance, logits.shape[1])
            if 0 <= mask_id < logits.shape[1]:
                logits[:, mask_id] = -np.inf
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            choices = [int(rng.choice(probs.shape[1], p=row)) for row in probs]
            confidence = [float(probs[i, c]) for i, c in enumerate(choices)]
            # Entropy is noise: a noisier exit-1 state commits fewer tokens per step.
            noise = self.noise_scale(request.entropy)
            if step == self.steps - 1:
                quota = len(remaining)
            else:
                quota = max(1, math.ceil(len(remaining) * (1.0 - noise) / (self.steps - step)))
            order = sorted(range(len(remaining)), key=lambda i: (-confidence[i], i))
            for i in order[:quota]:
                committed[remaining[i]] = choices[i]
                ids[remaining[i]] = choices[i]

        result = list(span)
        for i, position in zip(masked, positions):
            if position in committed:
                result[i] = scorer.decode(committed[position])
        return "".join(result)
