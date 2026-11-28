"""Second-stage re-ranking.

A first-stage retriever (BM25, embeddings, the hybrid) ranks the whole archive cheaply. A slower scorer, for
example a cross-encoder that reads query and message together, then re-scores only the first ``top_n``
messages. Everything below ``top_n`` keeps the first-stage order.

``fuse=False`` orders the top ``top_n`` by the scorer alone. ``fuse=True`` combines the first-stage order and
the scorer's order with reciprocal rank fusion, which is safer when the scorer is noisy.

The scorer is any ``score(query, texts) -> scores`` callable (higher is better), so tests and other models
plug in without extra dependencies. :func:`cross_encoder_scorer` wraps a sentence-transformers CrossEncoder.
Scores are cached per (query, text), so several budgets of the same conversation score each pair once.
"""
from __future__ import annotations

from typing import Callable, Sequence

from .adapters import RankedRetriever
from .types import Message

Scorer = Callable[[str, Sequence[str]], Sequence[float]]


class Rerank(RankedRetriever):
    """Wrap a ranked retriever and re-score its first ``top_n`` archive messages with ``scorer``."""

    def __init__(self, base: RankedRetriever, scorer: Scorer, top_n: int = 100, fuse: bool = False,
                 k: int = 60, max_chars: int | None = 2000, cache: dict | None = None):
        if top_n < 1:
            raise ValueError("top_n must be at least 1")
        super().__init__(base.recent_share, base.pair, base.max_hits)
        self.base, self.scorer, self.top_n, self.fuse, self.k, self.max_chars = base, scorer, top_n, fuse, k, max_chars
        self.cache = {} if cache is None else cache
        self.name = f"{base.name}+rerank" + (" rrf" if fuse else "") + f" N={top_n}"

    def reset(self, budget: int) -> None:
        super().reset(budget)
        self.base.reset(budget)  # some base retrievers read their budget in rank()

    def _scores(self, query: str, msgs: Sequence[Message]) -> list[float]:
        texts = [m.content if self.max_chars is None else m.content[:self.max_chars] for m in msgs]
        todo = sorted({t for t in texts if (query, t) not in self.cache})
        if todo:
            for t, s in zip(todo, self.scorer(query, todo)):
                self.cache[(query, t)] = float(s)
        return [self.cache[(query, t)] for t in texts]

    def rank(self, query: str, archive: list[Message]) -> list[Message]:
        orig = self.base.rank(query, archive)
        top, rest = orig[:self.top_n], orig[self.top_n:]
        if not top:
            return orig
        scores = self._scores(query, top)
        order = sorted(range(len(top)), key=lambda i: (-scores[i], i))  # ties keep first-stage order
        if not self.fuse:
            return [top[i] for i in order] + rest
        fused = [0.0] * len(top)
        for r, i in enumerate(order):
            fused[i] += 1 / (self.k + r + 1)
        for i in range(len(top)):
            fused[i] += 1 / (self.k + i + 1)
        return [top[i] for i in sorted(range(len(top)), key=lambda i: (-fused[i], i))] + rest


def cross_encoder_scorer(model: str = "BAAI/bge-reranker-v2-m3", device: str | None = None,
                         max_length: int = 512, batch_size: int = 64, fp16: bool = True) -> Scorer:
    """A scorer backed by ``sentence_transformers.CrossEncoder`` (install the ``embeddings`` extra)."""
    import torch
    from sentence_transformers import CrossEncoder

    kwargs = {"torch_dtype": torch.float16} if fp16 and device != "cpu" else {}
    ce = CrossEncoder(model, max_length=max_length, device=device, model_kwargs=kwargs)

    def score(query: str, texts: Sequence[str]) -> Sequence[float]:
        return ce.predict([(query, t) for t in texts], batch_size=batch_size, show_progress_bar=False).tolist()

    return score
