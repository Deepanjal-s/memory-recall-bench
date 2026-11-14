"""Embedding and hybrid retrievers (``pip install memory-recall-bench[embeddings]``).

The encoder is any callable ``list[str] -> array of shape (n, dim)``. Vectors are
cached by text in an :class:`EmbeddingCache`, so each message is embedded once and
reused across budgets and adapters, and the cache can be saved to disk to separate
the slow part (embedding, ideally on a GPU) from the fast part (the benchmark).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Sequence

from .adapters import RankedRetriever
from .bm25 import BM25, tokenize
from .types import Message

Encoder = Callable[[list[str]], "object"]  # returns an (n, dim) array-like


def _np():
    try:
        import numpy as np
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install the extra: pip install 'memory-recall-bench[embeddings]'") from e
    return np


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Text -> unit-normalised vector. ``max_chars`` truncates text before encoding
    (long chat messages are slow to embed and their opening is usually most telling)."""

    def __init__(self, encoder: Encoder, max_chars: int | None = 1000, batch_size: int = 64):
        self.encoder, self.max_chars, self.batch_size = encoder, max_chars, batch_size
        self._vec: dict[str, object] = {}

    def __len__(self) -> int:
        return len(self._vec)

    def _prep(self, text: str) -> str:
        return text[: self.max_chars] if self.max_chars else text

    def get(self, texts: Sequence[str]):
        np = _np()
        missing = list(dict.fromkeys(t for t in texts if _key(self._prep(t)) not in self._vec))
        for i in range(0, len(missing), self.batch_size):
            chunk = missing[i:i + self.batch_size]
            vecs = np.asarray(self.encoder([self._prep(t) for t in chunk]), dtype=np.float32)
            if vecs.ndim != 2 or len(vecs) != len(chunk):
                raise ValueError(f"encoder must return shape (n, dim); got {vecs.shape} for n={len(chunk)}")
            vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
            for t, v in zip(chunk, vecs):
                self._vec[_key(self._prep(t))] = v
        return np.stack([self._vec[_key(self._prep(t))] for t in texts]) if texts else np.zeros((0, 0))

    def save(self, path: str | Path) -> None:
        np = _np()
        keys = list(self._vec)
        np.savez_compressed(path, keys=np.array(keys), vecs=np.stack([self._vec[k] for k in keys]))

    def load(self, path: str | Path) -> "EmbeddingCache":
        np = _np()
        data = np.load(path)
        self._vec.update(zip(data["keys"].tolist(), data["vecs"]))
        return self


class RecentPlusEmbedding(RankedRetriever):
    """Recent window + archived messages ranked by cosine similarity to the query."""

    name = "recent+embed"

    def __init__(self, cache: EmbeddingCache, recent_share: float = 0.5, pair: bool = True,
                 query_prefix: str = ""):
        super().__init__(recent_share, pair)
        self.cache, self.query_prefix = cache, query_prefix

    def similarities(self, query: str, archive: list[Message]):
        q = self.cache.get([self.query_prefix + query])[0]
        return self.cache.get([m.content for m in archive]) @ q

    def rank(self, query: str, archive: list[Message]) -> list[Message]:
        sims = self.similarities(query, archive)
        order = sorted(range(len(archive)), key=lambda i: -float(sims[i]))
        return [archive[i] for i in order]


class RecentPlusHybrid(RecentPlusEmbedding):
    """Reciprocal rank fusion of BM25 and embedding rankings:
    score = 1/(k + rank_bm25) + 1/(k + rank_embedding). Messages with no BM25 match
    still get the embedding term, so paraphrases can be found."""

    name = "recent+hybrid"

    def __init__(self, cache: EmbeddingCache, recent_share: float = 0.5, pair: bool = True,
                 query_prefix: str = "", k: int = 60):
        super().__init__(cache, recent_share, pair, query_prefix)
        self.k = k

    def rank(self, query: str, archive: list[Message]) -> list[Message]:
        bm = BM25([tokenize(m.content) for m in archive]).scores(tokenize(query))
        sims = self.similarities(query, archive)
        score = [0.0] * len(archive)
        bm_order = [i for i in sorted(range(len(archive)), key=lambda i: -bm[i]) if bm[i] > 0]
        for r, i in enumerate(bm_order):
            score[i] += 1 / (self.k + r + 1)
        for r, i in enumerate(sorted(range(len(archive)), key=lambda i: -float(sims[i]))):
            score[i] += 1 / (self.k + r + 1)
        return [archive[i] for i in sorted(range(len(archive)), key=lambda i: -score[i])]


def fastembed_encoder(model: str = "BAAI/bge-small-en-v1.5", **kwargs) -> Encoder:
    """Default CPU-friendly encoder (ONNX). For bge models pass
    ``query_prefix="Represent this sentence for searching relevant passages: "`` to the retriever."""
    from fastembed import TextEmbedding

    m = TextEmbedding(model, **kwargs)
    return lambda texts: list(m.embed(texts))


def sentence_transformers_encoder(model: str = "BAAI/bge-small-en-v1.5", device: str | None = None) -> Encoder:
    """GPU-friendly encoder (e.g. on Kaggle/Colab)."""
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(model, device=device)
    return lambda texts: m.encode(texts, batch_size=128, convert_to_numpy=True, show_progress_bar=False)
