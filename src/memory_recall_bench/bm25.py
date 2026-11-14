"""A small Okapi BM25 implementation (no dependencies)."""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    """a an the and or but if then of to in on at for with by from as is are was were be been being i you he she it we
    they me my your our their this that these those do does did have has had not no so can could would should will just
    about what which who how when where why there here up out into over than too very also any some all more most other
    such only own same s t don""".split()
)


def tokenize(text: str) -> list[str]:
    return [w for w in _TOKEN.findall(text.lower()) if w not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._tf = [Counter(d) for d in docs]
        self._len = [len(d) for d in docs]
        self._avg = (sum(self._len) / len(docs)) if docs else 0.0
        df = Counter(w for d in docs for w in set(d))
        n = len(docs)
        self._idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, dl in zip(self._tf, self._len):
            s = 0.0
            for w in query:
                f = tf.get(w)
                if f:
                    norm = 1 - self.b + self.b * (dl / self._avg if self._avg else 0)
                    s += self._idf[w] * f * (self.k1 + 1) / (f + self.k1 * norm)
            out.append(s)
        return out
