"""Query expansion with a protected prefix.

An LLM rewrites each question into a few short sub-queries (one per event or item it needs). The base
retriever ranks the archive once for the question and once per sub-query, and the rankings are fused
(reciprocal rank fusion). Plain fusion helps at large budgets (multi-session questions find their second
answer session) but hurts at small ones, where sub-query matches push out the question's own best hits.
:class:`QueryExpansion` therefore lets the question's own ranking fill the first ``protect`` share of the
archive budget (in tokens) and fills the rest from the fused ranking. On LongMemEval_s with the hybrid,
``protect=0.5`` kept the 16k gain (all-evidence 89.8 -> 91.6) and removed most of the 4k loss
(see results/README.md).

Expansions are produced offline (``kaggle/query-expansion`` in the lab repo) and passed in as
``{question: [sub-query, ...]}``. Questions without an entry are ranked by the base retriever alone.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from .adapters import RankedRetriever
from .types import Message


def load_expansions(*paths: str | Path) -> dict[str, list[str]]:
    """Read JSONL files of ``{"question": ..., "queries": [...]}``; later files win. Torn lines are skipped."""
    out: dict[str, list[str]] = {}
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qs = [q.strip() for q in row.get("queries", []) if isinstance(q, str) and q.strip()]
            if qs:
                out[row["question"]] = qs
    return out


class QueryExpansion(RankedRetriever):
    """Wrap a ranked retriever; fuse its rankings for the query and each sub-query, with a protected prefix."""

    def __init__(self, base: RankedRetriever, expansions: Mapping[str, Sequence[str]], protect: float = 0.5,
                 k: int = 60):
        if not 0 <= protect <= 1:
            raise ValueError("protect must be between 0 and 1")
        super().__init__(base.recent_share, base.pair, base.max_hits)
        self.base, self.expansions, self.protect, self.k = base, expansions, protect, k
        self.name = f"{base.name}+expansion" + (f" protect={protect:g}" if protect else "")

    def rank(self, query: str, archive: list[Message]) -> list[Message]:
        orig = self.base.rank(query, archive)
        subs = self.expansions.get(query, ())
        if not subs:
            return orig
        score: dict[str, float] = {}
        for ranked in [orig] + [self.base.rank(s, archive) for s in subs]:
            for r, m in enumerate(ranked):
                score[m.id] = score.get(m.id, 0.0) + 1 / (self.k + r + 1)
        by_id = {m.id: m for m in archive}
        fused = [by_id[i] for i in sorted(score, key=lambda i: -score[i])]
        budget = getattr(self, "budget", None)
        if not self.protect or budget is None:
            return fused
        cap, used, head = budget * (1 - self.recent_share) * self.protect, 0, []
        for m in orig:
            if used + m.tokens > cap:
                break
            head.append(m)
            used += m.tokens
        seen = {m.id for m in head}
        return head + [m for m in fused if m.id not in seen]
