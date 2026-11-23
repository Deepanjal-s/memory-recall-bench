"""Session-level ranking from per-session summaries.

Idea: a short summary of each past session tells the retriever *which* session talks about
the topic, even when no single message matches the query well. :class:`SessionBoost` wraps any
:class:`RankedRetriever` and fuses its message ranking with a ranking of the sessions' summaries:

    score(message) = 1/(k + message_rank) + weight/(k + rank of the message's session)

Summaries are produced offline (see ``kaggle/session-summaries`` in the lab repo) and passed in as
``{session_id: summary}``. Message ids must look like ``"<session_id>:<turn>"`` (what
:func:`load_longmemeval` produces). Sessions without a summary get no session term.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping

from .adapters import RankedRetriever
from .bm25 import BM25, tokenize
from .types import Message


def session_of(message_id: str) -> str:
    return message_id.rsplit(":", 1)[0]


def load_summaries(*paths: str | Path) -> dict[str, str]:
    """Read JSONL files of ``{"session_id": ..., "summary": ...}``; later files win. Empty summaries are skipped."""
    out: dict[str, str] = {}
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("summary", "").strip():
                out[row["session_id"]] = row["summary"].strip()
    return out


class SessionBoost(RankedRetriever):
    """Wrap a ranked retriever and add a session-summary term (reciprocal rank fusion)."""

    def __init__(self, base: RankedRetriever, summaries: Mapping[str, str], weight: float = 1.0,
                 k: int = 60, rank_sessions: Callable[[str, list[str]], list[float]] | None = None):
        super().__init__(base.recent_share, base.pair)
        self.base, self.summaries, self.weight, self.k = base, summaries, weight, k
        self.rank_sessions = rank_sessions or self._bm25_sessions
        self.name = f"{base.name}+sessions"

    @staticmethod
    def _bm25_sessions(query: str, texts: list[str]) -> list[float]:
        return BM25([tokenize(t) for t in texts]).scores(tokenize(query))

    def rank(self, query: str, archive: list[Message]) -> list[Message]:
        base_order = self.base.rank(query, archive)
        sids = sorted({session_of(m.id) for m in archive if session_of(m.id) in self.summaries})
        session_rank: dict[str, int] = {}
        if sids:
            scores = self.rank_sessions(query, [self.summaries[s] for s in sids])
            ordered = [s for sc, s in sorted(zip(scores, sids), key=lambda x: -x[0]) if sc > 0]
            session_rank = {s: r for r, s in enumerate(ordered)}
        score: dict[str, float] = {m.id: 1 / (self.k + r + 1) for r, m in enumerate(base_order)}
        for m in archive:
            r = session_rank.get(session_of(m.id))
            if r is not None:
                score[m.id] = score.get(m.id, 0.0) + self.weight / (self.k + r + 1)
        by_id = {m.id: m for m in archive}
        return [by_id[i] for i in sorted(score, key=lambda i: -score[i])]
