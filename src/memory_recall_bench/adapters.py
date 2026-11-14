"""Reference memory strategies, implemented as streaming adapters.

They double as baselines: run your own adapter next to them to see where it sits.
"""
from __future__ import annotations

import random
from typing import Sequence

from .bm25 import BM25, tokenize
from .types import Message


class FIFO:
    """Keep the most recent messages that fit the budget."""

    name = "fifo"

    def reset(self, budget_tokens: int) -> None:
        self.budget = budget_tokens
        self.messages: list[Message] = []

    def add(self, message: Message) -> None:
        self.messages.append(message)

    def context(self, query: str) -> Sequence[str]:
        kept, used = [], 0
        for m in reversed(self.messages):
            if used + m.tokens > self.budget:
                break
            kept.append(m.id)
            used += m.tokens
        return kept


class FlushToArchive:
    """A dependency-free port of llama_index ``Memory``'s queue ("waterfall") logic,
    without memory blocks (flushed messages leave the context).

    Mirrors ``Memory._manage_queue`` in llama-index-core 0.14.x: when the active
    history exceeds ``ratio * budget``, oldest messages are flushed in chunks of at
    least ``flush_size`` tokens; at least one message is always kept; the remaining
    history must start with a user message; and if that would leave nothing, the last
    complete user turn is restored. ``tests/test_llamaindex.py`` checks that this
    produces exactly the same context as the real library.

    ``flush_size=None`` reproduces ``Memory.from_defaults``: 3000 tokens, or
    ``int(0.7 * budget)`` when 3000 exceeds the budget. ``flush_fraction=0.1`` instead
    scales the flush size to 10% of the history limit.
    """

    def __init__(self, flush_size: int | None = None, flush_fraction: float | None = None, ratio: float = 0.7):
        self.flush_size, self.flush_fraction, self.ratio = flush_size, flush_fraction, ratio
        self.name = "flush" if flush_fraction is None else f"flush-{int(flush_fraction * 100)}pct"

    def reset(self, budget_tokens: int) -> None:
        self.limit = budget_tokens * self.ratio
        if self.flush_fraction is not None:
            size = max(1, int(self.limit * self.flush_fraction))
        else:
            size = self.flush_size if self.flush_size is not None else 3000
            if size > budget_tokens:  # same rule as Memory.from_defaults
                size = int(budget_tokens * 0.7)
        self._flush = size
        self.active: list[Message] = []
        self.used = 0
        self.fill_trace: list[int] = []

    def add(self, message: Message) -> None:
        self.active.append(message)
        self._manage_queue()
        self.used = sum(m.tokens for m in self.active)
        self.fill_trace.append(self.used)

    def _manage_queue(self) -> None:
        queue = self.active
        tokens = sum(m.tokens for m in queue)
        if tokens <= self.limit:
            return
        rev = queue[::-1]  # newest first; pop() takes the oldest
        to_remove = tokens - self.limit
        archived = 0  # the library archives the N *oldest* stored rows each round
        while to_remove > 0:
            if len(rev) <= 1:
                break
            flushed: list[Message] = []
            flushed_tokens = 0
            while flushed_tokens < self._flush and rev and len(rev) > 1:
                m = rev.pop()
                flushed.append(m)
                flushed_tokens += m.tokens
            if not rev and flushed:
                rev.append(flushed.pop())
            chrono = rev[::-1]
            if chrono:
                while chrono and chrono[0].role != "user" and len(rev) > 1:
                    flushed.append(rev.pop())
                    chrono = rev[::-1]
                if (not rev or (len(rev) == 1 and rev[0].role != "user")) and flushed:
                    if rev and rev[0].role != "user":
                        flushed.append(rev.pop(0))
                    start = next((i for i in range(len(flushed) - 1, -1, -1) if flushed[i].role == "user"), -1)
                    if start >= 0:
                        turn = flushed[start:]
                        flushed = flushed[:start]
                        rev = turn[::-1] + rev
            archived += len(flushed)
            to_remove = sum(m.tokens for m in rev) - self.limit
            if not flushed:
                break
        self.active = queue[archived:]

    def context(self, query: str) -> Sequence[str]:
        return [m.id for m in self.active]


class _RecentPlusArchive:
    """Shared logic: a recent FIFO window plus a selection from the older archive."""

    def __init__(self, recent_share: float = 0.5):
        if not 0 <= recent_share <= 1:
            raise ValueError("recent_share must be between 0 and 1")
        self.recent_share = recent_share

    def reset(self, budget_tokens: int) -> None:
        self.budget = budget_tokens
        self.messages: list[Message] = []

    def add(self, message: Message) -> None:
        self.messages.append(message)

    def _split(self) -> tuple[list[Message], list[Message], int]:
        window, used = [], 0
        for m in reversed(self.messages):
            if used + m.tokens > self.budget * self.recent_share:
                break
            window.append(m)
            used += m.tokens
        recent_ids = {m.id for m in window}
        archive = [m for m in self.messages if m.id not in recent_ids]
        return window, archive, used

    def _fill(self, ranked: list[Message], window: list[Message], used: int) -> list[str]:
        kept = [m.id for m in window]
        for m in ranked:
            if used + m.tokens <= self.budget:
                kept.append(m.id)
                used += m.tokens
        return kept

    def _fill_pairs(self, ranked: list[Message], window: list[Message], used: int) -> list[str]:
        """Like _fill, but each hit brings its reply partner (user -> next message,
        assistant -> previous) so retrieved exchanges stay readable. Groups that don't
        fit are skipped."""
        position = {m.id: i for i, m in enumerate(self.messages)}
        kept = [m.id for m in window]
        kept_set = set(kept)
        for m in ranked:
            group = [m]
            i = position[m.id] + (1 if m.role == "user" else -1)
            if 0 <= i < len(self.messages):
                group.append(self.messages[i])
            new = [g for g in group if g.id not in kept_set]
            cost = sum(g.tokens for g in new)
            if not new or used + cost > self.budget:
                continue
            for g in new:
                kept.append(g.id)
                kept_set.add(g.id)
            used += cost
        return kept


class RankedRetriever(_RecentPlusArchive):
    """Base for retrieval strategies: a recent window plus the archive ranked by
    :meth:`rank`. Subclass and implement ``rank(query, archive) -> list[Message]``
    (best first; omit messages you would never retrieve)."""

    name = "ranked"

    def __init__(self, recent_share: float = 0.5, pair: bool = True):
        super().__init__(recent_share)
        self.pair = pair

    def rank(self, query: str, archive: list[Message]) -> list[Message]:  # pragma: no cover
        raise NotImplementedError

    def context(self, query: str) -> Sequence[str]:
        window, archive, used = self._split()
        if not archive:
            return [m.id for m in window]
        ranked = self.rank(query, archive)
        return self._fill_pairs(ranked, window, used) if self.pair else self._fill(ranked, window, used)


class RecentPlusBM25(RankedRetriever):
    """Recent window + archived messages ranked by BM25 against the query."""

    name = "recent+bm25"

    def rank(self, query: str, archive: list[Message]) -> list[Message]:
        scores = BM25([tokenize(m.content) for m in archive]).scores(tokenize(query))
        return [m for s, m in sorted(zip(scores, archive), key=lambda x: -x[0]) if s > 0]


class RecentPlusRandom(_RecentPlusArchive):
    """Control: same budget split as RecentPlusBM25 but random archive selection.
    If your retriever doesn't beat this, its gain comes from budget, not retrieval."""

    name = "recent+random"

    def __init__(self, recent_share: float = 0.5, seed: int = 0):
        super().__init__(recent_share)
        self.seed = seed

    def context(self, query: str) -> Sequence[str]:
        window, archive, used = self._split()
        rng = random.Random(f"{self.seed}:{query}")  # stable across processes
        shuffled = archive[:]
        rng.shuffle(shuffled)
        return self._fill(shuffled, window, used)


def _llamaindex():
    from .integrations.llamaindex import LlamaIndexMemory  # needs the [llamaindex] extra
    return LlamaIndexMemory()


BUILTIN = {
    "fifo": FIFO,
    "flush": FlushToArchive,
    "flush-10pct": lambda: FlushToArchive(flush_fraction=0.1),
    "recent+bm25": RecentPlusBM25,
    "recent+random": RecentPlusRandom,
    "llamaindex": _llamaindex,
}
