"""Reference memory strategies, implemented as streaming adapters.

They double as baselines: run your own adapter next to them to see where it sits.
"""
from __future__ import annotations

import random
from collections import deque
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
    """llama_index-style ``Memory``: when the active history exceeds ``ratio * budget``,
    move the oldest messages out until at least ``flush_size`` tokens are gone.
    Flushed messages are not returned (no memory blocks configured).

    ``flush_size=None`` reproduces the library default of 3000 tokens (lowered only when
    it exceeds the budget). Pass e.g. ``flush_fraction=0.1`` to scale it to the budget.
    """

    def __init__(self, flush_size: int | None = None, flush_fraction: float | None = None, ratio: float = 0.7):
        self.flush_size, self.flush_fraction, self.ratio = flush_size, flush_fraction, ratio
        self.name = "flush" if flush_fraction is None else f"flush-{int(flush_fraction * 100)}pct"

    def reset(self, budget_tokens: int) -> None:
        self.limit = budget_tokens * self.ratio
        if self.flush_fraction is not None:
            size = int(self.limit * self.flush_fraction)
        else:
            size = self.flush_size if self.flush_size is not None else 3000
        self._flush = max(1, min(size, budget_tokens))
        self.active: deque[Message] = deque()
        self.used = 0
        self.fill_trace: list[int] = []

    def add(self, message: Message) -> None:
        self.active.append(message)
        self.used += message.tokens
        if self.used > self.limit:
            removed = 0
            while self.active and (removed < self._flush or self.used > self.limit):
                gone = self.active.popleft()
                removed += gone.tokens
                self.used -= gone.tokens
        self.fill_trace.append(self.used)

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


class RecentPlusBM25(_RecentPlusArchive):
    """Recent window + archived messages ranked by BM25 against the query."""

    name = "recent+bm25"

    def __init__(self, recent_share: float = 0.5, pair: bool = True):
        super().__init__(recent_share)
        self.pair = pair

    def context(self, query: str) -> Sequence[str]:
        window, archive, used = self._split()
        if not archive:
            return [m.id for m in window]
        scores = BM25([tokenize(m.content) for m in archive]).scores(tokenize(query))
        ranked = [m for s, m in sorted(zip(scores, archive), key=lambda x: -x[0]) if s > 0]
        if not self.pair:
            return self._fill(ranked, window, used)

        # Bring each hit's reply partner (user -> next message, assistant -> previous)
        # so retrieved exchanges stay readable. Groups that don't fit are skipped.
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


BUILTIN = {
    "fifo": FIFO,
    "flush": FlushToArchive,
    "flush-10pct": lambda: FlushToArchive(flush_fraction=0.1),
    "recent+bm25": RecentPlusBM25,
    "recent+random": RecentPlusRandom,
}
