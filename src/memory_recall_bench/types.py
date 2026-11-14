from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence, runtime_checkable

TokenCounter = Callable[[str], int]


def approx_tokens(text: str) -> int:
    """Deterministic, dependency-free token estimate (~4 characters per token)."""
    return max(1, math.ceil(len(text) / 4))


@dataclass(frozen=True)
class Message:
    id: str
    role: str
    content: str
    tokens: int


@dataclass
class Conversation:
    """A long conversation followed by one question whose answer is in `evidence_ids`."""

    id: str
    messages: list[Message]
    question: str
    answer: str
    evidence_ids: frozenset[str]
    kind: str = "unknown"
    meta: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return sum(m.tokens for m in self.messages)

    def evidence_age_tokens(self) -> int:
        """Tokens from just after the last evidence message to the end."""
        last = max(i for i, m in enumerate(self.messages) if m.id in self.evidence_ids)
        return sum(m.tokens for m in self.messages[last + 1:])


@runtime_checkable
class MemoryAdapter(Protocol):
    """Wrap any chat-memory implementation in these three methods.

    - ``reset(budget_tokens)``: start a new conversation with this context budget.
    - ``add(message)``: a new message arrives (called in conversation order).
    - ``context(query)``: return the ids of the messages that would be placed in the
      model's context to answer ``query``. Their total tokens must fit the budget.
    """

    name: str

    def reset(self, budget_tokens: int) -> None: ...

    def add(self, message: Message) -> None: ...

    def context(self, query: str) -> Sequence[str]: ...
