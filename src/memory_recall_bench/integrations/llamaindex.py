"""Adapter for llama_index's ``Memory`` (``pip install memory-recall-bench[llamaindex]``).

How ids are recovered: without memory blocks, ``Memory``'s active history is always
the most recent contiguous run of messages (older ones are archived from the front).
So ``context()`` aligns the messages returned by ``Memory.get()`` with the last N
messages that were added, and checks that their text matches. Ids are *not* stored
in ``additional_kwargs``, because ``Memory`` counts those as tokens and that would
change the behaviour being measured.

The benchmark's token counter is passed as ``tokenizer_fn`` so both agree on sizes.
Memory blocks (vector, fact extraction) need an embedding model or an LLM and put
their content in a system message, so they are out of scope for this adapter.
"""
from __future__ import annotations

import itertools
from typing import Any, Sequence

from ..types import Message, TokenCounter, approx_tokens

_sessions = itertools.count()


class LlamaIndexMemory:
    """Wraps ``llama_index.core.memory.Memory`` (configured via ``Memory.from_defaults``)."""

    def __init__(self, count_tokens: TokenCounter = approx_tokens, name: str = "llamaindex-memory",
                 **memory_kwargs: Any):
        try:
            from llama_index.core.base.llms.types import ChatMessage, MessageRole
            from llama_index.core.memory import Memory
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise ImportError("Install the extra: pip install 'memory-recall-bench[llamaindex]'") from e
        if memory_kwargs.get("memory_blocks"):
            raise ValueError("memory_blocks are not supported by this adapter (see module docstring)")
        self._Memory, self._ChatMessage, self._Role = Memory, ChatMessage, MessageRole
        self._count = count_tokens
        self._kwargs = memory_kwargs
        self.name = name

    def reset(self, budget_tokens: int) -> None:
        count = self._count
        self.memory = self._Memory.from_defaults(
            session_id=f"mrb-{next(_sessions)}",
            token_limit=budget_tokens,
            tokenizer_fn=lambda text: [0] * count(text),
            **self._kwargs,
        )
        self.added: list[Message] = []

    def add(self, message: Message) -> None:
        role = self._Role.USER if message.role == "user" else self._Role.ASSISTANT
        self.memory.put(self._ChatMessage(role=role, content=message.content))
        self.added.append(message)

    def context(self, query: str) -> Sequence[str]:
        returned = [m for m in self.memory.get(input=query) if m.role in (self._Role.USER, self._Role.ASSISTANT)]
        if not returned:
            return []
        tail = self.added[-len(returned):]
        for got, expected in zip(returned, tail):
            if (got.content or "") != expected.content:
                raise RuntimeError("Memory.get() did not return a contiguous recent history; "
                                   "this adapter cannot attribute message ids")
        return [m.id for m in tail]
