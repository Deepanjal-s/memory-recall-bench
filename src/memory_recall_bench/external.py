"""Benchmark retrieval systems written in any language, via files.

1. ``export_conversations(convs, "convs.jsonl")`` writes one JSON object per conversation:
   ``{"id", "question", "sessions": [{"id", "messages": [{"id", "role", "content"}]}]}``
   (messages are grouped into sessions when ids look like ``"<session>:<n>"``).
2. Your system indexes each conversation, runs the question, and writes a rankings file:
   ``{"<conversation id>": ["<message id>", ...best first...], ...}``
3. ``RankedFromFile("rankings.json")`` replays those rankings through the same
   recent-window, pairing and budget rules as the built-in retrievers.
"""
from __future__ import annotations

import json
from pathlib import Path

from .adapters import RankedRetriever
from .types import Conversation, Message


def export_conversations(conversations: list[Conversation], path: str | Path) -> None:
    # ensure_ascii escapes U+2028/U+2029, which some line readers (e.g. Node's readline)
    # treat as line breaks; newline="\n" avoids "\r\n" line endings on Windows.
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for c in conversations:
            sessions: dict[str, list[dict]] = {}
            for m in c.messages:
                sid = m.id.rsplit(":", 1)[0] if ":" in m.id else c.id
                sessions.setdefault(sid, []).append({"id": m.id, "role": m.role, "content": m.content})
            f.write(json.dumps({
                "id": c.id, "question": c.question, "kind": c.kind,
                "sessions": [{"id": sid, "messages": msgs} for sid, msgs in sessions.items()],
            }, ensure_ascii=True) + "\n")


class RankedFromFile(RankedRetriever):
    """Uses precomputed rankings for the given conversations.

    ``reset()``/``add()`` don't receive a conversation id, so the conversation is
    identified by (first message id, question). The first message alone isn't enough:
    LongMemEval haystacks are drawn from a shared pool of sessions."""

    def __init__(self, path: str | Path, conversations: list[Conversation], name: str = "external",
                 recent_share: float = 0.5, pair: bool = True):
        super().__init__(recent_share, pair)
        self.name = name
        rankings = json.loads(Path(path).read_text(encoding="utf-8"))
        self._by_key: dict[tuple[str, str], list[str]] = {}
        missing = []
        for c in conversations:
            if c.id not in rankings:
                missing.append(c.id)
                continue
            key = (c.messages[0].id, c.question)
            if key in self._by_key:
                raise ValueError(f"two conversations share first message and question: {c.id}")
            self._by_key[key] = rankings[c.id]
        if missing:
            raise KeyError(f"rankings file has no entry for {len(missing)} conversations, e.g. {missing[:3]}")

    def rank(self, query: str, archive: list[Message]) -> list[Message]:
        ranking = self._by_key[(self.messages[0].id, query)]
        by_id = {m.id: m for m in archive}
        return [by_id[i] for i in ranking if i in by_id]
