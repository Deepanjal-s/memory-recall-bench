"""Dataset loaders. Each returns a list of :class:`Conversation`."""
from __future__ import annotations

import json
import random
from pathlib import Path

from .types import Conversation, Message, TokenCounter, approx_tokens


def load_longmemeval(path: str | Path, count_tokens: TokenCounter = approx_tokens,
                     limit: int | None = None) -> list[Conversation]:
    """Load a LongMemEval JSON file (any split: dev, _s, _m, _oracle).

    LongMemEval (Wu et al., 2024, MIT) marks answer-bearing turns with ``has_answer``.
    Questions with no marked turn (e.g. abstention questions) are skipped. Evidence is
    grouped by session: a multi-session question needs evidence from every answer session.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[Conversation] = []
    for q in data:
        messages, evidence = [], set()
        groups: dict[str, set[str]] = {}
        for s_idx, (sid, session) in enumerate(zip(q["haystack_session_ids"], q["haystack_sessions"])):
            for t_idx, turn in enumerate(session):
                mid = f"{sid}:{t_idx}" if sid else f"s{s_idx}:{t_idx}"
                messages.append(Message(mid, turn["role"], turn["content"], count_tokens(turn["content"])))
                if turn.get("has_answer"):
                    evidence.add(mid)
                    groups.setdefault(f"{sid or s_idx}", set()).add(mid)
        if not evidence:
            continue
        out.append(Conversation(
            id=q["question_id"], messages=messages, question=q["question"], answer=str(q["answer"]),
            evidence_ids=frozenset(evidence), kind=q.get("question_type", "unknown"),
            evidence_groups=tuple(frozenset(g) for g in groups.values()),
        ))
        if limit and len(out) >= limit:
            break
    return out


_FACTS = [
    ("my sister's name", "Priya", "What is my sister's name?"),
    ("the city I moved to", "Pune", "Which city did I move to?"),
    ("my laptop model", "ThinkPad X1", "What laptop model do I have?"),
    ("my favourite dish", "momos", "What is my favourite dish?"),
    ("the language I'm learning", "Rust", "Which programming language am I learning?"),
    ("my dog's name", "Bruno", "What is my dog's name?"),
]
_FILLER = [
    "Can you explain how binary search works on a rotated array?",
    "Give me three tips for writing clearer commit messages.",
    "What's the difference between a process and a thread?",
    "Suggest a weekend hiking plan near the mountains.",
    "How do I make a REST API idempotent?",
    "Summarise the causes of the 2008 financial crisis briefly.",
]


def synthetic(n: int = 20, turns: int = 120, seed: int = 0,
              count_tokens: TokenCounter = approx_tokens) -> list[Conversation]:
    """Small deterministic dataset for tests and demos: filler chat with one planted fact."""
    rng = random.Random(seed)
    convs = []
    for c in range(n):
        label, value, question = _FACTS[c % len(_FACTS)]
        at = rng.randrange(0, turns)
        msgs, evidence = [], set()
        for i in range(turns):
            if i == at:
                text = f"By the way, {label} is {value}. Anyway, {rng.choice(_FILLER).lower()}"
            else:
                text = rng.choice(_FILLER) + " " + " ".join(rng.choice(["detail", "context", "example", "note"])
                                                            for _ in range(rng.randrange(10, 60)))
            role = "user" if i % 2 == 0 else "assistant"
            mid = f"c{c}:{i}"
            msgs.append(Message(mid, role, text, count_tokens(text)))
            if i == at:
                evidence.add(mid)
        convs.append(Conversation(f"syn-{c}", msgs, question, value, frozenset(evidence), kind="synthetic"))
    return convs
