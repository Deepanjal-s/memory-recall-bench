from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

from .types import Conversation, MemoryAdapter


class BudgetExceeded(AssertionError):
    pass


@dataclass
class Result:
    adapter: str
    budget: int
    conversation: str
    kind: str
    recalled: bool
    context_tokens: int
    evidence_age_tokens: int


def _instantiate(make):
    """Accept an adapter class, a zero-argument factory, or a ready instance."""
    if isinstance(make, type) or not hasattr(make, "reset"):
        return make()
    return make


def run(adapters: Iterable[Callable[[], MemoryAdapter] | MemoryAdapter], conversations: list[Conversation],
        budgets: Iterable[int], strict: bool = True) -> list[Result]:
    """Replay every conversation through every adapter at every budget.

    An adapter whose context exceeds the budget is a bug in the adapter (it would
    "win" by cheating), so with ``strict=True`` that raises :class:`BudgetExceeded`.
    """
    results: list[Result] = []
    budgets = list(budgets)
    for make in adapters:
        for budget in budgets:
            for conv in conversations:
                adapter = _instantiate(make)
                adapter.reset(budget)
                for m in conv.messages:
                    adapter.add(m)
                ids = list(adapter.context(conv.question))
                by_id = {m.id: m for m in conv.messages}
                unknown = [i for i in ids if i not in by_id]
                if unknown:
                    raise ValueError(f"{adapter.name} returned unknown message ids: {unknown[:3]}")
                used = sum(by_id[i].tokens for i in set(ids))
                if strict and used > budget:
                    raise BudgetExceeded(f"{adapter.name} used {used} tokens with a budget of {budget} "
                                         f"on conversation {conv.id}")
                results.append(Result(adapter.name, budget, conv.id, conv.kind,
                                      bool(conv.evidence_ids & set(ids)), used, conv.evidence_age_tokens()))
    return results


def summarize(results: list[Result], n_boot: int = 2000, seed: int = 0) -> list[dict]:
    """Recall per (adapter, budget) with a 95% bootstrap confidence interval."""
    rng = random.Random(seed)
    groups: dict[tuple[str, int], list[Result]] = {}
    for r in results:
        groups.setdefault((r.adapter, r.budget), []).append(r)
    rows = []
    for (adapter, budget), rs in groups.items():
        hits = [1.0 if r.recalled else 0.0 for r in rs]
        n = len(hits)
        boots = sorted(sum(rng.choice(hits) for _ in range(n)) / n for _ in range(n_boot))
        rows.append({
            "adapter": adapter, "budget": budget, "n": n, "recall": sum(hits) / n,
            "ci_low": boots[int(0.025 * n_boot)], "ci_high": boots[int(0.975 * n_boot) - 1],
            "mean_context_tokens": sum(r.context_tokens for r in rs) / n,
        })
    return sorted(rows, key=lambda r: (r["adapter"], r["budget"]))


def to_records(results: list[Result]) -> list[dict]:
    return [asdict(r) for r in results]
