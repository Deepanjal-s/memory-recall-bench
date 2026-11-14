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
    recalled: bool          # at least one evidence message in context
    recalled_all: bool      # at least one message from every evidence group in context
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
                kept = set(ids)
                results.append(Result(adapter.name, budget, conv.id, conv.kind,
                                      bool(conv.evidence_ids & kept),
                                      all(g & kept for g in conv.evidence_groups),
                                      used, conv.evidence_age_tokens()))
    return results


def summarize(results: list[Result], n_boot: int = 2000, seed: int = 0, by_kind: bool = False,
              metric: str = "recalled") -> list[dict]:
    """Recall per (adapter, budget[, kind]) with a 95% bootstrap confidence interval.

    ``metric="recalled_all"`` requires evidence from every evidence group (e.g. every
    answer session of a multi-session question).
    """
    if metric not in ("recalled", "recalled_all"):
        raise ValueError("metric must be 'recalled' or 'recalled_all'")
    rng = random.Random(seed)
    groups: dict[tuple, list[Result]] = {}
    for r in results:
        groups.setdefault((r.adapter, r.budget, r.kind if by_kind else "all"), []).append(r)
    rows = []
    for (adapter, budget, kind), rs in groups.items():
        hits = [1.0 if getattr(r, metric) else 0.0 for r in rs]
        n = len(hits)
        boots = sorted(sum(rng.choice(hits) for _ in range(n)) / n for _ in range(n_boot))
        rows.append({
            "adapter": adapter, "budget": budget, "kind": kind, "n": n, "recall": sum(hits) / n,
            "ci_low": boots[int(0.025 * n_boot)], "ci_high": boots[int(0.975 * n_boot) - 1],
            "mean_context_tokens": sum(r.context_tokens for r in rs) / n,
        })
    return sorted(rows, key=lambda r: (r["kind"], r["adapter"], r["budget"]))


def to_records(results: list[Result]) -> list[dict]:
    return [asdict(r) for r in results]
