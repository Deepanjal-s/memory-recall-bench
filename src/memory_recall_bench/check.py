"""Retrieval regression checks for CI.

    mrb export --data longmemeval_s.json --limit 60 --out convs.jsonl
    # ... your system ranks each conversation's messages for its question -> rankings.json
    mrb check --data longmemeval_s.json --limit 60 --rankings rankings.json --budget 4k \\
              --min-recall 0.8 --baseline retrieval-baseline.json --max-drop 0.02

``check`` replays your rankings through the benchmark's budget rules and fails (exit code 1) when recall
is below ``--min-recall`` or more than ``--max-drop`` below the stored baseline. It lists the questions
that broke and the ones that got fixed, so a failing build says *which* facts your change lost.
``--write-baseline`` stores the current result for later runs. In GitHub Actions the report is also
written to the job summary.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .adapters import BUILTIN
from .datasets import load_longmemeval, synthetic
from .external import RankedFromFile, export_conversations
from .runner import run


def _budget(text: str) -> int:
    t = text.strip().lower()
    return int(float(t[:-1]) * 1000) if t.endswith("k") else int(t)


def _load(args) -> list:
    return load_longmemeval(args.data, limit=args.limit) if args.data else synthetic(n=args.limit or 20)


def export_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="mrb export", description="Write conversations as JSONL for an external system.")
    p.add_argument("--data", help="LongMemEval JSON file. Omit to use the built-in synthetic set.")
    p.add_argument("--limit", type=int)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    convs = _load(args)
    export_conversations(convs, args.out)
    print(f"wrote {len(convs)} conversations to {args.out}")
    return 0


def evaluate(convs, make, budget: int, metric: str) -> dict:
    results = run([make], convs, [budget])
    hits = sorted(r.conversation for r in results if getattr(r, metric))
    return {"budget": budget, "metric": metric, "n": len(results), "recall": len(hits) / len(results),
            "hits": hits}


def compare(current: dict, baseline: dict | None, min_recall: float | None, max_drop: float) -> tuple[bool, list[str]]:
    """Return (ok, report lines)."""
    lines = [f"**Retrieval check**: {current['metric']} at {current['budget']} tokens, "
             f"{current['n']} questions: **{current['recall']:.1%}**"]
    ok = True
    if min_recall is not None:
        passed = current["recall"] >= min_recall
        ok &= passed
        lines.append(f"- floor {min_recall:.1%}: {'pass' if passed else 'FAIL'}")
    if baseline is not None:
        if (baseline["budget"], baseline["metric"]) != (current["budget"], current["metric"]):
            raise ValueError("baseline was made with a different budget or metric")
        base_hits, now_hits = set(baseline["hits"]), set(current["hits"])
        broken, fixed = sorted(base_hits - now_hits), sorted(now_hits - base_hits)
        passed = current["recall"] >= baseline["recall"] - max_drop
        ok &= passed
        lines.append(f"- baseline {baseline['recall']:.1%} (allowed drop {max_drop:.1%}): "
                     f"{'pass' if passed else 'FAIL'}; {len(fixed)} fixed, {len(broken)} broken")
        if broken:
            lines.append("- broken: " + ", ".join(broken[:20]) + (" ..." if len(broken) > 20 else ""))
        if fixed:
            lines.append("- fixed: " + ", ".join(fixed[:20]) + (" ..." if len(fixed) > 20 else ""))
    return ok, lines


def check_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="mrb check", description="Fail when retrieval recall regresses.")
    p.add_argument("--data", help="LongMemEval JSON file. Omit to use the built-in synthetic set.")
    p.add_argument("--limit", type=int)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--rankings", help="Rankings JSON from your system: {conversation id: [message ids, best first]}")
    src.add_argument("--adapter", choices=sorted(BUILTIN), help="Check a built-in retriever instead (demo).")
    p.add_argument("--budget", default="4k")
    p.add_argument("--all-evidence", action="store_true", help="Require evidence from every answer session.")
    p.add_argument("--min-recall", type=float, help="Fail below this recall (0-1).")
    p.add_argument("--baseline", help="Baseline JSON written earlier with --write-baseline.")
    p.add_argument("--max-drop", type=float, default=0.0, help="Allowed drop below the baseline (0-1).")
    p.add_argument("--write-baseline", metavar="PATH", help="Store this run as the new baseline.")
    p.add_argument("--recent-share", type=float, default=0.5,
                   help="With --rankings: share of the budget kept for the most recent messages (default 0.5).")
    args = p.parse_args(argv)

    convs = _load(args)
    make = ((lambda: RankedFromFile(args.rankings, convs, name="yours", recent_share=args.recent_share))
            if args.rankings else BUILTIN[args.adapter])
    current = evaluate(convs, make, _budget(args.budget), "recalled_all" if args.all_evidence else "recalled")
    baseline = None
    if args.baseline and Path(args.baseline).exists():
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    elif args.baseline:
        print(f"note: baseline {args.baseline} not found; checking the floor only")
    ok, lines = compare(current, baseline, args.min_recall, args.max_drop)
    report = "\n".join(lines)
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(report + "\n")
    if args.write_baseline:
        Path(args.write_baseline).write_text(json.dumps(current, indent=1), encoding="utf-8")
        print(f"baseline written to {args.write_baseline}")
    return 0 if ok else 1
