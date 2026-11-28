from __future__ import annotations

import argparse
import json
import sys
from functools import partial

from .adapters import BUILTIN, RankedRetriever, _mem0, _RecentPlusArchive
from .datasets import load_longmemeval, synthetic
from .runner import run, summarize, to_records


def _budgets(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        part = part.strip().lower()
        out.append(int(float(part[:-1]) * 1000) if part.endswith("k") else int(part))
    return out


def recent_share_arg(text: str) -> float:
    message = "must be a number between 0 and 1 inclusive"
    try:
        share = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(message) from None
    if not 0 <= share <= 1:
        raise argparse.ArgumentTypeError(message)
    return share


def max_hits_arg(text: str) -> int:
    message = "must be a positive integer"
    try:
        hits = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(message) from None
    if hits < 1:
        raise argparse.ArgumentTypeError(message)
    return hits


def supports_max_hits(factory) -> bool:
    return factory is _mem0 or (isinstance(factory, type) and issubclass(factory, RankedRetriever))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["check"]:
        from .check import check_main
        return check_main(argv[1:])
    if argv[:1] == ["report"]:
        from .report import report_main
        return report_main(argv[1:])
    if argv[:1] == ["export"]:
        from .check import export_main
        return export_main(argv[1:])
    if argv[:1] == ["run"]:
        argv = argv[1:]
    p = argparse.ArgumentParser(epilog="Subcommands: 'mrb check --help' (CI regression check), "
                                       "'mrb report --help' (HTML audit), 'mrb export --help'.",
                                prog="mrb", description="Measure whether chat memory keeps the facts it will need.")
    p.add_argument("--data", help="LongMemEval JSON file. Omit to use the built-in synthetic set.")
    p.add_argument("--limit", type=int, help="Use only the first N conversations.")
    p.add_argument("--budgets", default="2k,8k,32k", help="Comma-separated, e.g. 2k,8k,32000")
    p.add_argument("--adapters", default="fifo,flush,recent+random,recent+bm25",
                   help=f"Comma-separated from: {', '.join(BUILTIN)}")
    p.add_argument("--recent-share", type=recent_share_arg,
                   help="Fraction of the budget reserved for recent messages in adapters with a recent window "
                        "(default: 0.5).")
    p.add_argument("--max-hits", type=max_hits_arg, metavar="N",
                   help="Pack at most N ranked hits in ranked adapters, plus their reply partners.")
    p.add_argument("--by-kind", action="store_true", help="One table per question type.")
    p.add_argument("--all-evidence", action="store_true",
                   help="Count a hit only if evidence from every answer session is in context.")
    p.add_argument("--json", metavar="PATH", help="Also write per-conversation results as JSON.")
    args = p.parse_args(argv)

    names = args.adapters.split(",")
    unknown = [a for a in names if a not in BUILTIN]
    if unknown:
        p.error(f"unknown adapters: {', '.join(unknown)}")

    factories = [BUILTIN[a] for a in names]
    ranked = {factory for factory in factories if supports_max_hits(factory)}
    if args.recent_share is not None:
        recent = {factory for factory in factories
                  if (isinstance(factory, type) and issubclass(factory, _RecentPlusArchive))
                  or (factory is _mem0 and args.max_hits is not None)}
        if not recent:
            print("Note: --recent-share has no effect because the selected adapters have no configurable recent window.",
                  file=sys.stderr)
    else:
        recent = set()
    settings = {}
    if args.max_hits is not None:
        if ranked:
            settings["max_hits"] = args.max_hits
            settings["recent_share"] = args.recent_share if args.recent_share is not None else 0.5
        else:
            print("Note: --max-hits applies only to ranked adapters.", file=sys.stderr)
    for i, factory in enumerate(factories):
        options = {}
        if factory in recent:
            options["recent_share"] = args.recent_share
        if args.max_hits is not None and factory in ranked:
            options["max_hits"] = args.max_hits
        if options:
            factories[i] = partial(factory, **options)

    convs = load_longmemeval(args.data, limit=args.limit) if args.data else synthetic(n=args.limit or 20)
    results = run(factories, convs, _budgets(args.budgets))
    metric = "recalled_all" if args.all_evidence else "recalled"
    rows = summarize(results, by_kind=args.by_kind, metric=metric)

    label = "All-evidence recall" if args.all_evidence else "Evidence recall"
    print(f"{len(convs)} conversations. {label} (95% CI):")
    if settings:
        print(f"Ranked packing: max_hits={settings['max_hits']}, recent_share={settings['recent_share']:g}")
    for kind in dict.fromkeys(r["kind"] for r in rows):
        sub = [r for r in rows if r["kind"] == kind]
        budgets = sorted({r["budget"] for r in sub})
        adapters = list(dict.fromkeys(r["adapter"] for r in sub))
        table = {(r["adapter"], r["budget"]): r for r in sub}
        width = max(len(a) for a in adapters) + 2
        if args.by_kind:
            print(f"\n[{kind}] n={sub[0]['n']}")
        else:
            print()
        print("adapter".ljust(width) + "".join(f"{b // 1000}k".rjust(18) for b in budgets))
        for a in adapters:
            cells = [f"{table[(a, b)]['recall']:.0%} [{table[(a, b)]['ci_low']:.0%}-{table[(a, b)]['ci_high']:.0%}]".rjust(18)
                     for b in budgets]
            print(a.ljust(width) + "".join(cells))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"summary": rows, "results": to_records(results), **settings}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
