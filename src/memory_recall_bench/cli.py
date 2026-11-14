from __future__ import annotations

import argparse
import json
import sys

from .adapters import BUILTIN
from .datasets import load_longmemeval, synthetic
from .runner import run, summarize, to_records


def _budgets(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        part = part.strip().lower()
        out.append(int(float(part[:-1]) * 1000) if part.endswith("k") else int(part))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mrb", description="Measure whether chat memory keeps the facts it will need.")
    p.add_argument("--data", help="LongMemEval JSON file. Omit to use the built-in synthetic set.")
    p.add_argument("--limit", type=int, help="Use only the first N conversations.")
    p.add_argument("--budgets", default="2k,8k,32k", help="Comma-separated, e.g. 2k,8k,32000")
    p.add_argument("--adapters", default="fifo,flush,recent+random,recent+bm25",
                   help=f"Comma-separated from: {', '.join(BUILTIN)}")
    p.add_argument("--json", metavar="PATH", help="Also write per-conversation results as JSON.")
    args = p.parse_args(argv)

    unknown = [a for a in args.adapters.split(",") if a not in BUILTIN]
    if unknown:
        p.error(f"unknown adapters: {', '.join(unknown)}")

    convs = load_longmemeval(args.data, limit=args.limit) if args.data else synthetic(n=args.limit or 20)
    results = run([BUILTIN[a] for a in args.adapters.split(",")], convs, _budgets(args.budgets))
    rows = summarize(results)

    budgets = sorted({r["budget"] for r in rows})
    adapters = list(dict.fromkeys(r["adapter"] for r in rows))
    table = {(r["adapter"], r["budget"]): r for r in rows}
    width = max(len(a) for a in adapters) + 2
    print(f"{len(convs)} conversations. Evidence recall (95% CI):\n")
    print("adapter".ljust(width) + "".join(f"{b // 1000}k".rjust(18) for b in budgets))
    for a in adapters:
        cells = []
        for b in budgets:
            r = table[(a, b)]
            cells.append(f"{r['recall']:.0%} [{r['ci_low']:.0%}-{r['ci_high']:.0%}]".rjust(18))
        print(a.ljust(width) + "".join(cells))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"summary": rows, "results": to_records(results)}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
