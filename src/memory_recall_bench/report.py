"""A readable retrieval audit report (one self-contained HTML file) from your system's rankings.

    mrb report --data longmemeval_s.json --rankings rankings.json --name "our search" --out report.html

It replays the rankings through the benchmark's budget rules at several budgets, then diagnoses every question
that misses evidence at the focus budget (default 4k), using where the missing evidence sits in *your* ranking:

- **never retrieved**: no message from a needed session appears anywhere in the ranking. A recall problem: the
  first stage doesn't find it (hybrid search, embeddings, chunking, query rewriting help).
- **ranked too low**: the evidence is in the ranking but below what fits the budget. A precision problem
  (a re-ranker or better scoring helps; a bigger budget also does, but costs answer quality).
- **incomplete**: some needed sessions made it in and others didn't (typical for multi-session questions).
"""
from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path

from .check import _budget, _load
from .cli import recent_share_arg
from .external import RankedFromFile
from .runner import run

NEVER, LOW = "never retrieved", "ranked too low"


def diagnose(conv, ranking: list[str], kept: set[str]) -> dict:
    """Why `conv` misses evidence. `ranking` is the system's raw order, `kept` the ids that fit the budget."""
    pos = {mid: i for i, mid in enumerate(ranking)}
    groups = []
    for g in conv.evidence_groups:
        if g & kept:
            groups.append({"status": "kept"})
            continue
        ranks = sorted(pos[m] for m in g if m in pos)
        groups.append({"status": LOW if ranks else NEVER, "best_rank": ranks[0] + 1 if ranks else None,
                       "ids": sorted(g)})
    missing = [g for g in groups if g["status"] != "kept"]
    causes = {g["status"] for g in missing}
    cause = causes.pop() if len(causes) == 1 else "mixed"
    return {"cause": cause, "incomplete": len(missing) < len(groups), "groups": groups,
            "n_groups": len(groups), "n_missing": len(missing)}


def build(convs, rankings_path: str, budgets: list[int], focus: int, name: str = "your system",
          recent_share: float = 0.5) -> dict:
    rankings = json.loads(Path(rankings_path).read_text(encoding="utf-8"))
    make = lambda: RankedFromFile(rankings_path, convs, name=name, recent_share=recent_share)
    results = run([make], convs, budgets)
    curve = {}
    for b in budgets:
        rs = [r for r in results if r.budget == b]
        curve[b] = {"any": sum(r.recalled for r in rs) / len(rs), "all": sum(r.recalled_all for r in rs) / len(rs)}
    kinds = sorted({c.kind for c in convs})
    focus_rs = {r.conversation: r for r in results if r.budget == focus}
    per_kind = {}
    for k in kinds:
        rs = [focus_rs[c.id] for c in convs if c.kind == k]
        per_kind[k] = {"n": len(rs), "all": sum(r.recalled_all for r in rs) / len(rs)}

    # Re-run the focus budget to get the kept ids per conversation (Result doesn't store them).
    misses = []
    for c in convs:
        if focus_rs[c.id].recalled_all:
            continue
        a = make()
        a.reset(focus)
        for m in c.messages:
            a.add(m)
        kept = set(a.context(c.question))
        d = diagnose(c, rankings.get(c.id, []), kept)
        by_id = {m.id: m for m in c.messages}
        first_missing = next(g for g in d["groups"] if g["status"] != "kept")
        sample = by_id[first_missing["ids"][0]].content if first_missing.get("ids") else ""
        misses.append({"id": c.id, "kind": c.kind, "question": c.question, "answer": str(c.answer),
                       "evidence": sample[:300], **d})
    causes = Counter(m["cause"] for m in misses)
    return {"name": name, "n": len(convs), "budgets": budgets, "focus": focus, "curve": curve,
            "per_kind": per_kind, "misses": misses, "causes": dict(causes),
            "incomplete": sum(m["incomplete"] for m in misses), "recent_share": recent_share,
            "fixes": fixes(causes, len(misses), sum(m["incomplete"] for m in misses), recent_share)}


def fixes(causes: Counter, n_miss: int, n_incomplete: int, recent_share: float = 0.0) -> list[str]:
    """Ranked suggestions, from the measured split of causes. Wording reflects results in this repo's results/."""
    if not n_miss:
        return ["No misses at this budget. Add harder questions (multi-session, temporal) before optimising."]
    out = []
    share = lambda k: causes.get(k, 0) / n_miss
    items = [
        (share(LOW), f"{share(LOW):.0%} of misses are **ranked too low**: the evidence was found but didn't fit. "
                     "Add a cross-encoder re-ranker over the top 50-100 candidates (in memory-recall-bench's own LongMemEval run it lifted all-evidence "
                     "recall at 4k from 63% to 71%). Prefer this over a bigger budget: more context can lower "
                     "answer accuracy."),
        (share(NEVER), f"{share(NEVER):.0%} of misses are **never retrieved**: the first stage never finds the "
                       "evidence. Add the missing signal: BM25 if you only use embeddings, embeddings if you only "
                       "use keywords (fused with RRF), and check chunk size against your embedding model's limit."),
        (n_incomplete / n_miss, f"{n_incomplete / n_miss:.0%} of misses are **incomplete** (some needed sessions "
                                "in, others out). Typical of questions that combine facts across conversations; a "
                                "re-ranker helps most here, query expansion a little."),
    ]
    for _, text in sorted(items, key=lambda x: -x[0]):
        out.append(text)
    if recent_share > 0 and (share(LOW) > 0 or causes.get("mixed")):
        out.append(f"{recent_share:.0%} of the budget goes to the most recent messages, so ranked evidence competes "
                   "for the rest. If your app sends the live conversation separately, rerun with --recent-share 0: "
                   "on LongMemEval that lifted all-evidence recall at 4k from 72.9% to 83.1% with the same "
                   "ranking, though answer accuracy didn't change, so check answers too.")
    if causes.get("mixed"):
        out.append(f"{causes['mixed']} misses have mixed causes (one session never found, another ranked too low).")
    return out


def to_html(rep: dict, max_examples: int = 25) -> str:
    e = html.escape
    pct = lambda x: f"{100 * x:.1f}%"
    bar = lambda x: f'<span class="bar"><span style="width:{100 * x:.1f}%"></span></span>'
    curve = "".join(f"<tr><td>{b // 1000 if b % 1000 == 0 else b / 1000:g}k</td><td>{pct(v['any'])}</td>"
                    f"<td>{pct(v['all'])} {bar(v['all'])}</td></tr>" for b, v in rep["curve"].items())
    kinds = "".join(f"<tr><td>{e(k)}</td><td>{v['n']}</td><td>{pct(v['all'])} {bar(v['all'])}</td></tr>"
                    for k, v in sorted(rep["per_kind"].items(), key=lambda kv: kv[1]["all"]))
    n_miss = len(rep["misses"])
    causes = "".join(f"<tr><td>{e(c)}</td><td>{k}</td><td>{pct(k / n_miss)}</td></tr>"
                     for c, k in sorted(rep["causes"].items(), key=lambda x: -x[1])) if n_miss else ""
    fixes = "".join(f"<li>{e(f).replace('**', '')}</li>" for f in rep["fixes"])
    ex = []
    for m in sorted(rep["misses"], key=lambda m: (m["cause"] != LOW, m["kind"]))[:max_examples]:
        ranks = ", ".join(f"rank {g['best_rank']}" if g.get("best_rank") else "not in ranking"
                          for g in m["groups"] if g["status"] != "kept")
        ex.append(f"<details><summary><b>{e(m['cause'])}</b> · {e(m['kind'])} · {e(m['question'])}</summary>"
                  f"<p><b>Expected answer:</b> {e(m['answer'])}</p>"
                  f"<p><b>Missing sessions:</b> {m['n_missing']} of {m['n_groups']} ({e(ranks)})</p>"
                  f"<p><b>Evidence that should have been in context:</b> “{e(m['evidence'])}…”</p></details>")
    f = rep["focus"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Retrieval audit: {e(rep['name'])}</title><style>
:root{{--fg:#16181d;--mut:#5b6270;--line:#e3e6ea;--acc:#3b6fd8;--bg:#fff}}
@media (prefers-color-scheme:dark){{:root{{--fg:#e8eaee;--mut:#a0a7b4;--line:#2c313a;--acc:#7fa3e8;--bg:#14161a}}}}
body{{font:15px/1.5 system-ui,sans-serif;color:var(--fg);background:var(--bg);max-width:880px;margin:0 auto;padding:24px 16px}}
h1{{font-size:24px;margin:0}} h2{{font-size:17px;margin:28px 0 8px}} .mut{{color:var(--mut)}}
table{{border-collapse:collapse;width:100%}} td,th{{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line)}}
.bar{{display:inline-block;width:120px;height:8px;background:var(--line);border-radius:4px;vertical-align:middle;margin-left:6px}}
.bar span{{display:block;height:100%;background:var(--acc);border-radius:4px}}
details{{border:1px solid var(--line);border-radius:8px;padding:8px 12px;margin:6px 0}} summary{{cursor:pointer}}
.kpi{{font-size:32px;font-weight:700;color:var(--acc)}}</style></head><body>
<h1>Retrieval audit: {e(rep['name'])}</h1>
<p class="mut">{rep['n']} questions · generated by memory-recall-bench (mrb report) · focus budget {f // 1000:g}k tokens</p>
<p><span class="kpi">{pct(rep['curve'][f]['all'])}</span><br>of questions get <b>all</b> the evidence they need into a {f // 1000:g}k-token context.</p>
<h2>Recall by context budget</h2><table><tr><th>budget</th><th>any evidence</th><th>all evidence</th></tr>{curve}</table>
<h2>By question type ({f // 1000:g}k, all evidence; weakest first)</h2><table><tr><th>type</th><th>n</th><th>all evidence</th></tr>{kinds}</table>
<h2>Why questions miss ({n_miss} at {f // 1000:g}k)</h2><table><tr><th>cause</th><th>questions</th><th>share</th></tr>{causes}</table>
<h2>Suggested fixes, largest first</h2><ol>{fixes}</ol>
<h2>Examples ({min(n_miss, max_examples)} of {n_miss})</h2>{''.join(ex)}
<p class="mut">“Ranked too low” uses the position of the best missing evidence message in your ranking. Recall is
necessary but not sufficient: check answer accuracy too (see the benchmark's results page).</p></body></html>"""


def report_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="mrb report", description="Write an HTML retrieval audit from your rankings.")
    p.add_argument("--data", help="LongMemEval JSON file. Omit to use the built-in synthetic set.")
    p.add_argument("--limit", type=int)
    p.add_argument("--rankings", required=True, help="{conversation id: [message ids, best first]}")
    p.add_argument("--name", default="your system")
    p.add_argument("--budgets", default="2k,4k,8k,16k")
    p.add_argument("--focus", default="4k", help="Budget whose misses are diagnosed.")
    p.add_argument("--examples", type=int, default=25)
    p.add_argument("--out", default="retrieval-report.html")
    p.add_argument("--json", help="Also write the raw report data here.")
    p.add_argument("--recent-share", type=recent_share_arg, default=0.5,
                   help="Share of the budget kept for the most recent messages (default 0.5). 0 gives the whole "
                        "budget to your ranking, as when the app sends the live conversation separately.")
    args = p.parse_args(argv)
    budgets = sorted({_budget(b) for b in args.budgets.split(",")} | {_budget(args.focus)})
    rep = build(_load(args), args.rankings, budgets, _budget(args.focus), args.name, args.recent_share)
    Path(args.out).write_text(to_html(rep, args.examples), encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    c = rep["curve"][rep["focus"]]
    print(f"{args.name}: all-evidence recall {c['all']:.1%} at {rep['focus']} tokens; "
          f"{len(rep['misses'])} misses {rep['causes']}; report written to {args.out}")
    return 0
