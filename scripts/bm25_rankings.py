"""Example "external system" for the GitHub Action self-test: BM25 rankings for exported conversations.

    mrb export --data lme.json --limit 30 --out convs.jsonl
    python scripts/bm25_rankings.py convs.jsonl rankings.json
"""
import json
import sys

from memory_recall_bench.bm25 import BM25, tokenize

rankings = {}
for line in open(sys.argv[1], encoding="utf-8"):
    c = json.loads(line)
    msgs = [m for s in c["sessions"] for m in s["messages"]]
    scores = BM25([tokenize(m["content"]) for m in msgs]).scores(tokenize(c["question"]))
    rankings[c["id"]] = [m["id"] for s, m in sorted(zip(scores, msgs), key=lambda x: -x[0]) if s > 0]
json.dump(rankings, open(sys.argv[2], "w", encoding="utf-8"))
print(f"ranked {len(rankings)} conversations")
