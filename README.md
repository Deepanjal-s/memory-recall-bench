# memory-recall-bench

[![CI](https://github.com/Deepanjal-s/memory-recall-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/Deepanjal-s/memory-recall-bench/actions/workflows/ci.yml) ![license](https://img.shields.io/badge/license-MIT-blue) ![dependencies](https://img.shields.io/badge/dependencies-0-brightgreen)

**Does your chatbot's memory keep the facts it will need?**

Chat-memory layers (sliding windows, flush-to-archive buffers, summarizers, vector stores) decide which old
messages the model sees. When they drop the one message that held the answer, the model can't answer, and nothing
in your test suite notices.

`memory-recall-bench` replays long conversations with known answer-bearing messages through **your** memory
implementation and reports **evidence recall** at each context budget: the share of questions for which the answer
message is still in the context. It runs offline and deterministically, needs no LLM calls, and is fast enough for CI.

```
$ mrb --data longmemeval_dev.json --budgets 2k,8k,64k
50 conversations. Evidence recall (95% CI):

adapter                        2k                8k               64k
fifo                  4% [0%-10%]       8% [2%-16%]     58% [44%-72%]
flush                  0% [0%-0%]       4% [0%-10%]     44% [30%-58%]
recent+bm25         86% [76%-94%]    98% [94%-100%]  100% [100%-100%]
recent+random         4% [0%-10%]       4% [0%-10%]     58% [44%-72%]
```

*(LongMemEval dev split, 50 questions. These numbers match an independent implementation in the
[Kaggle notebook](https://www.kaggle.com/code/deepanjalshukla95/what-does-your-chatbot-forget) exactly for FIFO, flush and BM25.
That split has only `single-session-user` questions, which flatter keyword retrieval; see Limitations.)*

## Results on the full benchmark

On all 479 labelled LongMemEval_s questions (~115k tokens of history each), at a 16k budget:

| | FIFO | llama_index `Memory` | Recent + BM25 |
|---|---|---|---|
| any answer message kept | 26% | 14% | **94%** |
| evidence from *every* answer session kept | 7% | 4% | **82%** |

Keyword retrieval is weakest on **preference** questions (77%) and **multi-session** questions needing several facts
(62% all-evidence). Full tables, per-type breakdown and raw per-question results are in [`results/`](results/).

### Real libraries, same questions

60 questions where keyword retrieval is weakest (all 30 preference questions, 30 random multi-session questions).
Evidence from every answer session kept:

| | 4k budget | 16k budget |
|---|---|---|
| FIFO | 3% | 5% |
| llama_index `Memory` (defaults, no memory blocks) | 0% | 5% |
| `recent+random` (control) | 5% | 12% |
| `recent+bm25` | 38% | 70% |
| mem0 2.2 search over verbatim messages (no LLM) | 57% | 87% |
| [ai-chat-memory](https://github.com/Deepanjal-s/ai-chat-memory) hybrid (BM25 + bge-small, RRF) | **62%** | **90%** |

mem0 and ai-chat-memory both use bge-small, and their difference is not significant at n = 60. This subset is
BM25's hardest. On **all 479 questions**, mem0 keeps all evidence in 89.6% at 16k (BM25 82.3%, hybrid 89.8%) and
60.8% at 4k (level with BM25). See [`results/`](results/README.md#mem0-on-all-479-questions).

Write-up of all findings: [What does your chatbot's memory actually keep?](docs/what-chat-memory-keeps.md) CIs and method: [`results/`](results/README.md#real-libraries-same-60-questions).

## Install

Not on PyPI yet. From GitHub (Python 3.10+; extras: `llamaindex`, `embeddings`, `mem0`):

```bash
pip install "memory-recall-bench @ git+https://github.com/Deepanjal-s/memory-recall-bench"
```

## Plug in your own memory

Implement three methods. Anything with them is a `MemoryAdapter`:

```python
from memory_recall_bench import Message, load_longmemeval, run, summarize, FIFO, RecentPlusRandom

class MyMemory:
    name = "my-memory"

    def reset(self, budget_tokens: int) -> None:   # new conversation, this context budget
        self.budget, self.store = budget_tokens, []

    def add(self, message: Message) -> None:       # messages arrive in order
        self.store.append(message)

    def context(self, query: str) -> list[str]:    # ids of messages you'd put in the prompt
        ...

convs = load_longmemeval("longmemeval_s.json")
print(summarize(run([MyMemory, FIFO, RecentPlusRandom], convs, budgets=[4000, 16000])))
```

The runner **rejects adapters that exceed the budget** (returning everything would trivially score 100%) and
ids that don't exist.

## Score real libraries

```bash
pip install "memory-recall-bench[llamaindex] @ git+https://github.com/Deepanjal-s/memory-recall-bench"
mrb --data longmemeval_s_cleaned.json --adapters fifo,llamaindex,recent+bm25 --by-kind
```

`llamaindex` wraps the real `llama_index.core.memory.Memory` (default configuration, no memory blocks), with the
benchmark's token counter passed as `tokenizer_fn` so both sides agree on sizes. Message ids are recovered by aligning
`Memory.get()` with the most recent messages and checking that their text matches; they are *not* stored in
`additional_kwargs`, because `Memory` counts those as tokens and that would change what's being measured.

### mem0

```bash
pip install "memory-recall-bench[mem0] @ git+https://github.com/Deepanjal-s/memory-recall-bench" && python -m spacy download en_core_web_sm
mrb --data longmemeval_s_cleaned.json --adapters recent+bm25,mem0 --by-kind
```

`mem0` measures the open-source `mem0.Memory` **retrieval**, not its LLM fact extraction: messages are stored verbatim
with `add(..., infer=False)` (no LLM call), and the archive is ranked by mem0's own `search()`. That is semantic
candidates from the vector store, boosted by mem0's BM25 and entity scores. The default stack runs on-device:
fastembed `BAAI/bge-small-en-v1.5`, local Qdrant and spaCy. Telemetry is switched off. Pass `config=` for a different
mem0 configuration. Indexing costs about 0.2 s per message on a laptop CPU, and each conversation is indexed once
for all budgets.

## Embedding and hybrid retrieval

```python
from memory_recall_bench.embeddings import EmbeddingCache, RecentPlusEmbedding, RecentPlusHybrid, fastembed_encoder

cache = EmbeddingCache(fastembed_encoder("BAAI/bge-small-en-v1.5"), max_chars=512)
qp = "Represent this sentence for searching relevant passages: "   # bge query instruction
results = run([lambda: RecentPlusEmbedding(cache, query_prefix=qp),
               lambda: RecentPlusHybrid(cache, query_prefix=qp)], convs, budgets=[4000, 16000])
cache.save("vectors.npz")   # embed once (e.g. on a GPU), reuse everywhere
```

- The encoder is any `list[str] -> (n, dim)` callable; `sentence_transformers_encoder()` is provided for GPUs.
- `EmbeddingCache` embeds each distinct text once and is shared across budgets and adapters.
- `RecentPlusHybrid` fuses BM25 and embedding rankings with reciprocal rank fusion (k=60).
- Subclass `RankedRetriever` and implement `rank(query, archive)` to benchmark your own retriever with the same
  recent-window and budget logic.

### Query expansion

```python
from memory_recall_bench.expansion import QueryExpansion, load_expansions

exp = load_expansions("expansions.jsonl")   # {"question": ..., "queries": [...]} per line, made by an LLM offline
results = run([lambda: QueryExpansion(RecentPlusHybrid(cache, query_prefix=qp), exp, protect=0.5)], convs, [4000, 16000])
```

`QueryExpansion` wraps any `RankedRetriever`: it ranks the archive for the question and for each sub-query, fuses the
rankings (RRF), and lets the question's own ranking fill the first `protect` share of the archive budget. With
`protect=0.5` on the hybrid: all-evidence 89.8 → 91.6 at 16k, about level at 4k (`results/README.md`).

### Second-stage re-ranking

```python
from memory_recall_bench.rerank import Rerank, cross_encoder_scorer

score = cross_encoder_scorer("BAAI/bge-reranker-v2-m3", device="cuda")   # or any score(query, texts) -> floats
cache = {}                                                               # shared, so each pair is scored once
results = run([lambda: Rerank(RecentPlusHybrid(emb, query_prefix=qp), score, top_n=100, cache=cache)],
              convs, [4000, 16000])
```

`Rerank` wraps any `RankedRetriever` and re-scores only its first `top_n` archive messages; the rest keep the
first-stage order. `fuse=True` blends the two orders with RRF instead of trusting the scorer alone.

> **Measured** (all 479, [results](results/#re-ranking-precision-at-small-budgets-all-479-kaggle-gpu)): on the
> hybrid's top 100, all-evidence recall at 4k goes from 63.0% to 71.4%, and judged answer accuracy peaks at
> 64.9% with an 8k context (vs 59.1% for the hybrid at 4k). Plain replacement beats `fuse=True`.

## Audit report: why does retrieval miss?

```bash
mrb report --data longmemeval_s.json --rankings rankings.json --name "our search" --out report.html
```

One self-contained HTML page: recall at each budget, recall per question type, and every question that misses
evidence at the focus budget (`--focus 4k`), labelled with the cause found in *your* ranking:
**never retrieved** (no needed session anywhere in the ranking: a first-stage recall problem), **ranked too
low** (found, but below what fits: a precision problem), and **incomplete** (some needed sessions in, others
out). Fixes are suggested in order of the measured split. On ai-chat-memory's BM25 search, 153 of 188 misses
at 4k were "ranked too low", which is what pointed to a re-ranker.

By default half of each budget holds the most recent messages (`--recent-share 0.5`). If your app sends the
live conversation separately, pass `--recent-share 0` (also on `mrb check`) so the whole budget goes to your
ranking. On LongMemEval that alone lifted all-evidence recall at 4k from 72.9% to 83.1% for the same ranking
([results](results/README.md)).

## Benchmark a system written in another language

> **Real use:** this interface found a ranking bug in a Next.js + PostgreSQL app's search and measured the fix
> (answer kept in a 4k context: 56% → 85% over 479 questions). See [`results/`](results/).


```python
from memory_recall_bench.external import export_conversations, RankedFromFile
export_conversations(convs, "convs.jsonl")     # 1. give your system the conversations
# 2. your system writes {"<conversation id>": ["<message id>", ...best first...]} to rankings.json
run([lambda: RankedFromFile("rankings.json", convs, name="my-app")], convs, budgets=[4000, 16000])
```

The rankings go through the same recent-window, pairing and budget rules as the built-in retrievers, so the
comparison is like-for-like. A missing conversation raises an error rather than silently scoring zero.

## Catch retrieval regressions in CI

A change to chunking, embeddings or ranking can silently drop facts your assistant needs. `mrb check` replays
your system's rankings through the same budget rules and fails when recall falls below a floor or below a
stored baseline. It names the questions that broke, so a red build says *which* facts the change lost.

```bash
mrb export --data longmemeval_s_cleaned.json --limit 60 --out convs.jsonl   # conversations for your system
# your system: rank each conversation's messages for its question -> rankings.json
mrb check --data longmemeval_s_cleaned.json --limit 60 --rankings rankings.json --budget 4k           --write-baseline retrieval-baseline.json                         # once; commit the file
mrb check --data longmemeval_s_cleaned.json --limit 60 --rankings rankings.json --budget 4k           --baseline retrieval-baseline.json --max-drop 0.02 --min-recall 0.8
```

As a GitHub Action (downloads and caches LongMemEval_s, writes the report to the job summary):

```yaml
- uses: Deepanjal-s/memory-recall-bench@v0.6.0
  with:
    rankings: rankings.json      # produced by your system in an earlier step
    limit: "60"
    budget: 4k
    baseline: retrieval-baseline.json
    max-drop: "0.02"
    report: "true"               # optional: HTML audit of every miss, uploaded as an artifact
```

`scripts/bm25_rankings.py` is a 20-line example of the "your system" step; this repository's CI runs the
Action end to end with it.

## Built-in baselines

| Adapter | Behaviour |
|---|---|
| `fifo` | Most recent messages that fit |
| `flush` | Dependency-free port of llama_index `Memory`'s queue logic. **Verified message-for-message against the real library** in CI (`tests/test_llamaindex.py`) |
| `flush-10pct` | Same, with the flush size scaled to 10% of the history budget |
| `recent+bm25` | Half the budget for recent messages, half for archived messages ranked by BM25 against the question |
| `recent+random` | **Control**: same split, random archive messages. If your retriever doesn't beat this, the gain is budget, not retrieval |

## How the measurements are checked

- FIFO has an exact closed form; the tests pin FIFO's exact output, and that test was checked by deliberately breaking FIFO.
- The `flush` baseline must produce exactly the same context as the real llama_index `Memory` (68/68 cases in development; enforced in CI).
- Recall is reported with 95% bootstrap confidence intervals, because small benchmarks are noisy.
- Token counts use a deterministic approximation (characters / 4) by default; pass your own counter to the loader.

## Finding: default flush settings waste half the context

With `token_limit=3644` (what llama_index chat engines use for a 3,900-token model) the history limit is about 2,551
tokens, smaller than the 3000-token default flush, so each flush throws away everything except the latest exchange.
Averaged over 50 LongMemEval conversations, the history is only **55% full**; a budget-scaled flush keeps it **87% full**
(see [llama_index#24117](https://github.com/run-llama/llama_index/issues/24117)). This is a regression test here
(`test_default_flush_size_underfills_small_budgets`).

## Data

Works with any split of [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (Wu et al., 2024, MIT), which marks
answer-bearing turns with `has_answer`. A small synthetic dataset is built in for tests and demos (`mrb` with no `--data`).

## Limitations

- **Recall, not accuracy:** having the answer in context is necessary, not sufficient.
- Questions whose words overlap the answer flatter keyword retrieval; report results per question type.
- The question is known when context is built (true for answering, not for proactive summarization).

## Development

```bash
pip install -e ".[dev,llamaindex,embeddings,mem0]"   # extras are optional; their tests skip without them
pytest
```

## Changelog

- **Unreleased**: `--recent-share` on `mrb report` and `mrb check` (how much of the budget the recent
  window takes); the report suggests it when misses are ranked too low.
- **0.6.0**: `mrb report`: one-page HTML retrieval audit (recall by budget and question type, every miss
  labelled never retrieved / ranked too low / incomplete, fixes ranked by the measured split); the Action
  uploads it as an artifact with `report: "true"`.
- **0.5.0**: `mrb check` / `mrb export` and a GitHub Action that fail CI when retrieval recall regresses;
  `Rerank` (second-stage re-ranking of a retriever's top N, any scorer, cross-encoder helper); `QueryExpansion`
  with a protected prefix; results on query expansion and on answer accuracy vs recall.
- **0.4.0**: `mem0` adapter (mem0's own search over verbatim messages, no LLM); embedding and hybrid retrievers
  (`[embeddings]` extra); `RankedFromFile` / `export_conversations` for systems in other languages; results for
  real libraries on the same 60 questions.
- **0.3.0**: `recalled_all` metric (evidence from every answer session), `--by-kind` and `--all-evidence` CLI options,
  full LongMemEval_s results in `results/`.
- **0.2.0**: real llama_index `Memory` adapter; the `flush` baseline is now an exact port of the library's queue logic
  (0.1.0 used a simplified model that emptied the whole history on each flush, which overstated the under-fill:
  46% → actually 55%).
- **0.1.0**: first release.

MIT licensed.
