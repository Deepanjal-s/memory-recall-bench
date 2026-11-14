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
pip install "memory-recall-bench[llamaindex]"
mrb --data longmemeval_s_cleaned.json --adapters fifo,llamaindex,recent+bm25 --by-kind
```

`llamaindex` wraps the real `llama_index.core.memory.Memory` (default configuration, no memory blocks), with the
benchmark's token counter passed as `tokenizer_fn` so both sides agree on sizes. Message ids are recovered by aligning
`Memory.get()` with the most recent messages and checking that their text matches; they are *not* stored in
`additional_kwargs`, because `Memory` counts those as tokens and that would change what's being measured.

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
pip install -e ".[dev,llamaindex,embeddings]"   # extras are optional; their tests skip without them
pytest
```

## Changelog

- **0.3.0**: `recalled_all` metric (evidence from every answer session), `--by-kind` and `--all-evidence` CLI options,
  full LongMemEval_s results in `results/`.
- **0.2.0**: real llama_index `Memory` adapter; the `flush` baseline is now an exact port of the library's queue logic
  (0.1.0 used a simplified model that emptied the whole history on each flush, which overstated the under-fill:
  46% → actually 55%).
- **0.1.0**: first release.

MIT licensed.
