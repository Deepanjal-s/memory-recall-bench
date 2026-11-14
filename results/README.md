# Results: LongMemEval_s (cleaned)

- Data: `longmemeval_s_cleaned.json` from [xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) (MIT).
  500 questions; the 479 with labelled evidence are used (abstention questions have none). ~115k tokens of history each.
- Run with memory-recall-bench 0.3.0, approximate token counts (chars/4), budgets 4k / 16k / 64k.
- Raw per-question results: [`longmemeval_s_v0.3.json`](longmemeval_s_v0.3.json) (adapter, budget, question, type, `recalled`, `recalled_all`, tokens used).

**Metrics:** `recalled` means at least one answer-bearing message is in context. `recalled_all` means at least one from **every**
answer session is in context (what a multi-session question actually needs).

## Overall (95% bootstrap CI)

| Adapter | 4k any | 4k all | 16k any | 16k all | 64k any | 64k all |
|---|---|---|---|---|---|---|
| fifo | 5% | 2% | 26% | 7% | 75% | 35% |
| flush (llama_index `Memory` default) | 1% | 0% | 14% | 4% | 58% | 23% |
| recent+random (control) | 5% | 2% | 27% | 7% | 75% | 42% |
| **recent+bm25** | **82%** [79–86] | **61%** [56–65] | **94%** [92–96] | **82%** [79–86] | **99%** | **94%** |

## By question type, 16k budget

| Type | n | fifo any | bm25 any | bm25 all |
|---|---|---|---|---|
| single-session-user | 64 | 17% | 100% | 100% |
| single-session-assistant | 56 | 16% | 100% | 100% |
| knowledge-update | 72 | 43% | 100% | 94% |
| temporal-reasoning | 132 | 21% | 89% | 80% |
| multi-session | 125 | 34% | 95% | **62%** |
| single-session-preference | 30 | 10% | **77%** | 77% |

## What it shows

1. Recency-only memory (FIFO, flush) fails most questions at realistic budgets; the random control matches FIFO, so
   retrieval, not budget, is what helps.
2. llama_index's default `Memory` does worse than plain FIFO at every budget, because the 3000-token flush keeps the history underfilled.
3. **Keyword retrieval breaks on two kinds of question:** preferences ("recommend me…" rarely shares words with where the
   preference was stated) and multi-session questions that need *several* facts. These are the next targets
   (embedding retrieval, per-session summaries).

Reproduce: `mrb --data longmemeval_s_cleaned.json --budgets 4k,16k,64k --by-kind [--all-evidence]` (about 3 minutes on a laptop).

## Pilot: embedding retrieval on preference questions (BM25's weakest type)

`single-session-preference`, all 30 questions, `BAAI/bge-small-en-v1.5` on CPU (first 512 characters of each message,
bge query instruction). Script: [`preference_embedding_pilot.py`](preference_embedding_pilot.py); output:
[`preference_embedding_pilot.json`](preference_embedding_pilot.json).

| Adapter | 4k | 16k |
|---|---|---|
| fifo | 7% | 10% |
| recent+bm25 | 47% [30–63] | 77% [60–90] |
| **recent+embed** | **80%** [63–93] | **93%** [83–100] |
| recent+hybrid (BM25 + embedding, RRF) | 67% [50–83] | 93% [83–100] |

Preference questions ("can you recommend…") share meaning but not words with where the preference was stated, so
embeddings fix BM25's weakest case. At the tight 4k budget, fusing in BM25 *hurts* (67% vs 80%): when only a few
messages fit, a weak ranking signal crowds out a strong one. n=30, so intervals are wide; the full 479-question run
with all retrievers is pending.
