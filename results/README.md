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
messages fit, a weak ranking signal crowds out a strong one. n=30, so intervals are wide; the full run below
confirms the numbers (the preference rows are identical).

## Full run: BM25 vs embeddings vs hybrid (all 479 questions, GPU)

Same embedding setting as the pilot (`bge-small-en-v1.5`, first 512 characters), all 183,194 distinct messages,
embedded on a Kaggle T4 in 7 minutes. Notebook:
[Chat Memory Retrieval on LongMemEval](https://www.kaggle.com/code/deepanjalshukla95/chat-memory-retrieval-on-longmemeval);
per-question output: [`longmemeval_s_embeddings_gpu.csv`](longmemeval_s_embeddings_gpu.csv).

Evidence from **every** answer session kept (95% bootstrap CI):

| Adapter | 4k | 16k | 64k |
|---|---|---|---|
| recent+bm25 | 61% [56–65] | 82% [79–86] | 94% [91–96] |
| recent+embed | 59% [55–63] | 89% [86–92] | 99% [98–100] |
| **recent+hybrid** | **63%** [59–67] | **90%** [87–92] | **99%** [98–100] |

Per question against BM25 (all-evidence): hybrid at 16k fixes 37 and breaks 1; at 4k it fixes 39 and breaks 28.
Embedding alone at 4k fixes 57 but breaks 65, so it is not simply better: it wins on preference questions
(80% vs 47%) and multi-session any-evidence (86% vs 78%), and loses on knowledge-update all-evidence (64% vs 75%),
where the exact words of the updated fact matter.

What it means for an app: if you can afford one retriever, BM25 is the cheap strong baseline; adding embeddings
with rank fusion is the safe upgrade at 16k and above (almost no regressions). The hard cases left are
multi-session at 4k (35–36% all-evidence for every method), which needs something other than ranking single messages, such
as per-session summaries.

## Case study: finding and fixing a ranking bug in a real app

[ai-chat-memory](https://github.com/Deepanjal-s/ai-chat-memory) (Next.js + PostgreSQL) was scored through the
external-rankings interface: its own import pipeline and `search()` ran over every conversation
(`scripts/bench-rankings.ts` in that repo), and the rankings went through the same recent-window and budget rules.

**Finding:** its keyword search ranked with PostgreSQL `ts_rank_cd` plus a flat bonus for matching all terms. That has
no IDF, so it kept the answer in a 4k context far less often than plain BM25.

**Fix:** Postgres still selects candidates; BM25 (with document frequencies from the same filters) orders them.

| 479 questions | 4k any | 4k all | 16k all | 64k all |
|---|---|---|---|---|
| before (`ts_rank_cd`) | 56% | 28% | 65% | 95% |
| **after (BM25 re-rank)** | **85%** | **61%** | **86%** | **97%** |
| reference `recent+bm25` | 82% | 61% | 82% | 94% |

Paired, all-evidence at 4k: **163 questions fixed, 8 broken**. Per-question results:
[`ai_chat_memory_before_after.json`](ai_chat_memory_before_after.json).

### Follow-up: embedding unit size

With on-device embeddings (`bge-small-en-v1.5`) fused into the same BM25 search, the gain was smaller than this
benchmark's message-level retriever shows. The app embedded units of up to 4,000 characters, while bge-small reads
only ~512 tokens. Cutting units to 1,500 characters (60 questions: all 30 preference, 30 random multi-session, CPU):

| 60 questions | pref 4k | multi 4k all | pref 16k | multi 16k all |
|---|---|---|---|---|
| app keyword (BM25) | 60% | 30% | 80% | 70% |
| hybrid, 4,000-char units | 77% | 30% | 90% | 73% |
| **hybrid, 1,500-char units** | **83%** | **40%** | **93%** | **87%** |

Paired against 4,000-char units, all-evidence: 7 fixed / 2 broken at 4k, 5 / 0 at 16k. Small n; indicative only.
Summary: [`ai_chat_memory_chunk_size.json`](ai_chat_memory_chunk_size.json).

## mem0 (open-source `Memory`, retrieval only)

The `mem0` adapter stores each message verbatim with `add(..., infer=False)` and ranks with mem0 2.2.1's own
`search()`. That search is vector candidates boosted by BM25 and entity matches. The stack is on-device: fastembed
bge-small, local Qdrant and spaCy, with no LLM. mem0's default mode (LLM fact extraction) is **not** measured here. The
same 60 questions as above (all 30 preference, 30 random multi-session), CPU, about 64 s per question:

| 60 questions | 4k any | 4k all | 16k any | 16k all |
|---|---|---|---|---|
| `recent+bm25` (reference) | 63% | 38% | 88% | 70% |
| ai-chat-memory keyword (BM25) | 73% | 45% | 88% | 75% |
| **mem0, verbatim** | **83%** | **57%** | **95%** | **87%** |
| ai-chat-memory hybrid, 1,500-char units | 85% | 62% | 97% | 90% |

- Against `recent+bm25`, all-evidence: **12 fixed / 1 broken at 4k**, 11 / 1 at 16k. The embedding step helps,
  mostly on preference questions (4k: 47% → 73%).
- Against ai-chat-memory's hybrid it is roughly even: 4 fixed / 7 broken at 4k, 1 / 3 at 16k, all-evidence. With
  n = 60 the difference is not significant.
- Design note: mem0 takes candidates from the vector search only. BM25 re-scores those candidates but cannot add a
  message that matches only by keyword. ai-chat-memory fuses both lists (reciprocal rank fusion), so a keyword-only match
  can still get in.

Summary: [`mem0_subset.json`](mem0_subset.json). Scripts: `rank_mem0.py`, `score_mem0.py` in the lab repo
(`kaggle/longmemeval-full`).
