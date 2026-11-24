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

## Real libraries, same 60 questions

The same 60 questions as the mem0 section, all adapters in one run (`score_libraries.py` in the lab repo).
Percent with 95% bootstrap CI:

| | 4k any | 4k all | 16k any | 16k all |
|---|---|---|---|---|
| FIFO | 5 [0-12] | 3 [0-8] | 27 [17-38] | 5 [0-12] |
| llama_index `Memory` 0.14.26 (defaults) | 0 [0-0] | 0 [0-0] | 12 [5-20] | 5 [0-12] |
| `recent+random` (control) | 13 [5-22] | 5 [0-12] | 30 [18-42] | 12 [5-20] |
| `recent+bm25` | 63 [50-75] | 38 [27-50] | 88 [80-95] | 70 [58-82] |
| ai-chat-memory keyword | 73 [62-85] | 45 [33-57] | 88 [80-97] | 75 [63-85] |
| mem0 2.2.1, verbatim | 83 [73-92] | 57 [43-68] | 95 [88-100] | 87 [78-95] |
| ai-chat-memory hybrid | 85 [75-93] | 62 [50-73] | 97 [92-100] | 90 [82-97] |

llama_index `Memory` without memory blocks keeps only recent history, and its flush archives most of it when the
budget is small (llama_index#24117). Its retrieval features (vector memory block) need an embedding model and are
not measured here. Summary: [`libraries_subset.json`](libraries_subset.json).

## mem0 on all 479 questions

The same mem0 setup as above, run on 3 private Kaggle CPU kernels (about 160 questions each). It is compared with the
GPU run's retrievers on the same questions. Evidence from every answer session kept (95% CI for mem0):

| | 4k | 16k | 64k |
|---|---|---|---|
| recent+bm25 | 60.8% | 82.3% | 93.7% |
| recent+embed | 59.1% | 89.1% | 99.2% |
| recent+hybrid (BM25 + embed, RRF) | 63.0% | 89.8% | 99.2% |
| **mem0 2.2.1, verbatim** | 60.8% [56–65] | 89.6% [87–92] | 98.7% [98–100] |

Per question, all-evidence:

| mem0 vs | 4k fixed / broken | 16k fixed / broken |
|---|---|---|
| recent+bm25 | 44 / 44 | **42 / 7** |
| recent+hybrid | 23 / 34 | 14 / 15 |

- At 16k and above, mem0's search is as good as a BM25+embedding hybrid and clearly better than BM25 alone.
- At 4k it is no better than BM25 on all-evidence. It trades wins and losses there, like the pure embedding
  retriever, and is slightly behind the hybrid (23 fixed, 34 broken). The likely cause is the design noted above:
  mem0's candidates come only from the vector search, so keyword-only matches are lost. That is consistent with
  these numbers but not proven by them.
- The 60-question subset suggested a larger gap to BM25 at 4k (57% vs 38%). On all 479 the gap at 4k disappears,
  a reminder that a hand-picked hard subset overstates differences.
- The embedding inputs differ slightly: the GPU run embedded the first 512 characters of each message, while mem0
  embeds the whole message (bge-small truncates at 512 tokens).

Summary: [`mem0_full.json`](mem0_full.json). Scripts: `kaggle/mem0-full/` in the lab repo.

## Negative result: session-diverse packing

Idea: multi-session questions need one message from *each* answer session, so put the best message of each session
first and only then the rest. Tested as a re-ordering of BM25 and of mem0's rankings, on all 479 questions. Paired
all-evidence counts (fixed / broken) against the unmodified ranking:

| Diversify across | BM25 4k | BM25 16k | mem0 4k | mem0 16k |
|---|---|---|---|---|
| every session | 9 / 64 | 3 / 130 | 17 / 88 | 1 / 189 |
| top 5 sessions | 9 / 52 | 0 / 4 | 21 / 75 | 0 / 2 |
| top 3 sessions | 6 / 27 | 0 / 0 | 19 / 42 | 0 / 0 |
| top 2 sessions | 5 / 2 | 0 / 0 | 10 / 10 | 0 / 0 |

It never helps beyond noise, even on multi-session questions alone (every session, BM25 at 16k: 0 fixed, 39
broken). With ~50 sessions per conversation, a slot for each session's best message spends the budget on weak
matches from irrelevant sessions. A missing answer session is usually missing because its best message ranks low,
not because another session's messages crowd it out. Ranking single messages seems to have hit its limit here.
The next candidates change *what* is ranked: session-level summaries, or query expansion per sub-question.

Summaries: `session_diverse_summary_k{all,5,3,2}.json`. Script: `kaggle/mem0-full/session_diverse.py` in the lab repo.

## Negative result: session summaries (60-question pilot)

Idea: summarize each past session once (≤120 words, kept to facts) and use a ranking of the summaries to lift
the right sessions. Summaries: `Qwen/Qwen3-4B-Instruct-2507` on a free Kaggle T4 (2,746 of 2,747 sessions; one
failed to come back). `SessionBoost` (`memory_recall_bench.summaries`) fuses the BM25 message ranking with a BM25
ranking of the summaries (reciprocal rank fusion). 59 questions with every session summarized:

| Ranking | any 4k | any 16k | all-evidence 4k | all-evidence 16k |
|---|---|---|---|---|
| recent+bm25 | 62.7 | 88.1 | 37.3 | 69.5 |
| + sessions, weight 0.5 | 57.6 | 84.7 | 28.8 | 64.4 |
| + sessions, weight 1 | 57.6 | 81.4 | 30.5 | 64.4 |
| + sessions, weight 2 | 59.3 | 79.7 | 30.5 | 61.0 |
| + sessions, only re-order BM25 hits (best of 6) | 61.0 | 89.8 | 33.9 | 72.9 |

The plain wrapper is worse at every weight; the best variant is within noise (±12 points at n=59). The reason is
visible one level down. Rank of the first answer session among ~47:

| Sessions ranked by | top 1 | top 5 | top 10 |
|---|---|---|---|
| BM25 over the summary | 59% | 71% | 76% |
| best BM25 message in the session | 61% | 83% | 95% |

Squeezing a session into 120 words drops exactly the small details these questions ask about, so the summary is a
*worse* pointer to the session than the session's own best message. Removing the summaries' "no names, numbers…"
boilerplate changed nothing. Not tried: embedding the summaries, longer summaries, or a bigger model; given the
gap above, query expansion per sub-question looks like the better next step.

Scripts in the lab repo: `kaggle/session-summaries/` (`gpu/build_kernel.py`, `score_sessions.py`,
`variant_hits.py`, `diag_session_rank.py`).

## Mixed result: LLM query expansion (all 479 questions)

Each question rewritten into 2–4 short search queries (one per event or item it needs) by
`Qwen/Qwen3-4B-Instruct-2507` on a free Kaggle T4. Ranking: reciprocal rank fusion of BM25 on the original question
(weight w0) and BM25 on each sub-query. Paired against plain BM25 (fixed / broken; p = two-sided sign test):

| | any 4k | any 16k | all-evidence 4k | all-evidence 16k |
|---|---|---|---|---|
| recent+bm25 | 82.5 | 94.2 | 60.8 | 82.3 |
| + expansion, w0=2 | 84.1 | 95.0 | 57.6 | 84.8 |
| paired, w0=2 | 32 / 24 | 8 / 4 | 24 / 39 (p≈0.08) | 24 / 12 (p≈0.07) |
| multi-session only, paired | 12 / 9 | 2 / 2 | 4 / 9 | 14 / 5 (p≈0.06) |

With a large budget, sub-queries help multi-session questions find their second answer session (all-evidence
62.4 → 69.6 on multi-session at 16k). With a small budget, the extra matches crowd the window and all-evidence drops.
Neither effect clears p < 0.05, and BM25 + expansion at 16k (84.8) is still below the embedding hybrid (89.8), so
this is not a replacement for embeddings. Untested: expansion on top of the hybrid.

Lab repo `kaggle/query-expansion/`: `build_kernel.py` regenerates the expansions on Kaggle, `score_expansion.py` scores them, `expansion_summary.json` holds the numbers.

### Follow-up: expansion on top of the hybrid (all 479, Kaggle GPU)

Same expansions, now fused with the embedding hybrid (each sub-query ranked by BM25 + bge-small RRF). Paired against
the plain hybrid (fixed / broken, sign test):

| | all-evidence 4k | all-evidence 16k | multi-session all-evidence 16k |
|---|---|---|---|
| recent+hybrid | 63.0 | 89.8 | |
| + expansion, w0=1 | 59.3 (25 / 43, p≈0.04) | 91.2 (17 / 10) | 12 / 3 (p≈0.04) |
| + expansion, w0=2 | 61.0 (22 / 32) | 91.4 (17 / 9) | 13 / 4 (p≈0.05) |

The sign depends on the budget, as with BM25: at 16k the sub-queries recover the second answer session of
multi-session questions; at 4k they spend scarce budget on extra matches. With several comparisons run, the 16k
multi-session gain is suggestive, not proven. Practical reading: expand queries only when the budget is large
(≥16k), or give sub-query hits a share of the budget capped per session. Results:
`kaggle/query-expansion/hybrid_out/hybrid_expansion_results.csv` (lab repo).
