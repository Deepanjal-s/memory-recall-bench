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

### Follow-up: an on-device re-ranker in the same app (all 479)

`mrb report` on the app's BM25 search showed that 153 of its 188 misses at 4k were *ranked too low* (evidence
found but cut by the budget), not *never retrieved*. So the next fix was precision: a small cross-encoder
(`Xenova/ms-marco-MiniLM-L-6-v2`, ~23 MB, ONNX on CPU via transformers.js) re-orders the top 30 hits inside the
app's own `search()` (`RERANKER=local`).

| | any evidence, 4k | all evidence, 2k | all evidence, 4k | all evidence, 8k | all evidence, 16k |
|---|---|---|---|---|---|
| BM25 (before) | 84.6 | 39.0 | 60.8 | 74.3 | 85.6 |
| + re-ranker | **91.0** | **43.4** | **70.1** | **83.7** | **88.9** |
| fixed / broken | 38 / 7 | 35 / 14 | 55 / 10 | 52 / 7 | 18 / 2 |

All differences in the all-evidence row are significant (sign test, p < 0.005). This is about the same gain as the
much larger `bge-reranker-v2-m3` on a GPU (63.0 → 71.4 at 4k, below), from a model that runs on a laptop. The cost
is latency: a median 3.3 s per search on a laptop CPU, so the app ships it switched off. Re-running `mrb report`
on the re-ranked search: misses at 4k fall from 188 to 143, and *ranked too low* from 153 to 111. It is still the
largest cause, so the next lever is re-ranking more than 30 candidates (at a latency cost). Lab repo
`kaggle/longmemeval-full/score_rerank.py`; rankings come from the app's `scripts/bench-rankings.ts --rerank`.

**How many candidates to re-rank.** The same model (original `cross-encoder/ms-marco-MiniLM-L-6-v2` weights, fp32 on
a Kaggle GPU) re-scoring the top N of the app's own BM25 rankings. Rankings stop at 200, so N=200 re-ranks
everything the search returns. At N=30 the GPU copy reproduces the app's q8 run within half a point (70.6 vs 70.1 at
4k), so the port is faithful. Latency is the re-ranker alone on the same laptop CPU (q8, 10 searches each).

| N | all evidence, 2k | 4k | 8k | 16k | vs N=30 at 4k (fixed / broken) | re-ranker latency, median |
|---|---|---|---|---|---|---|
| 30 | 43.4 | 70.6 | 84.1 | 89.1 | | 2.7 s |
| 50 | 44.1 | 71.6 | 85.2 | 90.6 | 6 / 1 (p=0.13) | 4.7 s |
| 100 | 44.3 | **72.0** | **85.8** | 91.2 | 8 / 1 (p=0.04) | 8.6 s |
| 200 | 44.3 | 72.0 | 85.8 | 91.4 | 8 / 1 | 17.6 s |

Diminishing returns: going from 30 to 100 candidates adds 1.4 points at 4k for about 3x the latency, and past 100
nothing changes. So depth is not what holds back the remaining *ranked too low* misses: of the 172 missing evidence
sessions that are in the ranking at all (N=30, 4k), 116 (67%) were already among the 30 candidates, and this small
re-ranker didn't score them high enough. Fusing the two orders doesn't help either: RRF
(k=60) of the BM25 order and the re-ranker order *lowers* all-evidence recall at 4k to 66.2 (N=30; 7 fixed / 28
broken vs plain re-ranking, p=0.0005), and N=50 or 100 doesn't rescue it (66.4). BM25's order is mostly what the
re-ranker is correcting, so mixing it back in undoes the fix.

**Which model.** A bigger re-ranker doesn't help either. Same top 30, six public cross-encoders, all with ONNX
ports the app could load (all-evidence recall; fixed / broken vs MiniLM-L-6 at 4k):

| model | params | 2k | 4k | 8k | 16k | vs L-6 at 4k | GPU time vs L-6 |
|---|---|---|---|---|---|---|---|
| `ms-marco-MiniLM-L-6-v2` (app default) | 23M | 43.4 | **70.6** | 84.1 | 89.1 | | 1x |
| `ms-marco-MiniLM-L-12-v2` | 33M | 42.2 | 69.5 | 83.5 | 89.4 | 7 / 12 | 2.0x |
| `mxbai-rerank-xsmall-v1` | 71M | 44.3 | 67.4 | 83.5 | 88.9 | 19 / 34 (p=0.05) | 1.7x |
| `mxbai-rerank-base-v1` | 184M | 42.6 | 68.1 | 83.5 | 88.9 | 33 / 45 | 3.4x |
| `bge-reranker-base` | 278M | 42.6 | 69.5 | **84.8** | 88.7 | 20 / 25 | 5.3x |
| `bge-reranker-v2-m3` (fp16) | 568M | 43.8 | **70.6** | 83.7 | 88.9 | 18 / 18 | 3.7x |
| *perfect re-ranker (evidence first)* | | *58.5* | *83.9* | *87.1* | *89.1* | | |

The headroom is real: a perfect re-ranker over the same 30 candidates would reach 83.9 at 4k, and every model
stops near 70 regardless of size. The models disagree on *which* questions they fix (bge-v2-m3: 18 / 18 against
L-6), so this looks like a ceiling of query-to-single-message relevance scoring, not of model capacity. A typical
miss: "Where did I complete my Bachelor's degree in Computer Science?", where the evidence is a message that starts
by weighing the cost of a graduate program and mentions the old degree only in passing. Closing the gap likely needs context the message alone doesn't carry (its session, neighbouring
turns), not a bigger generic re-ranker. For the app, the 23M default stays: nothing larger is better, and it is the
fastest.

Two context fixes, same top 30, MiniLM-L-6 unless noted (all-evidence at 4k vs 70.6 truncated to 2,000 chars): scoring
the best 1,200-char window of the whole message (MaxP; a third of candidates are longer than 2,000 chars) 69.7;
prefixing the previous turn 68.1; both 71.2 (16 / 13, n.s.); `bge-reranker-v2-m3` reading up to 6,000 chars 67.2
(14 / 30, p=0.02). So truncation isn't it either.

What does help a little is *whose* message it is. 94% of evidence messages are the user's own turns (842 of 896),
but 63% of the top-30 candidates are assistant replies, which re-rankers trained on web search score highly because
they are long and on-topic. Adding a constant to user turns (unless the question asks what the assistant said:
"you ... said / recommended / ...", 42 questions), with the constant picked on half the questions:

| all evidence | 2k | 4k | 8k | 16k |
|---|---|---|---|---|
| re-rank | 43.4 | 70.6 | 84.1 | 89.1 |
| + user-turn prior (B=3) | 44.5 | **72.9** | 84.1 | 88.3 |
| fixed / broken | 10 / 5 | 16 / 5 (p=0.03) | 6 / 6 | 2 / 6 |

On the held-out half alone: 72.8 → 75.7 at 4k (9 / 2, p=0.07). A small, budget-specific gain. It partly reflects how
LongMemEval labels evidence (on user turns by construction), so treat it as a hint for apps whose questions are about
the user, not a general rule. The gap to the 83.9 ceiling remains open, and it is concentrated: of the 64
questions between re-ranking and the perfect re-ranker at 4k, 27 are temporal-reasoning and 22 multi-session (both
need two or more sessions). Capping each session at K messages after re-ranking, to make room for a second session,
does **not** close it: on plain re-ranking the tuning half picks no cap (K=1 drops 68.3 to 57.1); on the user prior
K=2 costs 5 points at 8k and 16k (3 / 28, p<0.001). This matches the earlier session-diverse packing result. Lab repo
`kaggle/rerank-topn/` (`build_context_kernel.py`, `score_context.py`, `role_prior.py`, `session_cap.py`, summaries). Lab repo `kaggle/rerank-topn/` (`build_models_kernel.py`, `score_models.py`, `oracle.py`, summaries). The app keeps N=30 as the default; `RERANK_TOP_N=50` is a cheap
+1 point. Lab repo `kaggle/rerank-topn/` (kernel builder, scorer, `topn_summary.json`).

**Packing, not ranking.** Many of the remaining temporal misses had every evidence message in the top 5 (e.g. ranks
0 and 3), so the ranking was fine. The context is the problem. Every number on this page uses the bench's standard
packing: half the budget for the most recent messages (`recent_share=0.5`), and each hit brings its reply partner
(`pair=True`). On LongMemEval the question never concerns the latest session in particular, and assistant replies are
long, so at 4k only about 2,000 tokens (2 to 4 exchanges) go to retrieved hits. Same ranking (re-rank + user-turn prior),
different packing, all evidence:

| packing | 2k | 4k | 8k | 16k | temporal, 4k |
|---|---|---|---|---|---|
| recent 50%, pairs (standard) | 44.5 | 72.9 | 84.1 | 88.3 | 65.2 |
| recent 25%, pairs | 62.8 | 80.0 | 86.4 | 90.0 | 72.0 |
| **no recent window, pairs** | **68.3** | **83.1** | **87.5** | **91.2** | **76.5** |
| no recent window, hits only (no reply partner) | 83.5 | 84.3 | 85.6 | 88.5 | 83.3 |
| fixed / broken, no window + pairs vs standard | 115 / 1 | 50 / 1 | 16 / 0 | 14 / 0 | |

Dropping the recent window keeps whole exchanges and costs almost nothing: 1 question broken at 4k, none at 8k or 16k.
At 4k it reaches 83.1, about the 83.9 "perfect re-ranker" ceiling above under standard packing. Plain `recent+bm25`
shows the same effect (4k: 60.8 → 73.3 with no window; 16k: 82.3 → 87.3), so it isn't specific to the re-ranker.
Hits without their replies score higher still at 2k, but recall can't judge that: the reader then sees only the user's
side of each exchange. Two caveats. First, this is about LongMemEval: in a live chat the current conversation is what
the user is talking about, and an app should keep it (most send it separately from retrieved memory, which is what
`recent_share=0` models). Second, recall isn't accuracy (see the answer-accuracy section). An accuracy run of the
two pair packings at 4k and 8k is in progress (Kaggle kernel `lme-packing-accuracy`). Lab repo
`kaggle/rerank-topn/packing.py`, `packing_bm25.py`, summaries; `kaggle/packing-accuracy/`.

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

### Budget-aware expansion: protect the original ranking (all 479)

The 4k loss comes from sub-query matches displacing the original query's best messages. Fix: let the original
query's ranking fill the first fraction *f* of the archive budget (in tokens), then fill the rest from the fused
(original + sub-queries) ranking. No budget threshold is needed; the protected share scales with the budget.

Step 1, BM25, choosing *f* (local; paired vs recent+bm25, all-evidence):

| | 4k | 16k |
|---|---|---|
| recent+bm25 | 60.8 | 82.3 |
| gate (expand only at ≥16k) | 60.8 (0 / 0) | 83.9 (27 / 19) |
| prefix f=0.25 | 59.3 (12 / 19) | 84.1 (18 / 9) |
| **prefix f=0.5** | **61.2 (8 / 6)** | **84.3 (11 / 1, p≈0.006)** |
| prefix f=0.75 | 61.4 (4 / 1) | 83.5 (7 / 1) |

Step 2, the check: f=0.5 fixed in advance, on the embedding hybrid (Kaggle GPU; paired vs recent+hybrid):

| | any 4k | any 16k | all-evidence 4k | all-evidence 16k | multi-session all-ev 16k |
|---|---|---|---|---|---|
| recent+hybrid | 87.3 | 97.5 | 63.0 | 89.8 | |
| + prefix expansion f=0.5 | 87.9 (6 / 3) | 98.1 (4 / 1) | 62.2 (5 / 9, p≈0.42) | **91.6 (13 / 4, p≈0.05)** | 9 / 1 |

The protected prefix keeps the 16k gain of plain expansion (91.6 vs 91.4) and removes most of the 4k loss
(5 / 9 instead of 25 / 43). With a few variants tried on BM25 first, the hybrid run is the cleaner test, and it
lands at p≈0.05: a consistent improvement, not a large one. Recommendation for chat-memory retrievers that can
afford one small-LLM call per query: expand, but reserve half of the retrieval budget for the original query.
Lab repo `kaggle/query-expansion/`: `score_budget_aware.py` (BM25), `build_hybrid_prefix_kernel.py` (hybrid),
`score_csv.py` (paired scoring of a kernel CSV), `hybrid_prefix_out/hybrid_prefix_results.csv`.

## Answer accuracy: does better recall give better answers? (all 479, Kaggle GPU)

Recall only says the evidence was *in* the context. This run asks whether a reader then answers correctly.
For each question, Qwen3-4B-Instruct-2507 (fp16, greedy, one T4) answers from the context of two retrievers
at 4k and 16k. The same model then judges each answer with the LongMemEval judge prompts (yes/no, with the
per-type rules). A 4B judge is not GPT-4o, so absolute numbers are rough. Reader and judge are the same for
every arm, so the paired differences are the measurement. Identical contexts share one answer (1,714 distinct).

| | recall all-ev 4k | **accuracy 4k** | recall all-ev 16k | **accuracy 16k** |
|---|---|---|---|---|
| recent+hybrid | 63.0 | 59.1 | 89.8 | 58.9 |
| + prefix expansion f=0.5 | 62.2 | 60.8 (21 / 13, p≈0.23) | 91.6 | 58.7 (24 / 25, p=1.0) |

Accuracy split by whether all evidence was in the context:

| | 4k, all evidence | 4k, missing some | 16k, all evidence | 16k, missing some |
|---|---|---|---|---|
| recent+hybrid | 79.1 | 24.9 | 63.5 | 18.4 |
| + prefix expansion | 81.2 | 27.1 | 62.4 | 17.5 |

What it shows:

- **A 27-point recall gain bought no accuracy.** Going from 4k to 16k lifts all-evidence recall from 63.0 to
  89.8. Accuracy stays flat (59.1 → 58.9; 52 fixed / 53 broken).
- **The extra context costs the reader.** On the 302 questions where *both* the 4k and the 16k context hold
  all the evidence, accuracy falls from 79.1 to 70.9 (18 fixed / 43 broken, p≈0.002; prefix expansion:
  81.2 → 68.5, n=298). At 16k the
  reader sees ~65 messages (median) against ~18 at 4k. The distractors cost about as much as the extra
  evidence gains.
- **The +1.8 recall from budget-aware expansion does not show up as accuracy** at either budget (both n.s.).
- Multi-session (36–39%) and temporal reasoning (40–45%) stay low even with all evidence present. For a 4B
  reader those are reasoning failures, not retrieval failures.

Related work: Du et al., "Context Length Alone Hurts LLM Performance Despite Perfect Retrieval" (Findings of
EMNLP 2025) report the same effect on other tasks. This run replicates it for long-term chat memory, on all 479
LongMemEval_s evidence questions with paired tests.

Consequence for this benchmark: recall@budget is necessary but not sufficient. A retriever comparison should
report recall at the budget the reader actually works well at. For a small reader, 4k with a precise ranker
beats 16k with a recall-maximising one. Next candidates are precision at small budgets (a re-ranker on the
top candidates), or the same run with a stronger reader to see whether the 16k penalty is a small-model
effect. Lab repo `kaggle/answer-accuracy/`: `build_kernel.py` (kernel `lme-answer-accuracy`, private),
`score_answers.py`, `answer_accuracy_summary.json`.

## Re-ranking: precision at small budgets (all 479, Kaggle GPU)

A cross-encoder (`BAAI/bge-reranker-v2-m3`) re-scores the hybrid's top 100 messages for each question,
then the usual budget packing runs on the new order (`Rerank(..., top_n=100)`). The "rrf" arm fuses the
re-ranker order with the hybrid order instead of replacing it. Same Qwen3-4B reader and judge as above.

All-evidence recall, paired against recent+hybrid (fixed / broken, sign test):

| budget | recent+hybrid | + rerank | fixed / broken | + rerank rrf | fixed / broken |
|---|---|---|---|---|---|
| 2k | 36.5 | **44.3** | 50 / 13, p<0.001 | 40.9 | 32 / 11, p=0.002 |
| 4k | 63.0 | **71.4** | 58 / 18, p<0.001 | 70.1 | 44 / 10, p<0.001 |
| 8k | 79.5 | **85.8** | 43 / 13, p<0.001 | 83.3 | 23 / 5, p=0.001 |
| 16k | 89.8 | **92.5** | 23 / 10, p=0.035 | 91.4 | 15 / 7, p=0.13 |

Answer accuracy (judged):

| | 4k | 8k | 16k |
|---|---|---|---|
| recent+hybrid | 59.1 | 61.8 | 58.9 |
| + rerank | 62.2 (45 / 30, p=0.11) | **64.9** (39 / 24, p=0.08) | |
| + rerank rrf | 62.8 (38 / 20, p=0.025) | 62.4 | |

What it shows:

- **Re-ranking is the first change that moves recall a lot at small budgets.** +7.8 points at 2k and +8.4
  at 4k; multi-session all-evidence at 4k goes from 36.0 to 46.4. Plain replacement beats RRF fusion at
  every budget: the cross-encoder's order is better than the hybrid's, so mixing the hybrid back in dilutes it.
- **This time some of the recall turns into accuracy.** Re-ranked 8k (64.9) is the best configuration in
  this benchmark so far: better than the old best, recent+hybrid at 4k (63 fixed / 35 broken, p≈0.006),
  and better than recent+hybrid at 16k (59 / 30, p≈0.003). Single-budget gains over the hybrid are
  +3 points and borderline (p 0.03–0.11).
- **The distractor penalty is still there.** Accuracy when all evidence is present falls from 76.9 at 4k
  to 71.8 at 8k for the re-ranked contexts. Re-ranking wins by putting the evidence into fewer tokens, not
  by making the reader immune to noise.
- Cost: one cross-encoder pass over 100 messages per question: 47,900 pairs in 705 s on one T4, about
  1.5 s per question (fp16, max length 512).

Consequence: for chat memory with a small reader, spend compute on a re-ranker over a wide candidate set
and give the reader a mid-sized context (8k here), rather than widening the context. Lab repo
`kaggle/reranker/`: `build_kernel.py` (kernel `lme-reranker`, private), `score_recall.py`,
`score_answers.py`, `recall_summary.json`, `answer_accuracy_summary.json`. Public notebook (reproduces these tables
from the saved per-question outputs, CPU): [kaggle.com/code/deepanjalshukla95/more-context-didn-t-help-a-re-ranker-did](https://www.kaggle.com/code/deepanjalshukla95/more-context-didn-t-help-a-re-ranker-did).

## Inconclusive: a "stronger" reader (Qwen3-8B, 4-bit) and a judge audit

To test whether the 16k accuracy penalty is a small-model effect, the same recent+hybrid contexts (4k and
16k, identical to the answer-accuracy run) were answered by Qwen3-8B in NF4 4-bit (thinking off) and
judged by the same Qwen3-4B judge.

- **At 4k the 8B reader is worse, not better:** 54.1% vs 59.1% for Qwen3-4B-Instruct-2507 (31 fixed / 55
  broken, p≈0.013). It does not refuse more often (it refuses somewhat less), but multi-session drops
  from 36.0 to 24.8. So this model can't answer the question it was meant to answer. A newer, larger reader
  in 16-bit (which needs more than a free T4) is the real test.
- **The 16k half failed:** every 16k prompt ran out of GPU memory on the T4 and was recorded as `[OOM]`.
  Those rows are excluded; no 16k number is reported for the 8B.

**Judge audit (prompted by the failed run).** The judge answered "yes" to 204 of the 479 literal `[OOM]`
responses (109 of 132 temporal-reasoning questions), so a junk response string can fool it. To check that
this doesn't affect the real results, I hand-graded a random sample of 50 judged answers from the 4B run
(recent+hybrid at 4k, preference questions excluded) against the gold answers. I agreed with the judge on
47–48 of 50 (about 95%): one off-by-one day count wrongly marked wrong, one partial job title wrongly
marked right, one borderline case. On real refusals ("not mentioned"), the judge says yes about 10% of the
time, mostly where the refusal still contains part of the answer. The published accuracy numbers stand
with roughly ±2–3 points of judge noise. Paired comparisons are less affected, because both arms share it.
Future kernels drop non-answers like `[OOM]` before judging. Lab repo `kaggle/stronger-reader/`:
`build_kernel.py` (kernel `lme-stronger-reader`, private), `score_compare.py`.
