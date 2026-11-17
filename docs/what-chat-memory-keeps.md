# What does your chatbot's memory actually keep?

*Measuring FIFO windows, llama_index, mem0 and a custom app on LongMemEval, with no LLM in the loop.*

A chat assistant with "memory" has to decide which of thousands of past messages go into a few thousand tokens of
context. If the message that answers the user's question isn't in that context, no model can answer it. That
decision is a retrieval problem, and it can be measured without calling a model at all.

[memory-recall-bench](https://github.com/Deepanjal-s/memory-recall-bench) replays long conversations through a memory
implementation and checks one thing: **is the answer-bearing message in the context at a given token budget?** It uses
[LongMemEval](https://github.com/xiaowu0162/LongMemEval) (Wu et al., 2024), whose 500 questions come with about 115k
tokens of chat history each and labels for the turns that contain the answer.

## 1. Recency alone loses almost everything

On all 479 labelled questions, at a 16k-token budget (evidence from every answer session kept):

| Memory | 16k | 64k |
|---|---|---|
| FIFO (most recent messages that fit) | 7% | 35% |
| llama_index `Memory`, default configuration | 4% | 23% |
| recent + **random** archive messages (control) | 7% | 42% |
| recent + **BM25** archive messages | 82% | 94% |

Two lessons:

- Recency-only memory fails most questions, even with a large budget. The random control matches FIFO, so what
  helps is *retrieval*, not budget.
- llama_index's default `Memory` does worse than plain FIFO. Its flush moves a fixed 3,000 tokens to the archive,
  which leaves the active history half empty at small budgets (reported as
  [llama_index#24117](https://github.com/run-llama/llama_index/issues/24117)). Its vector memory block would add
  retrieval, but that needs an embedding model and isn't the default.

## 2. Keyword retrieval has two blind spots

BM25 is the cheap, strong baseline, but it breaks on two kinds of question:

- **Preferences.** A question like "Can you recommend some resources where I can learn more about video editing?"
  (a real LongMemEval question) rarely shares words with the earlier turn where the user's preference was stated.
  BM25: 47% at 4k.
- **Multi-session questions** that need several facts ("how many…", "what changed…"). BM25 keeps at least one
  answer turn 78% of the time at 4k, but *all* of them only 36%.

Embeddings (bge-small, on a GPU over 183k messages) fix the first: preference recall rises from 47% to 80% at 4k.
Fusing BM25 and embeddings with reciprocal rank fusion is the safe upgrade at 16k: 37 questions fixed, 1 broken.
At 4k it is a trade, not a strict improvement (39 fixed, 28 broken), because only a few messages fit.

## 3. Real libraries on the same questions

Sixty questions where keyword retrieval is weakest (all 30 preference questions and 30 random multi-session ones),
every system on the same data. Evidence from every answer session kept:

| | 4k | 16k |
|---|---|---|
| FIFO | 3% | 5% |
| llama_index `Memory` (defaults) | 0% | 5% |
| recent + BM25 | 38% | 70% |
| **mem0** 2.2 search over verbatim messages | 57% | 87% |
| **ai-chat-memory** hybrid (BM25 + bge-small, RRF) | 62% | 90% |

mem0 was measured in retrieval-only mode: each message stored verbatim with `add(..., infer=False)`, ranked by mem0's
own `search()`. That search uses semantic candidates, boosted by BM25 and entity matches. Its default LLM fact
extraction was not measured. One design difference shows in the code: mem0 takes its candidates **only** from the
vector search, and BM25 re-scores those candidates, so a message that matches only by keyword can't get in.
ai-chat-memory fuses both lists. At n = 60 the two are not significantly different; both clearly beat BM25 alone.

<!-- FULL-RUN: replace with the 479-question mem0 numbers when the Kaggle shards are downloaded -->

## 4. Measuring a real app found a real bug

The same harness accepts rankings from any system, in any language. Pointed at
[ai-chat-memory](https://github.com/Deepanjal-s/ai-chat-memory), a Next.js and PostgreSQL app, it found that keyword
search ranked with `ts_rank_cd`. That function has no inverse document frequency, so it favoured long messages full
of common words. With BM25 re-ranking on top of Postgres's candidate selection, recall at 4k went **56% → 85%** over
479 questions (163 fixed, 8 broken).

Two later findings came from the same loop:

- The app embedded 4,000-character chunks, but bge-small reads only about 512 tokens. Cutting chunks to 1,500
  characters raised multi-session all-evidence recall at 16k from 73% to 87%.
- A 36-question slice of the benchmark now runs in the app's CI. A change that orders results by date instead of
  BM25 drops recall from 89% to 78% and fails the build.

## What to take away

1. **Measure retrieval separately from generation.** Whether the evidence reached the context is a deterministic,
   cheap, CI-friendly number. Most memory failures happen there, before the model sees anything.
2. **Recency is not memory.** At realistic budgets a window keeps under 10% of the evidence a question needs.
3. **BM25 first, then embeddings with fusion.** BM25 is strong and cheap. Embeddings fix preference questions. Fusion
   avoids most regressions once the budget is 16k or more.
4. **The hard case is still open.** Multi-session questions at small budgets stay near 40% for every method, because
   ranking single messages can't gather several facts into 4k tokens. Per-session summaries are the next thing to
   test.

*Method notes: token counts are approximate (4 characters per token); confidence intervals are 95% bootstrap;
"fixed/broken" counts compare the same question under two systems. Everything is reproducible from the repository.*
