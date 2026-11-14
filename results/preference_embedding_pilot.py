"""Local pilot: embedding vs BM25 on LongMemEval_s preference questions (bge-small, CPU)."""
import json, sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from memory_recall_bench import load_longmemeval, run, summarize, FIFO, RecentPlusBM25
from memory_recall_bench.embeddings import EmbeddingCache, RecentPlusEmbedding, RecentPlusHybrid, fastembed_encoder

data = Path(sys.argv[1]); kind = sys.argv[2]; out = Path(sys.argv[3])
convs = [c for c in load_longmemeval(data) if c.kind == kind]
cache = EmbeddingCache(fastembed_encoder(), max_chars=512)
cache_file = out.with_suffix(".npz")
if cache_file.exists():
    cache.load(cache_file)
t = time.time()
cache.get([m.content for c in convs for m in c.messages])
print(f"{kind}: {len(convs)} conversations, {len(cache)} vectors, embedded in {time.time()-t:.0f}s", flush=True)
cache.save(cache_file)
QP = "Represent this sentence for searching relevant passages: "
res = run([FIFO, RecentPlusBM25, lambda: RecentPlusEmbedding(cache, query_prefix=QP),
           lambda: RecentPlusHybrid(cache, query_prefix=QP)], convs, [4000, 16000])
rows = {m: summarize(res, metric=m) for m in ("recalled", "recalled_all")}
json.dump({"kind": kind, "max_chars": 512, "model": "BAAI/bge-small-en-v1.5", "summary": rows}, open(out, "w"), indent=1)
for m, rs in rows.items():
    for r in rs:
        print(f"{m:13} {r['adapter']:14} {r['budget']//1000:>3}k {r['recall']:.0%} [{r['ci_low']:.0%}-{r['ci_high']:.0%}]")
