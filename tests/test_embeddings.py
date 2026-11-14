"""Embedding retrievers, tested with a tiny deterministic encoder (no model download)."""
import pytest

np = pytest.importorskip("numpy")

from memory_recall_bench import FIFO, run, summarize, synthetic  # noqa: E402
from memory_recall_bench.embeddings import EmbeddingCache, RecentPlusEmbedding, RecentPlusHybrid  # noqa: E402

VOCAB = ["sister", "priya", "sibling", "city", "pune", "moved", "laptop", "thinkpad", "computer", "dish", "momos",
         "food", "rust", "language", "learning", "dog", "bruno", "pet", "name"]
SYNONYMS = {"sibling": "sister", "computer": "laptop", "food": "dish", "pet": "dog"}


def toy_encoder(texts):
    """Bag-of-words over a tiny vocabulary, with synonyms mapped together, so a
    paraphrased query can match without sharing a surface word."""
    calls.append(len(texts))
    out = np.zeros((len(texts), len(VOCAB) + 1), dtype=np.float32)  # last column: bias only
    for i, t in enumerate(texts):
        for w in t.lower().replace("'", " ").replace("?", " ").replace(".", " ").split():
            w = SYNONYMS.get(w, w)
            if w in VOCAB:
                out[i, VOCAB.index(w)] += 1
        out[i, -1] += 0.01  # dedicated bias dimension, so no text is an all-zero vector
    return out


calls: list[int] = []
CONVS = synthetic(n=12, turns=120, seed=5)


def test_cache_embeds_each_text_once_and_normalises():
    calls.clear()
    cache = EmbeddingCache(toy_encoder)
    v1 = cache.get(["my dog Bruno", "my dog Bruno", "pune city"])
    v2 = cache.get(["my dog Bruno"])
    assert sum(calls) == 2  # duplicate and repeat served from cache
    assert np.allclose(np.linalg.norm(v1, axis=1), 1)
    assert np.allclose(v1[0], v2[0])


def test_cache_rejects_bad_encoder_shapes():
    cache = EmbeddingCache(lambda texts: np.zeros(len(texts)))
    with pytest.raises(ValueError, match="shape"):
        cache.get(["a", "b"])


def test_cache_round_trips_to_disk(tmp_path):
    cache = EmbeddingCache(toy_encoder)
    cache.get(["sister priya", "dog bruno"])
    cache.save(tmp_path / "c.npz")
    calls.clear()
    other = EmbeddingCache(toy_encoder).load(tmp_path / "c.npz")
    assert np.allclose(other.get(["dog bruno"]), cache.get(["dog bruno"]))
    assert calls == []  # loaded, not re-encoded


@pytest.mark.parametrize("cls", [RecentPlusEmbedding, RecentPlusHybrid])
def test_embedding_retrievers_find_planted_facts_within_budget(cls):
    cache = EmbeddingCache(toy_encoder)
    res = run([lambda: cls(cache)], CONVS, [1500])
    assert all(r.context_tokens <= r.budget for r in res)
    recall = summarize(res)[0]["recall"]
    fifo = summarize(run([FIFO], CONVS, [1500]))[0]["recall"]
    assert recall >= 0.9 > fifo


def test_embedding_finds_a_paraphrase_that_bm25_cannot():
    from memory_recall_bench import Conversation, Message, RecentPlusBM25
    filler = [Message(f"f{i}", "user" if i % 2 == 0 else "assistant", "weather talk " * 20, 60) for i in range(40)]
    fact = Message("fact", "user", "My sister is called Priya.", 8)
    conv = Conversation("para", [fact] + filler, "What is my sibling's name?", "Priya", frozenset({"fact"}))
    cache = EmbeddingCache(toy_encoder)
    emb = run([lambda: RecentPlusEmbedding(cache, pair=False)], [conv], [300])[0]
    bm = run([lambda: RecentPlusBM25(pair=False)], [conv], [300])[0]
    assert emb.recalled and not bm.recalled
