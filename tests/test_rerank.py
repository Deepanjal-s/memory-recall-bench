import pytest

from memory_recall_bench import RecentPlusBM25
from memory_recall_bench.rerank import Rerank
from memory_recall_bench.types import Message


def msg(mid, text, tokens=10, role="user"):
    return Message(mid, role, text, tokens)


ARCHIVE = [
    msg("a:0", "Paris Paris Paris trip notes"),
    msg("b:0", "Paris hotel booked, the flight is on Monday"),
    msg("c:0", "Paris museum list"),
    msg("d:0", "unrelated chat about tea"),
]
QUERY = "when is the Paris flight"


def ids(ms):
    return [m.id for m in ms]


def by_word(word):
    """Toy scorer: 1 if the text contains ``word``; counts its calls."""
    calls = []

    def score(query, texts):
        calls.append(list(texts))
        return [float(word in t) for t in texts]
    score.calls = calls
    return score


def test_scorer_reorders_the_top():
    base = ids(RecentPlusBM25().rank(QUERY, ARCHIVE))
    assert base[0] != "c:0"
    ranked = ids(Rerank(RecentPlusBM25(), by_word("museum")).rank(QUERY, ARCHIVE))
    assert ranked[0] == "c:0" and sorted(ranked) == sorted(base)


def test_below_top_n_keeps_first_stage_order():
    base = ids(RecentPlusBM25().rank(QUERY, ARCHIVE))
    ranked = ids(Rerank(RecentPlusBM25(), by_word("museum"), top_n=2).rank(QUERY, ARCHIVE))
    assert ranked[2:] == base[2:]
    assert sorted(ranked[:2]) == sorted(base[:2])


def test_ties_keep_first_stage_order():
    base = RecentPlusBM25().rank(QUERY, ARCHIVE)
    assert Rerank(RecentPlusBM25(), lambda q, ts: [0.0] * len(ts)).rank(QUERY, ARCHIVE) == base


def test_fuse_moves_less_than_pure_rerank():
    base = ids(RecentPlusBM25().rank(QUERY, ARCHIVE))
    target = "c:0"
    pure = ids(Rerank(RecentPlusBM25(), by_word("museum")).rank(QUERY, ARCHIVE))
    fused = ids(Rerank(RecentPlusBM25(), by_word("museum"), fuse=True).rank(QUERY, ARCHIVE))
    assert base.index(target) > 0
    assert pure.index(target) <= fused.index(target) <= base.index(target)


def test_scores_are_cached_per_query_and_text():
    s = by_word("Monday")
    r = Rerank(RecentPlusBM25(), s)
    r.rank(QUERY, ARCHIVE)
    r.rank(QUERY, ARCHIVE)
    assert len(s.calls) == 1


def test_validation_and_name():
    with pytest.raises(ValueError):
        Rerank(RecentPlusBM25(), by_word("x"), top_n=0)
    assert Rerank(RecentPlusBM25(), by_word("x")).name == "recent+bm25+rerank N=100"
    assert Rerank(RecentPlusBM25(), by_word("x"), fuse=True, top_n=50).name == "recent+bm25+rerank rrf N=50"


def test_runs_inside_the_budget_logic():
    r = Rerank(RecentPlusBM25(), by_word("Monday"), top_n=10)
    r.reset(40)  # archive share 20 tokens -> room for one pair-less message beyond the window
    for m in ARCHIVE + [msg("e:0", "latest message", role="assistant")]:
        r.add(m)
    assert "e:0" in r.context(QUERY)
