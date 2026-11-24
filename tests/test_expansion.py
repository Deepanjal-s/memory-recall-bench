import json

import pytest

from memory_recall_bench import RecentPlusBM25
from memory_recall_bench.expansion import QueryExpansion, load_expansions
from memory_recall_bench.types import Message


def msg(mid, text, tokens=10, role="user"):
    return Message(mid, role, text, tokens)


ARCHIVE = [
    msg("a:0", "my flight to Paris leaves on Monday"),
    msg("b:0", "the Paris hotel is near the river"),
    msg("c:0", "I bought hiking boots for the Alps trip"),
    msg("d:0", "unrelated chat about tea"),
]
QUERY = "which came first, Paris"
EXP = {QUERY: ["Alps hiking boots", "Paris flight"]}


def ids(ms):
    return [m.id for m in ms]


def test_without_an_expansion_the_order_is_the_base_order():
    base = RecentPlusBM25()
    assert QueryExpansion(base, {}).rank(QUERY, ARCHIVE) == base.rank(QUERY, ARCHIVE)


def test_sub_queries_add_messages_the_question_misses():
    assert "c:0" not in ids(RecentPlusBM25().rank(QUERY, ARCHIVE))
    assert "c:0" in ids(QueryExpansion(RecentPlusBM25(), EXP, protect=0).rank(QUERY, ARCHIVE))


def test_protected_prefix_keeps_the_questions_own_top_hits_first():
    base = RecentPlusBM25()
    qe = QueryExpansion(base, EXP, protect=0.5)
    qe.reset(80)  # archive share 40 tokens, protected 20 tokens = two 10-token messages
    head = ids(base.rank(QUERY, ARCHIVE))[:2]
    ranked = ids(qe.rank(QUERY, ARCHIVE))
    assert ranked[:2] == head
    assert "c:0" in ranked and len(ranked) == len(set(ranked))


def test_protect_one_is_the_base_order_up_to_the_budget():
    base = RecentPlusBM25()
    qe = QueryExpansion(base, EXP, protect=1.0)
    qe.reset(1000)
    assert ids(qe.rank(QUERY, ARCHIVE))[:len(base.rank(QUERY, ARCHIVE))] == ids(base.rank(QUERY, ARCHIVE))


def test_protect_is_validated_and_named():
    with pytest.raises(ValueError):
        QueryExpansion(RecentPlusBM25(), {}, protect=1.5)
    assert QueryExpansion(RecentPlusBM25(), {}).name == "recent+bm25+expansion protect=0.5"
    assert QueryExpansion(RecentPlusBM25(), {}, protect=0).name == "recent+bm25+expansion"


def test_load_expansions_skips_empty_and_torn_lines(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text(json.dumps({"question": "q1", "queries": ["a", " "]}) + "\n"
                 + json.dumps({"question": "q2", "queries": []}) + "\n" + '{"question": "q3", "que')
    assert load_expansions(p) == {"q1": ["a"]}
