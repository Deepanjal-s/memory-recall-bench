import json

from memory_recall_bench import RecentPlusBM25
from memory_recall_bench.summaries import SessionBoost, load_summaries, session_of
from memory_recall_bench.types import Message


def msg(mid, text, role="user"):
    return Message(mid, role, text, max(1, len(text) // 4))


ARCHIVE = [
    msg("a:0", "we talked about pasta recipes and tomato sauce"),
    msg("a:1", "sure, here is a sauce", "assistant"),
    msg("b:0", "I adopted a puppy last week"),          # the evidence: no word overlaps the query
    msg("b:1", "congrats on the new family member", "assistant"),
    msg("c:0", "what is a good name for a cat"),
]
QUERY = "what is my dog called"


def test_session_of_splits_on_last_colon():
    assert session_of("sharegpt_x:1_0:12") == "sharegpt_x:1_0"


def test_without_summaries_the_order_is_the_base_order():
    base = RecentPlusBM25()
    assert SessionBoost(base, {}).rank(QUERY, ARCHIVE) == base.rank(QUERY, ARCHIVE)


def test_a_matching_summary_lifts_its_session_above_lexical_noise():
    summaries = {"a": "cooking pasta", "b": "user got a dog named Bruno", "c": "cat names"}
    ranked = SessionBoost(RecentPlusBM25(), summaries).rank(QUERY, ARCHIVE)
    assert ranked[0].id.startswith("b:")
    assert {m.id for m in ranked} >= {"b:0", "b:1"}


def test_weight_zero_disables_the_session_term():
    summaries = {"b": "user got a dog named Bruno"}
    base = RecentPlusBM25()
    boosted = SessionBoost(base, summaries, weight=0).rank(QUERY, ARCHIVE)
    assert [m.id for m in boosted][: len(base.rank(QUERY, ARCHIVE))] == [m.id for m in base.rank(QUERY, ARCHIVE)]


def test_load_summaries_skips_empty_and_torn_lines(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text(json.dumps({"session_id": "a", "summary": "x"}) + "\n"
                 + json.dumps({"session_id": "b", "summary": "  "}) + "\n" + '{"session_id": "c", "su')
    assert load_summaries(p) == {"a": "x"}


def test_name_marks_the_wrapper():
    assert SessionBoost(RecentPlusBM25(), {}).name == "recent+bm25+sessions"
