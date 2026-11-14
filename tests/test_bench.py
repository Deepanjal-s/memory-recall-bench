import json

import pytest

from memory_recall_bench import (
    FIFO, BudgetExceeded, FlushToArchive, MemoryAdapter, RecentPlusBM25, RecentPlusRandom,
    load_longmemeval, run, summarize, synthetic,
)
from memory_recall_bench.cli import main

CONVS = synthetic(n=24, turns=150, seed=1)
BUDGETS = [500, 2000, 6000]


def test_adapters_satisfy_the_protocol():
    for a in [FIFO(), FlushToArchive(), RecentPlusBM25(), RecentPlusRandom()]:
        assert isinstance(a, MemoryAdapter)


def test_run_accepts_classes_factories_and_instances():
    res = run([FIFO, lambda: FlushToArchive(flush_fraction=0.1), RecentPlusBM25()], CONVS[:3], [1000])
    assert {r.adapter for r in res} == {"fifo", "flush-10pct", "recent+bm25"}


def test_fifo_matches_its_closed_form():
    # FIFO keeps the evidence iff the evidence message and everything after it fit.
    for r in run([FIFO], CONVS, BUDGETS):
        conv = next(c for c in CONVS if c.id == r.conversation)
        idx = next(i for i, m in enumerate(conv.messages) if m.id in conv.evidence_ids)
        assert r.recalled == (sum(m.tokens for m in conv.messages[idx:]) <= r.budget)


def test_fifo_returns_exactly_the_longest_suffix_that_fits():
    # Stronger than the recall check: pins the exact context, so an off-by-some
    # budget comparison is caught even when it doesn't change recall.
    for conv in CONVS:
        for budget in [137, 500, 2000, 6000]:
            fifo = FIFO()
            fifo.reset(budget)
            for m in conv.messages:
                fifo.add(m)
            expected, used = [], 0
            for m in reversed(conv.messages):
                if used + m.tokens > budget:
                    break
                expected.append(m.id)
                used += m.tokens
            assert list(fifo.context("")) == expected


@pytest.mark.parametrize("adapter", [FIFO, FlushToArchive, RecentPlusBM25, RecentPlusRandom])
def test_reference_adapters_respect_the_budget(adapter):
    for r in run([adapter], CONVS, BUDGETS):
        assert r.context_tokens <= r.budget


def test_budget_violations_are_caught():
    class Cheater(FIFO):
        name = "cheater"

        def context(self, query):
            return [m.id for m in self.messages]

    with pytest.raises(BudgetExceeded):
        run([Cheater], CONVS[:1], [500])
    assert run([Cheater], CONVS[:1], [500], strict=False)[0].recalled


def test_unknown_ids_are_rejected():
    class Liar(FIFO):
        name = "liar"

        def context(self, query):
            return ["not-a-real-id"]

    with pytest.raises(ValueError, match="unknown message ids"):
        run([Liar], CONVS[:1], [500])


def test_retrieval_beats_its_random_control_on_planted_facts():
    rows = {(r["adapter"], r["budget"]): r["recall"]
            for r in summarize(run([RecentPlusBM25, RecentPlusRandom, FIFO], CONVS, [2000]))}
    assert rows[("recent+bm25", 2000)] >= 0.9
    assert rows[("recent+bm25", 2000)] > rows[("recent+random", 2000)] + 0.3


def test_default_flush_size_underfills_small_budgets():
    # Regression for the llama_index #24117 behaviour: a 3000-token flush with a ~2.5k
    # history limit empties the history; a budget-scaled flush keeps it mostly full.
    def mean_fill(adapter):
        adapter.reset(3644)
        for m in CONVS[0].messages:
            adapter.add(m)
        trace = adapter.fill_trace[20:]
        return sum(trace) / len(trace) / adapter.limit

    default, scaled = mean_fill(FlushToArchive()), mean_fill(FlushToArchive(flush_fraction=0.1))
    assert default < 0.65 < 0.8 < scaled


def test_random_control_is_deterministic():
    a = run([RecentPlusRandom], CONVS[:5], [2000])
    b = run([RecentPlusRandom], CONVS[:5], [2000])
    assert [r.recalled for r in a] == [r.recalled for r in b]


def test_summarize_reports_bounded_intervals():
    for row in summarize(run([FIFO, RecentPlusBM25], CONVS, BUDGETS)):
        assert 0 <= row["ci_low"] <= row["recall"] <= row["ci_high"] <= 1
        assert row["n"] == len(CONVS)


def test_longmemeval_loader(tmp_path):
    sample = [{
        "question_id": "q1", "question_type": "single-session-user", "question": "What degree did I get?",
        "answer": "Business Administration", "haystack_session_ids": ["s1", "s2"],
        "haystack_sessions": [
            [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
            [{"role": "user", "content": "I graduated in Business Administration", "has_answer": True}],
        ],
    }, {
        "question_id": "q2", "question_type": "abstention", "question": "?", "answer": "n/a",
        "haystack_session_ids": ["s9"], "haystack_sessions": [[{"role": "user", "content": "x"}]],
    }]
    path = tmp_path / "lme.json"
    path.write_text(json.dumps(sample), encoding="utf-8")
    convs = load_longmemeval(path)
    assert len(convs) == 1  # the question with no marked evidence is skipped
    assert convs[0].evidence_ids == {"s2:0"}
    assert convs[0].evidence_age_tokens() == 0


def test_cli_runs_on_synthetic_data(capsys, tmp_path):
    out = tmp_path / "r.json"
    assert main(["--limit", "6", "--budgets", "1k,4k", "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "recent+bm25" in text and "4k" in text
    assert json.loads(out.read_text())["summary"]


def test_cli_rejects_unknown_adapters():
    with pytest.raises(SystemExit):
        main(["--adapters", "fifo,nope"])


def test_bm25_pairing_brings_the_reply_partner_and_respects_budget():
    conv = CONVS[0]
    paired, single = RecentPlusBM25(pair=True), RecentPlusBM25(pair=False)
    for a in (paired, single):
        a.reset(1500)
        for m in conv.messages:
            a.add(m)
    tokens = {m.id: m.tokens for m in conv.messages}
    ids_p, ids_s = list(paired.context(conv.question)), list(single.context(conv.question))
    assert sum(tokens[i] for i in ids_p) <= 1500 and sum(tokens[i] for i in ids_s) <= 1500
    assert len(set(ids_p)) == len(ids_p)  # no duplicates
    ev = next(iter(conv.evidence_ids))
    assert ev in ids_p
    idx = [m.id for m in conv.messages].index(ev)
    partner = conv.messages[idx + 1 if conv.messages[idx].role == "user" else idx - 1].id
    assert partner in ids_p


def test_all_evidence_requires_every_group():
    from memory_recall_bench import Conversation, Message
    msgs = [Message(f"m{i}", "user" if i % 2 == 0 else "assistant", f"text {i}", 10) for i in range(10)]
    conv = Conversation("multi", msgs, "q?", "a", frozenset({"m0", "m8"}), kind="multi-session",
                        evidence_groups=(frozenset({"m0"}), frozenset({"m8"})))
    r = run([FIFO], [conv], [30])[0]  # keeps m7..m9: has m8, not m0
    assert r.recalled and not r.recalled_all
    rows = summarize([r], by_kind=True, metric="recalled_all")
    assert rows[0]["kind"] == "multi-session" and rows[0]["recall"] == 0


def test_single_group_default_makes_both_metrics_equal():
    for r in run([FIFO, RecentPlusBM25], CONVS, [2000]):
        assert r.recalled == r.recalled_all


def test_cli_by_kind_and_all_evidence(capsys):
    assert main(["--limit", "4", "--budgets", "2k", "--by-kind", "--all-evidence"]) == 0
    out = capsys.readouterr().out
    assert "All-evidence recall" in out and "[synthetic]" in out


def test_external_rankings_round_trip(tmp_path):
    from memory_recall_bench.external import RankedFromFile, export_conversations
    convs = CONVS[:5]
    export_conversations(convs, tmp_path / "c.jsonl")
    lines = [json.loads(line) for line in (tmp_path / "c.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [line["id"] for line in lines] == [c.id for c in convs]
    assert sum(len(s["messages"]) for s in lines[0]["sessions"]) == len(convs[0].messages)
    # A "perfect" external system: ranks the evidence first. It must recall everything.
    rankings = {c.id: sorted((m.id for m in c.messages), key=lambda i: i not in c.evidence_ids) for c in convs}
    (tmp_path / "r.json").write_text(json.dumps(rankings), encoding="utf-8")
    res = run([lambda: RankedFromFile(tmp_path / "r.json", convs, name="perfect")], convs, [600])
    assert all(r.recalled for r in res) and all(r.context_tokens <= 600 for r in res)
    # Missing conversations are an error, not a silent zero.
    (tmp_path / "bad.json").write_text("{}", encoding="utf-8")
    with pytest.raises(KeyError):
        RankedFromFile(tmp_path / "bad.json", convs)
