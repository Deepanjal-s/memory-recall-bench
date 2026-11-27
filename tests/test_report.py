import json

from memory_recall_bench.cli import main
from memory_recall_bench.datasets import synthetic
from memory_recall_bench.report import LOW, NEVER, build, diagnose, to_html
from memory_recall_bench.types import Conversation, Message


def _conv():
    msgs = [Message(f"m{i}", "user", f"text {i}", 10) for i in range(6)]
    return Conversation("c", msgs, "q?", "a", frozenset({"m1", "m4"}), kind="multi-session",
                        evidence_groups=(frozenset({"m1"}), frozenset({"m4"})))


def test_diagnose_causes():
    c = _conv()
    # m1 kept, m4 ranked but cut off -> incomplete + ranked too low
    d = diagnose(c, ["m1", "m0", "m4"], kept={"m1", "m0"})
    assert d["cause"] == LOW and d["incomplete"] and d["groups"][1]["best_rank"] == 3
    # neither group in ranking -> never retrieved, not incomplete
    d = diagnose(c, ["m0", "m2"], kept={"m0"})
    assert d["cause"] == NEVER and not d["incomplete"] and d["n_missing"] == 2
    # one never retrieved, one ranked too low -> mixed
    assert diagnose(c, ["m0", "m4"], kept={"m0"})["cause"] == "mixed"


def _rankings(convs, path, evidence_last=True):
    r = {}
    for c in convs:
        ids = [m.id for m in c.messages]
        ev = [i for i in ids if i in c.evidence_ids]
        rest = [i for i in ids if i not in c.evidence_ids]
        r[c.id] = rest + ev if evidence_last else ev + rest
    path.write_text(json.dumps(r))
    return path


def test_build_counts_add_up(tmp_path):
    convs = synthetic(n=12)
    rep = build(convs, str(_rankings(convs, tmp_path / "r.json")), [1000, 4000], 1000, name="worst")
    n_miss = round((1 - rep["curve"][1000]["all"]) * rep["n"])
    assert len(rep["misses"]) == n_miss == sum(rep["causes"].values())
    assert rep["fixes"]
    best = build(convs, str(_rankings(convs, tmp_path / "b.json", evidence_last=False)), [1000, 4000], 1000)
    assert best["curve"][1000]["all"] >= rep["curve"][1000]["all"]
    page = to_html(rep)
    assert page.startswith("<!doctype html>") and "Retrieval audit: worst" in page


def test_cli_writes_report(tmp_path, capsys):
    convs = synthetic(n=6)
    out = tmp_path / "rep.html"
    assert main(["report", "--limit", "6", "--rankings", str(_rankings(convs, tmp_path / "r.json")),
                 "--focus", "2k", "--budgets", "1k,2k", "--out", str(out), "--json", str(tmp_path / "r.json.out")]) == 0
    assert out.exists() and "all-evidence recall" in capsys.readouterr().out
    assert json.loads((tmp_path / "r.json.out").read_text())["focus"] == 2000


def test_html_escapes_content(tmp_path):
    rep = {"name": "<b>x</b>", "n": 1, "budgets": [1000], "focus": 1000, "curve": {1000: {"any": 0, "all": 0}},
           "per_kind": {"k": {"n": 1, "all": 0}}, "causes": {NEVER: 1}, "incomplete": 0, "fixes": ["f"],
           "misses": [{"id": "c", "kind": "k", "question": "<script>", "answer": "a", "evidence": "e",
                       "cause": NEVER, "incomplete": False, "n_groups": 1, "n_missing": 1,
                       "groups": [{"status": NEVER, "best_rank": None}]}]}
    page = to_html(rep)
    assert "<script>" not in page and "&lt;script&gt;" in page and "<b>x</b>" not in page


def test_recent_share_zero_gives_ranking_the_whole_budget(tmp_path):
    # 10 messages of 10 tokens, evidence m1 and m3 ranked first, budget 40. With the default half-budget recent
    # window (m8, m9) only the m1 pair fits; with no window both pairs fit.
    msgs = [Message(f"m{i}", "user", f"text {i}", 10) for i in range(10)]
    conv = Conversation("c", msgs, "q?", "a", frozenset({"m1", "m3"}), kind="multi-session",
                        evidence_groups=(frozenset({"m1"}), frozenset({"m3"})))
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"c": ["m1", "m3"]}))
    half = build([conv], str(path), [40], 40)
    whole = build([conv], str(path), [40], 40, recent_share=0.0)
    assert half["curve"][40]["all"] == 0.0 and whole["curve"][40]["all"] == 1.0
    assert any("--recent-share 0" in f for f in half["fixes"])
    assert whole["recent_share"] == 0.0 and not whole["misses"]
