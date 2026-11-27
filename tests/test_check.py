import json

import pytest

from memory_recall_bench.check import compare
from memory_recall_bench.cli import main
from memory_recall_bench.datasets import synthetic


def test_floor_pass_and_fail(capsys):
    assert main(["check", "--adapter", "recent+bm25", "--limit", "6", "--budget", "4k", "--min-recall", "0"]) == 0
    assert main(["check", "--adapter", "recent+random", "--limit", "6", "--budget", "1k", "--min-recall", "1.01"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_baseline_round_trip_and_regression(tmp_path, capsys):
    base = tmp_path / "base.json"
    assert main(["check", "--adapter", "recent+bm25", "--limit", "8", "--budget", "2k",
                 "--write-baseline", str(base)]) == 0
    saved = json.loads(base.read_text())
    assert saved["n"] == 8 and saved["budget"] == 2000
    # same retriever against its own baseline passes
    assert main(["check", "--adapter", "recent+bm25", "--limit", "8", "--budget", "2k", "--baseline", str(base)]) == 0
    # a worse retriever fails unless the allowed drop covers the gap
    if saved["recall"] > 0:
        assert main(["check", "--adapter", "fifo", "--limit", "8", "--budget", "2k", "--baseline", str(base)]) == 1
        assert "broken" in capsys.readouterr().out
        assert main(["check", "--adapter", "fifo", "--limit", "8", "--budget", "2k", "--baseline", str(base),
                     "--max-drop", "1"]) == 0


def test_compare_lists_fixed_and_broken():
    base = {"budget": 4000, "metric": "recalled", "n": 4, "recall": 0.5, "hits": ["a", "b"]}
    now = {"budget": 4000, "metric": "recalled", "n": 4, "recall": 0.5, "hits": ["b", "c"]}
    ok, lines = compare(now, base, None, 0.0)
    text = "\n".join(lines)
    assert ok and "1 fixed, 1 broken" in text and "broken: a" in text and "fixed: c" in text


def test_compare_rejects_mismatched_baseline():
    base = {"budget": 2000, "metric": "recalled", "n": 1, "recall": 1.0, "hits": ["a"]}
    with pytest.raises(ValueError):
        compare({"budget": 4000, "metric": "recalled", "n": 1, "recall": 1.0, "hits": ["a"]}, base, None, 0)


def test_external_rankings_and_export(tmp_path):
    convs = synthetic(n=5)
    out = tmp_path / "convs.jsonl"
    assert main(["export", "--limit", "5", "--out", str(out)]) == 0
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["id"] for r in rows] == [c.id for c in convs]
    # a perfect system: evidence first
    rankings = {c.id: sorted((m.id for m in c.messages), key=lambda i: i not in c.evidence_ids) for c in convs}
    rp = tmp_path / "rankings.json"
    rp.write_text(json.dumps(rankings))
    assert main(["check", "--rankings", str(rp), "--limit", "5", "--budget", "2k", "--min-recall", "1"]) == 0


def test_writes_github_step_summary(tmp_path, monkeypatch):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    main(["check", "--adapter", "recent+bm25", "--limit", "3", "--budget", "2k"])
    assert "Retrieval check" in summary.read_text()


def test_check_accepts_recent_share(tmp_path, capsys):
    from memory_recall_bench.datasets import synthetic
    convs = synthetic(n=10)
    r = {c.id: [m.id for m in c.messages if m.id in c.evidence_ids] for c in convs}
    (tmp_path / "r.json").write_text(json.dumps(r))
    assert main(["check", "--rankings", str(tmp_path / "r.json"), "--limit", "10", "--budget", "1k",
                 "--recent-share", "0", "--min-recall", "1.0"]) == 0


@pytest.mark.parametrize("share", ["1.5", "nan", "x"])
def test_check_rejects_invalid_recent_share(share, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["check", "--adapter", "recent+bm25", "--limit", "2", "--recent-share", share])
    assert exc.value.code == 2
