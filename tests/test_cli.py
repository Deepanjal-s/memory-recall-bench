import json
from functools import partial
from hashlib import sha256

import pytest

from memory_recall_bench import FIFO, RecentPlusBM25, RecentPlusRandom, run, synthetic
from memory_recall_bench.cli import main
from memory_recall_bench.runner import to_records


def test_default_run_matches_pre_max_hits_output(tmp_path, capsys):
    output = tmp_path / "results.json"
    assert main(["--json", str(output)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    # SHA-256 of the stdout and JSON bytes from 5410857, with LF line endings.
    assert sha256(captured.out.encode()).hexdigest() == (
        "4b2b99e7b703a4394df64e407bafae3d3d209ae8f6c4e4993de717853f6e8639"
    )
    assert sha256(output.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == (
        "54c213106c406ef56d51b388ef2ae4f8c8ff5b7a92ccafd81989c0d46ec83dcf"
    )


@pytest.mark.parametrize("args", [
    ["run"], ["check", "--adapter", "recent+bm25"], ["report", "--rankings", "unused.json"],
])
@pytest.mark.parametrize("hits", ["0", "-1", "1.5", "nan", "invalid"])
def test_max_hits_rejects_invalid_values(args, hits, capsys):
    with pytest.raises(SystemExit) as exc:
        main([*args, "--max-hits", hits])
    assert exc.value.code == 2
    assert "argument --max-hits: must be a positive integer" in capsys.readouterr().err


def test_run_applies_max_hits(tmp_path, capsys):
    output = tmp_path / "results.json"
    assert main(["run", "--adapters", "recent+bm25", "--budgets", "2k", "--recent-share", "0",
                 "--max-hits", "1", "--json", str(output)]) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    expected = run([partial(RecentPlusBM25, recent_share=0, max_hits=1)], synthetic(), [2000])
    assert result["results"] == to_records(expected)
    assert result["max_hits"] == 1
    assert "max_hits=1" in capsys.readouterr().out


def test_check_and_report_apply_max_hits(tmp_path, monkeypatch, capsys):
    from memory_recall_bench.types import Conversation, Message

    messages = [Message("distractor", "user", "other", 10), Message("evidence", "user", "answer", 10)]
    conv = Conversation("c", messages, "question", "answer", {"evidence"})
    monkeypatch.setattr("memory_recall_bench.check._load", lambda args: [conv])
    monkeypatch.setattr("memory_recall_bench.report._load", lambda args: [conv])
    rankings = tmp_path / "rankings.json"
    rankings.write_text(json.dumps({"c": ["distractor", "evidence"]}), encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    raw = tmp_path / "report.json"
    html = tmp_path / "report.html"
    packing = ["--rankings", str(rankings), "--recent-share", "0", "--max-hits", "1"]

    assert main(["check", *packing, "--write-baseline", str(baseline)]) == 0
    check = json.loads(baseline.read_text(encoding="utf-8"))
    assert (check["max_hits"], check["recall"]) == (1, 0)
    assert "max_hits=1" in capsys.readouterr().out

    assert main(["report", *packing, "--out", str(html), "--json", str(raw)]) == 0
    report = json.loads(raw.read_text(encoding="utf-8"))
    assert (report["max_hits"], report["curve"]["4000"]["all"]) == (1, 0)
    assert "max_hits=1" in html.read_text(encoding="utf-8")
    assert "max_hits=1" in capsys.readouterr().out


def test_default_run_keeps_adapter_defaults(tmp_path, capsys):
    output = tmp_path / "results.json"
    assert main(["--adapters", "recent+bm25,recent+random", "--budgets", "2k",
                 "--json", str(output)]) == 0
    expected = run([RecentPlusBM25, RecentPlusRandom], synthetic(), [2000])
    assert json.loads(output.read_text(encoding="utf-8"))["results"] == to_records(expected)
    assert capsys.readouterr().err == ""


def test_recent_share_zero_uses_the_full_budget_for_retrieval(tmp_path, capsys):
    output = tmp_path / "results.json"
    args = ["--adapters", "recent+bm25,recent+random,flush-10pct", "--budgets", "1k",
            "--json", str(output)]
    assert main(args) == 0
    default = json.loads(output.read_text(encoding="utf-8"))
    assert main([*args, "--recent-share", "0"]) == 0
    whole = json.loads(output.read_text(encoding="utf-8"))
    default_recall = {r["adapter"]: r["recall"] for r in default["summary"]}
    whole_recall = {r["adapter"]: r["recall"] for r in whole["summary"]}
    assert whole_recall["recent+bm25"] >= default_recall["recent+bm25"]
    convs = synthetic()
    for adapter in (RecentPlusBM25, RecentPlusRandom):
        expected = run([partial(adapter, recent_share=0)], convs, [1000])
        assert [r for r in whole["results"] if r["adapter"] == adapter.name] == to_records(expected)
    assert [r for r in whole["results"] if r["adapter"] == "flush-10pct"] == [
        r for r in default["results"] if r["adapter"] == "flush-10pct"
    ]
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("share", ["-0.1", "1.1", "nan", "inf", "invalid"])
def test_recent_share_rejects_invalid_values(share, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--recent-share", share])
    assert exc.value.code == 2
    assert "argument --recent-share: must be a number between 0 and 1 inclusive" in capsys.readouterr().err


def test_recent_share_prints_note_for_fifo(tmp_path, capsys):
    output = tmp_path / "results.json"
    assert main(["--adapters", "fifo", "--limit", "2", "--budgets", "1k",
                 "--recent-share", "0.5", "--json", str(output)]) == 0
    expected = run([FIFO], synthetic(n=2), [1000])
    assert json.loads(output.read_text(encoding="utf-8"))["results"] == to_records(expected)
    note, = capsys.readouterr().err.splitlines()
    assert note.startswith("Note: --recent-share has no effect")
