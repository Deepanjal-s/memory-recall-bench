import json
from functools import partial

import pytest

from memory_recall_bench import FIFO, RecentPlusBM25, RecentPlusRandom, run, synthetic
from memory_recall_bench.cli import main
from memory_recall_bench.runner import to_records


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
