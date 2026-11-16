"""The mem0 adapter on the synthetic set: it must find planted facts that a recent window misses.

Skipped unless the extra is installed: pip install -e ".[mem0]" and
python -m spacy download en_core_web_sm (downloads bge-small on first run).
"""
import pytest

pytest.importorskip("mem0")
pytest.importorskip("fastembed")

from memory_recall_bench import FIFO, run, summarize, synthetic  # noqa: E402
from memory_recall_bench.integrations.mem0 import Mem0Retriever  # noqa: E402


@pytest.fixture(scope="module")
def results():
    convs = synthetic(n=4, turns=60, seed=3)
    return run([FIFO(), Mem0Retriever()], convs, [300])


def test_finds_planted_facts_that_fifo_drops(results):
    recall = {s["adapter"]: s["recall"] for s in summarize(results)}
    assert recall["fifo"] < 1.0
    assert recall["mem0"] == 1.0


def test_context_stays_within_budget(results):  # run(strict=True) also rejects duplicates and unknown ids
    for r in results:
        if r.adapter == "mem0":
            assert r.context_tokens <= 300
