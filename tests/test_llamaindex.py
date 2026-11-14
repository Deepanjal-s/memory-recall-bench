"""The built-in FlushToArchive must behave exactly like llama_index's real Memory.

Skipped unless the extra is installed: pip install -e ".[llamaindex]"
"""
import pytest

pytest.importorskip("llama_index.core")

from memory_recall_bench import FlushToArchive, run, synthetic  # noqa: E402
from memory_recall_bench.integrations.llamaindex import LlamaIndexMemory  # noqa: E402


def _contexts(adapter, conv, budget):
    adapter.reset(budget)
    for m in conv.messages:
        adapter.add(m)
    return list(adapter.context(conv.question))


@pytest.mark.parametrize("budget", [300, 800, 2000, 3644, 6000])
def test_port_matches_real_memory_with_default_flush(budget):
    for conv in synthetic(n=4, turns=120, seed=7):
        assert _contexts(FlushToArchive(), conv, budget) == _contexts(LlamaIndexMemory(), conv, budget)


def test_port_matches_real_memory_with_custom_flush_size():
    for conv in synthetic(n=3, turns=120, seed=8):
        for budget in [800, 3644]:
            ours = _contexts(FlushToArchive(flush_size=200), conv, budget)
            real = _contexts(LlamaIndexMemory(token_flush_size=200), conv, budget)
            assert ours == real


def test_adapter_respects_budget_and_runs_in_the_runner():
    res = run([LlamaIndexMemory], synthetic(n=3, turns=80, seed=9), [1000, 4000])
    assert all(r.context_tokens <= r.budget for r in res)


def test_memory_blocks_are_rejected():
    with pytest.raises(ValueError, match="memory_blocks"):
        LlamaIndexMemory(memory_blocks=[object()])
