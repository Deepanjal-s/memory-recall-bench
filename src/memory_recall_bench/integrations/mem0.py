"""Adapter for mem0's open-source ``Memory`` (``pip install memory-recall-bench[mem0]``).

What is measured: mem0's *retrieval*, not its fact extraction. Messages are stored with
``add(..., infer=False)``, which keeps each message verbatim (one memory per message) and
needs no LLM. Ranking is mem0's own ``search()``: semantic candidates from the vector store,
boosted by BM25 and entity matches (mem0 2.x). A message that matches only by keyword and is
not among the semantic candidates is not returned; that is mem0's behaviour, kept as is.

Default stack, all on-device: fastembed ``BAAI/bge-small-en-v1.5`` embeddings, local Qdrant
(which supports mem0's BM25 step), spaCy for lemmas and entities. Each conversation gets a
fresh store. The LLM config is a placeholder that ``infer=False`` never calls. Telemetry is
switched off (``MEM0_TELEMETRY=False``) before mem0 is imported.

Indexing is the slow part (~0.2 s per message on a laptop CPU), so a conversation is indexed
once and reused for every budget; ``rank`` filters mem0's results to the archived messages.
"""
from __future__ import annotations

import gc
import os
import shutil
import tempfile
from typing import Any

from ..adapters import RankedRetriever
from ..types import Message


def _memory_class():
    os.environ["MEM0_TELEMETRY"] = "False"
    try:
        from mem0 import Memory
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError("Install the extra: pip install 'memory-recall-bench[mem0]' "
                          "and python -m spacy download en_core_web_sm") from e
    return Memory


class Mem0Retriever(RankedRetriever):
    """Recent window + archive ranked by mem0 ``Memory.search()`` over verbatim messages."""

    def __init__(self, recent_share: float = 0.5, pair: bool = True, name: str = "mem0",
                 embed_model: str = "BAAI/bge-small-en-v1.5", embed_dims: int = 384,
                 top_k: int = 200, threshold: float = 0.1, config: dict[str, Any] | None = None):
        super().__init__(recent_share, pair)
        self._Memory = _memory_class()
        self.name = name
        self.top_k, self.threshold = top_k, threshold
        self._config = config
        self._embed = (embed_model, embed_dims)
        self._indexed: tuple[str, ...] | None = None
        self._ranking: dict[str, list[str]] = {}

    def _new_store(self, workdir: str):
        if self._config is not None:
            return self._Memory.from_config(self._config)
        model, dims = self._embed
        return self._Memory.from_config({
            "embedder": {"provider": "fastembed", "config": {"model": model, "embedding_dims": dims}},
            "vector_store": {"provider": "qdrant", "config": {
                "collection_name": "mrb", "embedding_model_dims": dims, "path": os.path.join(workdir, "qdrant")}},
            "llm": {"provider": "openai", "config": {"api_key": "unused: add(infer=False) never calls the LLM"}},
            "history_db_path": os.path.join(workdir, "history.db"),
        })

    def rank_all(self, messages: list[Message], queries: list[str]) -> dict[str, list[str]]:
        """Indexes ``messages`` in a fresh mem0 store; returns each query's ranked message ids."""
        workdir = tempfile.mkdtemp(prefix="mrb-mem0-")
        memory = self._new_store(workdir)
        try:
            for m in messages:
                role = "user" if m.role == "user" else "assistant"
                memory.add([{"role": role, "content": m.content}], user_id="mrb",
                           metadata={"mrb_id": m.id}, infer=False)
            out = {}
            for q in queries:
                hits = memory.search(q, filters={"user_id": "mrb"}, top_k=self.top_k, threshold=self.threshold)
                out[q] = [h["metadata"]["mrb_id"] for h in hits["results"] if h.get("metadata", {}).get("mrb_id")]
            return out
        finally:
            client = getattr(memory.vector_store, "client", None)
            if client is not None and hasattr(client, "close"):
                client.close()
            del memory
            gc.collect()
            shutil.rmtree(workdir, ignore_errors=True)

    def rank(self, query: str, archive: list[Message]) -> list[Message]:
        key = tuple(m.id for m in self.messages)
        if key != self._indexed or query not in self._ranking:
            self._ranking = self.rank_all(self.messages, [query])
            self._indexed = key
        by_id = {m.id: m for m in archive}
        return [by_id[i] for i in self._ranking[query] if i in by_id]
