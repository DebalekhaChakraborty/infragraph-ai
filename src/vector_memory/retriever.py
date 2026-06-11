"""Semantic retrieval from InfraGraph vector memory."""
from __future__ import annotations

from typing import Any

from .chroma_store import DEFAULT_COLLECTION, DEFAULT_PERSIST_DIR, get_or_create_collection
from .embeddings import EmbeddingModel


def retrieve_graph_memory(
    query: str,
    k: int = 8,
    filters: dict | None = None,
    *,
    collection_name: str = DEFAULT_COLLECTION,
    persist_dir: str = DEFAULT_PERSIST_DIR,
) -> list[dict[str, Any]]:
    if not query.strip():
        return []
    collection = get_or_create_collection(name=collection_name, persist_dir=persist_dir)
    embedding = EmbeddingModel().embed_query(query)
    kwargs: dict[str, Any] = {
        "query_embeddings": [embedding],
        "n_results": max(1, int(k)),
        "include": ["documents", "metadatas", "distances"],
    }
    if filters:
        kwargs["where"] = filters
    result = collection.query(**kwargs)
    docs = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    rows: list[dict[str, Any]] = []
    for text, metadata, distance in zip(docs, metadatas, distances):
        score = 1.0 - float(distance) if distance is not None else None
        rows.append({"text": text, "metadata": metadata or {}, "distance": distance, "score": score})
    return rows
