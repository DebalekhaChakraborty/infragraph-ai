"""ChromaDB persistence helpers for InfraGraph vector memory."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .embeddings import EmbeddingModel, SETUP_MESSAGE

DEFAULT_PERSIST_DIR = "./outputs/vector_memory/chroma"
DEFAULT_COLLECTION = "infragraph_memory"


def get_chroma_client(persist_dir: str = DEFAULT_PERSIST_DIR):
    try:
        import chromadb  # type: ignore
    except Exception as exc:
        raise RuntimeError(SETUP_MESSAGE) from exc
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def get_or_create_collection(name: str = DEFAULT_COLLECTION, persist_dir: str = DEFAULT_PERSIST_DIR):
    client = get_chroma_client(persist_dir)
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    cleaned: dict[str, str | int | float | bool] = {}
    for key in (
        "source_type",
        "evidence_id",
        "scenario_id",
        "diagram_id",
        "node_id",
        "edge_id",
        "incident_id",
        "rca_source",
        "scope",
        "path",
    ):
        value = metadata.get(key, "")
        if value is None:
            value = ""
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = json.dumps(value, sort_keys=True)
    return cleaned


def upsert_documents(collection, docs: list[dict[str, Any]]) -> int:
    valid_docs = [doc for doc in docs if doc.get("id") and doc.get("text")]
    if not valid_docs:
        return 0
    ids = [str(doc["id"]) for doc in valid_docs]
    texts = [str(doc["text"]) for doc in valid_docs]
    metadatas = [_clean_metadata(doc.get("metadata", {})) for doc in valid_docs]
    embeddings = EmbeddingModel().embed_texts(texts)
    collection.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    return len(valid_docs)
