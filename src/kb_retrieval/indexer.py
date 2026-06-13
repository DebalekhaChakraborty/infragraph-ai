"""
KB index builder — loads KB docs, chunks them, and stores embeddings in ChromaDB.

build_kb_index : main entry point
"""
from __future__ import annotations

import json
from pathlib import Path

from .chunker import chunk_document
from .loader import load_kb_documents
from .schema import DEFAULT_COLLECTION, DEFAULT_EMBED_MODEL, DEFAULT_INDEX_DIR, DEFAULT_KB_ROOT


def build_kb_index(
    *,
    kb_root: Path | str | None = None,
    index_dir: Path | str | None = None,
    collection_name: str = DEFAULT_COLLECTION,
    reset: bool = False,
) -> dict:
    """
    Build (or rebuild) the ChromaDB vector index for the KB documents.

    Parameters
    ----------
    kb_root         : path to the KB root directory (default: DEFAULT_KB_ROOT)
    index_dir       : path for the persistent ChromaDB store (default: DEFAULT_INDEX_DIR)
    collection_name : ChromaDB collection name
    reset           : if True, delete the collection before rebuilding

    Returns
    -------
    dict with keys: documents_loaded, chunks_indexed, collection_name, index_dir
    """
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "chromadb is required for KB indexing.\n"
            "Install it with:  pip install chromadb>=0.5.0"
        ) from exc

    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for KB indexing.\n"
            "Install it with:  pip install sentence-transformers>=3.0.0"
        ) from exc

    _kb_root  = Path(kb_root  or DEFAULT_KB_ROOT)
    _idx_dir  = Path(index_dir or DEFAULT_INDEX_DIR)
    _idx_dir.mkdir(parents=True, exist_ok=True)

    embed_model = DEFAULT_EMBED_MODEL
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=embed_model)

    client = chromadb.PersistentClient(path=str(_idx_dir))

    if reset:
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
    )

    documents = load_kb_documents(_kb_root)
    if not documents:
        return {
            "documents_loaded": 0,
            "chunks_indexed":   0,
            "collection_name":  collection_name,
            "index_dir":        str(_idx_dir),
            "embed_model":      embed_model,
        }

    all_chunks: list[dict] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))

    if not all_chunks:
        return {
            "documents_loaded": len(documents),
            "chunks_indexed":   0,
            "collection_name":  collection_name,
            "index_dir":        str(_idx_dir),
            "embed_model":      embed_model,
        }

    # Upsert in batches of 100
    batch_size = 100
    for start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[start : start + batch_size]
        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[_serialize_metadata(c["metadata"]) for c in batch],
        )

    return {
        "documents_loaded": len(documents),
        "chunks_indexed":   len(all_chunks),
        "collection_name":  collection_name,
        "index_dir":        str(_idx_dir),
        "embed_model":      embed_model,
    }


def _serialize_metadata(meta: dict) -> dict:
    """ChromaDB metadata values must be str, int, float, or bool — no lists."""
    out: dict = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, list):
            out[k] = json.dumps(v)
        else:
            out[k] = str(v)
    return out
