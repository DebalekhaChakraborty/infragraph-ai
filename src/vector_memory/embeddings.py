"""Embedding backend for InfraGraph vector memory."""
from __future__ import annotations

import os

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SETUP_MESSAGE = "Vector memory is not installed. Run pip install chromadb sentence-transformers."


class EmbeddingModel:
    """Thin wrapper around sentence-transformers embeddings."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.environ.get("INFRAGRAPH_EMBEDDING_MODEL", DEFAULT_MODEL)
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:
            raise RuntimeError(SETUP_MESSAGE) from exc
        self.model = SentenceTransformer(self.model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, vec)) for vec in vectors]

    def embed_query(self, query: str) -> list[float]:
        vectors = self.embed_texts([query])
        return vectors[0] if vectors else []
