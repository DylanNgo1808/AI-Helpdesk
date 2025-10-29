"""A lightweight vector store backed by NumPy."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from ..models import Document, DocumentChunk, SearchResult
from ..utils.chunking import chunk_text, enumerate_chunks


class EmbeddingBackend:
    """Protocol for embedding providers."""

    model_name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class VectorStore:
    """Persist embeddings and associated document chunks on disk."""

    def __init__(self, storage_dir: str | Path = "data/store") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._vectors_path = self.storage_dir / "vectors.npy"
        self._meta_path = self.storage_dir / "metadata.json"
        self._chunks: List[DocumentChunk] = []
        self._embeddings: np.ndarray | None = None
        self._embedding_model: str | None = None
        self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self._meta_path.exists() and self._vectors_path.exists():
            with self._meta_path.open("r", encoding="utf-8") as fh:
                meta = json.load(fh)
            self._chunks = [DocumentChunk(**item) for item in meta.get("chunks", [])]
            self._embedding_model = meta.get("embedding_model")
            self._embeddings = np.load(self._vectors_path)

    def _save(self) -> None:
        payload = {
            "chunks": [asdict(chunk) for chunk in self._chunks],
            "embedding_model": self._embedding_model,
        }
        with self._meta_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        if self._embeddings is not None:
            np.save(self._vectors_path, self._embeddings)

    # ------------------------------------------------------------------
    def add_documents(
        self,
        documents: Iterable[Document],
        *,
        embedder: EmbeddingBackend,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> None:
        """Chunk documents, embed them and update the on-disk store."""

        new_chunks: List[DocumentChunk] = []
        chunk_texts: List[str] = []

        for document in documents:
            chunks = chunk_text(document.content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunk_ids = enumerate_chunks(chunks, prefix=document.id)
            for chunk_id, chunk_content in zip(chunk_ids, chunks):
                new_chunks.append(
                    DocumentChunk(
                        id=chunk_id,
                        document_id=document.id,
                        source=document.source,
                        content=chunk_content,
                        metadata=document.metadata,
                    )
                )
                chunk_texts.append(chunk_content)

        if not new_chunks:
            return

        embeddings = embedder.embed(chunk_texts)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")

        if self._embeddings is None:
            self._embeddings = embeddings
            self._chunks = new_chunks
        else:
            if embeddings.shape[1] != self._embeddings.shape[1]:
                raise ValueError("Embedding dimension mismatch")
            self._embeddings = np.vstack([self._embeddings, embeddings])
            self._chunks.extend(new_chunks)

        self._embedding_model = embedder.model_name
        self._save()

    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        embedder: EmbeddingBackend,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Return the ``top_k`` most relevant chunks for ``query``."""

        if not self._chunks or self._embeddings is None:
            return []

        query_vec = embedder.embed([query])
        query_vec = np.asarray(query_vec, dtype=np.float32)
        if query_vec.ndim != 2 or query_vec.shape[0] != 1:
            raise ValueError("Query embedding must be a 2D array with a single row")
        query_vec = query_vec[0]

        doc_vectors = self._embeddings
        # Cosine similarity
        doc_norms = np.linalg.norm(doc_vectors, axis=1) + 1e-10
        query_norm = np.linalg.norm(query_vec) + 1e-10
        similarities = (doc_vectors @ query_vec) / (doc_norms * query_norm)

        top_indices = similarities.argsort()[::-1][:top_k]
        results = [
            SearchResult(chunk=self._chunks[idx], score=float(similarities[idx]))
            for idx in top_indices
        ]
        return results

    # ------------------------------------------------------------------
    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def embedding_dimension(self) -> int | None:
        if self._embeddings is None:
            return None
        return int(self._embeddings.shape[1])
