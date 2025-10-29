"""Core domain models for the AI Helpdesk project."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Document:
    """A raw document ingested from one of the knowledge sources."""

    id: str
    source: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    """A semantically meaningful chunk of a document."""

    id: str
    document_id: str
    source: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """A retrieval result from the vector store."""

    chunk: DocumentChunk
    score: float

    @property
    def citation(self) -> Optional[str]:
        """Return a human readable citation string when available."""

        title = self.chunk.metadata.get("title")
        if title:
            return f"{title}"
        path = self.chunk.metadata.get("path")
        if path:
            return path
        return None
