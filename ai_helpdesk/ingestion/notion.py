"""Utilities for ingesting documents exported from Notion."""

from __future__ import annotations

from pathlib import Path
from typing import List

from ..models import Document


def load_notion_export(path: str | Path, *, source_id: str = "notion") -> List[Document]:
    """Load a text or Markdown export from Notion.

    This helper treats the file as a single document. When exporting a Notion
    workspace as Markdown the resulting files can be concatenated prior to
    ingestion. The ``metadata`` attribute keeps the original filename to
    simplify debugging.
    """

    file_path = Path(path)
    if not file_path.exists():  # pragma: no cover - file system guard
        raise FileNotFoundError(f"Notion export not found: {file_path}")

    text = file_path.read_text(encoding="utf-8")

    return [
        Document(
            id=file_path.stem,
            source=source_id,
            content=text,
            metadata={"path": str(file_path)},
        )
    ]
