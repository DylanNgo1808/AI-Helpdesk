"""Utilities for splitting documents into manageable chunks."""

from __future__ import annotations

import math
from typing import Iterable, List

try:  # pragma: no cover - optional dependency
    import tiktoken
except Exception:  # pragma: no cover - fallback when tiktoken is unavailable
    tiktoken = None  # type: ignore


DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100


def _split_by_tokens(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text by approximate token counts using tiktoken when available."""

    assert tiktoken is not None
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + chunk_size)
        chunk_tokens = tokens[start:end]
        chunks.append(encoding.decode(chunk_tokens))
        start += max(1, chunk_size - chunk_overlap)
    return chunks


def _split_by_chars(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Fallback chunking strategy that works without tokenizers."""

    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += step
    return chunks


def chunk_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """Split ``text`` into a sequence of overlapping chunks."""

    if not text:
        return []

    if tiktoken is not None:
        try:
            return _split_by_tokens(text, chunk_size, chunk_overlap)
        except Exception:
            pass
    return _split_by_chars(text, chunk_size, chunk_overlap)


def enumerate_chunks(chunks: Iterable[str], prefix: str) -> List[str]:
    """Attach monotonically increasing suffixes to chunk identifiers."""

    chunk_list = list(chunks)
    total_digits = max(4, math.ceil(math.log10(len(chunk_list) + 1))) if chunk_list else 4
    return [f"{prefix}-{str(idx).zfill(total_digits)}" for idx, _ in enumerate(chunk_list, start=1)]
