"""Wrappers around the OpenAI API for embedding and chat models."""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
from openai import OpenAI

from ..storage.vector_store import EmbeddingBackend


class OpenAIEmbedder(EmbeddingBackend):
    """Thin wrapper around OpenAI's embedding endpoint."""

    def __init__(self, *, model: str = "text-embedding-3-large") -> None:
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model_name, input=list(texts))
        vectors = [item.embedding for item in response.data]
        return np.asarray(vectors, dtype=np.float32)


class OpenAIChatModel:
    """Wrapper around OpenAI's Chat Completions API."""

    def __init__(self, *, model: str = "gpt-4o-mini") -> None:
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def generate(self, messages: Sequence[dict]) -> str:
        response = self.client.chat.completions.create(model=self.model, messages=list(messages))
        choice = response.choices[0]
        return choice.message.content or ""
