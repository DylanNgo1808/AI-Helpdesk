"""Chat orchestration for the AI Helpdesk."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from ..models import SearchResult
from ..storage.vector_store import VectorStore
from .llm import OpenAIChatModel, OpenAIEmbedder


DEFAULT_SYSTEM_PROMPT = (
    "You are an AI helpdesk assistant. Answer questions using only the provided "
    "context. Cite the titles or paths of the relevant documents in parentheses. "
    "If the answer is not present in the context, say you do not know."
)


@dataclass
class ChatResponse:
    answer: str
    references: List[SearchResult]


class ChatEngine:
    """Glue together retrieval and generation steps."""

    def __init__(
        self,
        store: VectorStore,
        *,
        embedder: OpenAIEmbedder | None = None,
        chat_model: OpenAIChatModel | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.store = store
        self.embedder = embedder or OpenAIEmbedder()
        self.chat_model = chat_model or OpenAIChatModel()
        self.system_prompt = system_prompt

    def _build_context(self, results: Iterable[SearchResult]) -> str:
        segments = []
        for idx, result in enumerate(results, start=1):
            citation = result.citation or result.chunk.id
            segments.append(
                f"[Source {idx}: {citation}]\n{result.chunk.content.strip()}"
            )
        return "\n\n".join(segments)

    def ask(self, question: str, *, top_k: int = 5) -> ChatResponse:
        references = self.store.search(question, embedder=self.embedder, top_k=top_k)
        context = self._build_context(references)
        if not context:
            answer = (
                "I could not find any relevant information in the knowledge base. "
                "Please ingest documents before chatting."
            )
            return ChatResponse(answer=answer, references=[])

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    "Context:\n" + context + "\n\n" + f"Question: {question}\n"
                ),
            },
        ]
        answer = self.chat_model.generate(messages)
        return ChatResponse(answer=answer, references=list(references))
