"""FastAPI application that exposes a lightweight chat UI."""

from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..chat.engine import ChatEngine
from ..chat.llm import OpenAIEmbedder
from ..storage.vector_store import VectorStore


class ChatRequest(BaseModel):
    """Request payload for the chat endpoint."""

    question: str = Field(..., description="User question to answer")
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Maximum number of reference chunks to retrieve",
    )


class ReferencePayload(BaseModel):
    """Serialized reference returned to the UI."""

    chunk_id: str
    document_id: str
    citation: str | None
    score: float
    content: str
    source: str


class ChatResponsePayload(BaseModel):
    """Response payload for the chat endpoint."""

    answer: str
    references: List[ReferencePayload]


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>AI Helpdesk</title>
    <style>
      :root {
        color-scheme: light dark;
        --bg: #0f172a;
        --bg-light: #f8fafc;
        --text: #0f172a;
        --text-light: #e2e8f0;
        --accent: #6366f1;
        --border: #1e293b;
      }
      body {
        margin: 0;
        font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: var(--bg);
        color: var(--text-light);
        display: flex;
        min-height: 100vh;
        justify-content: center;
        align-items: center;
        padding: 24px;
      }
      .card {
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 16px;
        width: min(960px, 100%);
        box-shadow: 0 24px 60px rgba(15, 23, 42, 0.35);
        display: flex;
        flex-direction: column;
        overflow: hidden;
      }
      header {
        padding: 24px 32px 16px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
      }
      header h1 {
        margin: 0 0 8px;
        font-size: 1.5rem;
      }
      header p {
        margin: 0;
        color: rgba(226, 232, 240, 0.75);
      }
      main {
        padding: 24px 32px;
        display: flex;
        flex-direction: column;
        gap: 16px;
      }
      .chat-log {
        flex: 1;
        min-height: 320px;
        max-height: 480px;
        overflow-y: auto;
        padding-right: 8px;
      }
      .message {
        margin-bottom: 18px;
      }
      .message strong {
        display: block;
        margin-bottom: 6px;
        color: var(--accent);
      }
      .message p {
        margin: 0 0 8px;
        white-space: pre-wrap;
        line-height: 1.6;
      }
      .references {
        margin: 0;
        padding-left: 18px;
        font-size: 0.9rem;
        color: rgba(226, 232, 240, 0.75);
      }
      form {
        display: flex;
        gap: 12px;
      }
      textarea {
        flex: 1;
        resize: vertical;
        min-height: 72px;
        max-height: 160px;
        padding: 12px;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.4);
        background: rgba(15, 23, 42, 0.6);
        color: inherit;
        font-size: 1rem;
        line-height: 1.5;
      }
      textarea:focus {
        outline: 2px solid rgba(99, 102, 241, 0.6);
        outline-offset: 2px;
      }
      button {
        background: var(--accent);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0 24px;
        font-size: 1rem;
        cursor: pointer;
        transition: background 0.2s ease;
      }
      button:hover {
        background: #4f46e5;
      }
      button:disabled {
        background: rgba(148, 163, 184, 0.4);
        cursor: not-allowed;
      }
      footer {
        border-top: 1px solid rgba(148, 163, 184, 0.2);
        padding: 16px 32px;
        text-align: right;
        font-size: 0.85rem;
        color: rgba(226, 232, 240, 0.55);
      }
      @media (prefers-color-scheme: light) {
        body {
          background: var(--bg-light);
          color: var(--text);
        }
        .card {
          background: white;
          border-color: rgba(15, 23, 42, 0.08);
          box-shadow: 0 30px 80px rgba(15, 23, 42, 0.1);
        }
        header, footer {
          border-color: rgba(15, 23, 42, 0.08);
        }
        header p,
        .references,
        footer {
          color: rgba(15, 23, 42, 0.6);
        }
        textarea {
          background: rgba(248, 250, 252, 0.9);
          color: inherit;
          border-color: rgba(15, 23, 42, 0.1);
        }
      }
    </style>
  </head>
  <body>
    <div class="card">
      <header>
        <h1>AI Helpdesk</h1>
        <p>Ask questions about your documentation and receive answers with citations.</p>
      </header>
      <main>
        <div class="chat-log" id="chat-log"></div>
        <form id="chat-form">
          <textarea id="question" placeholder="Ask a question about your docs..." required></textarea>
          <button type="submit" id="submit-btn">Send</button>
        </form>
      </main>
      <footer>
        Built with FastAPI · Powered by OpenAI
      </footer>
    </div>
    <script>
      const form = document.getElementById("chat-form");
      const questionInput = document.getElementById("question");
      const chatLog = document.getElementById("chat-log");
      const submitBtn = document.getElementById("submit-btn");

      function appendMessage(role, content, references = []) {
        const wrapper = document.createElement("div");
        wrapper.className = "message";
        const header = document.createElement("strong");
        header.textContent = role === "user" ? "You" : "Assistant";
        const message = document.createElement("p");
        message.textContent = content;
        wrapper.appendChild(header);
        wrapper.appendChild(message);

        if (references.length > 0) {
          const list = document.createElement("ul");
          list.className = "references";
          references.forEach((ref, index) => {
            const item = document.createElement("li");
            const label = ref.citation || ref.chunk_id;
            item.textContent = `${index + 1}. ${label} (score ${ref.score.toFixed(3)})`;
            list.appendChild(item);
          });
          wrapper.appendChild(list);
        }

        chatLog.appendChild(wrapper);
        chatLog.scrollTop = chatLog.scrollHeight;
      }

      async function submitQuestion(event) {
        event.preventDefault();
        const question = questionInput.value.trim();
        if (!question) {
          return;
        }
        appendMessage("user", question);
        submitBtn.disabled = true;
        questionInput.value = "";
        try {
          const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question })
          });
          if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Request failed");
          }
          const data = await response.json();
          appendMessage("assistant", data.answer, data.references);
        } catch (err) {
          appendMessage("assistant", `Error: ${err.message}`);
        } finally {
          submitBtn.disabled = false;
          questionInput.focus();
        }
      }

      form.addEventListener("submit", submitQuestion);
    </script>
  </body>
</html>
"""


def create_app(*, store_dir: str | Path = "data/store", default_top_k: int = 5) -> FastAPI:
    """Instantiate the FastAPI app with shared dependencies."""

    store = VectorStore(store_dir)
    embedder = OpenAIEmbedder()
    engine = ChatEngine(store, embedder=embedder)

    app = FastAPI(title="AI Helpdesk", version="1.0.0")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return HTML_PAGE

    @app.get("/healthz")
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/chat", response_model=ChatResponsePayload)
    async def chat_endpoint(payload: ChatRequest) -> ChatResponsePayload:
        question = payload.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")
        top_k = payload.top_k or default_top_k
        response = engine.ask(question, top_k=top_k)
        references = [
            ReferencePayload(
                chunk_id=result.chunk.id,
                document_id=result.chunk.document_id,
                citation=result.citation,
                score=result.score,
                content=result.chunk.content,
                source=result.chunk.source,
            )
            for result in response.references
        ]
        return ChatResponsePayload(answer=response.answer, references=references)

    return app
