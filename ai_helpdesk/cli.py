"""Command line interface for the AI Helpdesk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .chat.engine import ChatEngine
from .chat.llm import OpenAIEmbedder
from .ingestion.notion import load_notion_export
from .ingestion.web import crawl_website
from .models import Document
from .storage.vector_store import VectorStore


def _load_config(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():  # pragma: no cover - CLI guard
        raise FileNotFoundError(f"Configuration file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _ingest_from_config(config: Dict[str, Any], store: VectorStore, embedder: OpenAIEmbedder) -> None:
    documents: List[Document] = []

    for web_source in config.get("web", []):
        docs = crawl_website(
            web_source["url"],
            max_pages=web_source.get("max_pages", 50),
            delay=web_source.get("delay", 0.5),
            allowed_paths=web_source.get("allowed_paths"),
        )
        documents.extend(docs)

    for notion_source in config.get("notion", []):
        docs = load_notion_export(notion_source["path"], source_id=notion_source.get("id", "notion"))
        documents.extend(docs)

    if not documents:
        print("No documents found in configuration. Nothing to ingest.")
        return

    store.add_documents(
        documents,
        embedder=embedder,
        chunk_size=config.get("chunk_size", 500),
        chunk_overlap=config.get("chunk_overlap", 100),
    )
    print(f"Ingested {len(documents)} documents into the vector store.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Helpdesk CLI")
    parser.add_argument(
        "--store-dir",
        default="data/store",
        help="Directory used to persist the vector store (default: data/store)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a JSON configuration file for ingestion",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents from configured sources")
    ingest_parser.add_argument(
        "--web-url",
        help="Single web URL to crawl (overrides configuration file)",
    )
    ingest_parser.add_argument(
        "--max-pages",
        type=int,
        default=20,
        help="Maximum number of pages to crawl when using --web-url",
    )
    ingest_parser.add_argument(
        "--notion-file",
        type=Path,
        help="Path to a Notion export file (overrides configuration file)",
    )

    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat session")
    chat_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of document chunks to retrieve for each query",
    )

    web_parser = subparsers.add_parser("web", help="Launch the web interface")
    web_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for the web server (default: 127.0.0.1)",
    )
    web_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the web server (default: 8000)",
    )
    web_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of document chunks to retrieve for each query",
    )

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    store = VectorStore(args.store_dir)
    embedder = OpenAIEmbedder()

    if args.command == "ingest":
        config = _load_config(args.config)
        documents: List[Document] = []

        if args.web_url:
            documents.extend(
                crawl_website(args.web_url, max_pages=args.max_pages)
            )
        if args.notion_file:
            documents.extend(load_notion_export(args.notion_file))

        if documents:
            store.add_documents(documents, embedder=embedder)
            print(f"Ingested {len(documents)} documents into the vector store.")
        else:
            _ingest_from_config(config, store, embedder)

    elif args.command == "chat":
        engine = ChatEngine(store, embedder=embedder)
        print("Enter your questions. Press Ctrl+C or Ctrl+D to exit.\n")
        try:
            while True:
                question = input("?> ").strip()
                if not question:
                    continue
                response = engine.ask(question, top_k=args.top_k)
                print("\n" + response.answer + "\n")
                if response.references:
                    print("References:")
                    for result in response.references:
                        citation = result.citation or result.chunk.id
                        print(f"- {citation} (score={result.score:.3f})")
                    print()
        except (KeyboardInterrupt, EOFError):  # pragma: no cover - interactive session
            print("\nGoodbye!")
    elif args.command == "web":
        from .web import create_app

        try:
            import uvicorn
        except ImportError as exc:  # pragma: no cover - runtime guard
            raise SystemExit(
                "uvicorn is required to launch the web server. Install the optional"
                " dependencies with 'pip install fastapi uvicorn'."
            ) from exc

        app = create_app(store_dir=args.store_dir, default_top_k=args.top_k)
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
