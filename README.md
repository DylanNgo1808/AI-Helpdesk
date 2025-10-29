# AI-Helpdesk

An opinionated Retrieval-Augmented Generation (RAG) starter kit that lets you
chat with your internal documentation. Point it at your knowledge sources and
ask questions that reference the original docs.

## Features

- Crawl public documentation sites such as [`help.aov.ai`](https://help.aov.ai/)
- Ingest Markdown or plain text exports from Notion
- Chunk, embed, and store documents locally using OpenAI embeddings
- Retrieve the most relevant passages with cosine similarity
- Chat with OpenAI models while automatically citing supporting snippets

## Project layout

```
ai_helpdesk/
├── chat/              # Chat orchestration and OpenAI wrappers
├── ingestion/         # Data ingestion helpers for the supported sources
├── storage/           # Lightweight on-disk vector store
├── utils/             # Shared utilities (chunking, etc.)
├── cli.py             # Command line interface entry points
├── models.py          # Document & retrieval dataclasses
main.py                # Console script shim
config.example.json    # Sample ingestion configuration
```

## Getting started

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Export your OpenAI API key:

   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

3. Ingest your documentation. You can either rely on a JSON configuration file
   (see `config.example.json`) or pass the sources directly via the CLI:

   ```bash
   python main.py ingest --config config.example.json
   # or crawl a single website
   python main.py ingest --web-url https://help.aov.ai/ --max-pages 25
   # or ingest a local Notion export
   python main.py ingest --notion-file path/to/notion_export.md
   ```

4. Start chatting with the embedded knowledge base:

   ```bash
   python main.py chat
   ```

   Ask a question and the assistant will respond with citations and relevance
   scores for the snippets that informed the answer.

5. Prefer a browser experience? Launch the web interface:

   ```bash
   python main.py web --host 0.0.0.0 --port 8000
   ```

   Then open <http://localhost:8000> to chat with the assistant and browse the
   cited references.

## Configuration reference

The ingestion CLI accepts an optional JSON configuration with the following
shape:

```json
{
  "web": [
    {
      "url": "https://help.aov.ai/",
      "max_pages": 30,
      "delay": 1.0,
      "allowed_paths": ["/"]
    }
  ],
  "notion": [
    {
      "id": "notion-handbook",
      "path": "./docs/notion_handbook.md"
    }
  ],
  "chunk_size": 600,
  "chunk_overlap": 120
}
```

- `web`: A list of sites to crawl. Restrict crawling by specifying `allowed_paths`.
- `notion`: Files exported from Notion (Markdown or plain text).
- `chunk_size`/`chunk_overlap`: Override the chunking strategy used before
  embedding.

## Data directory

Embedded vectors and chunk metadata are saved under `data/store` by default.
You can change the location with the global `--store-dir` flag.

## Limitations

- The CLI currently relies on the OpenAI API for both embeddings and chat. Make
  sure you have sufficient quota before running bulk ingestion.
- The web crawler intentionally avoids aggressive scraping behaviour. Increase
  `max_pages` and reduce `delay` cautiously and abide by the target site's
  terms of service.
