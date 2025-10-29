"""Utilities for ingesting content from web sources."""

from __future__ import annotations

import time
from collections import deque
from typing import Dict, Iterable, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from ..models import Document

DEFAULT_HEADERS = {"User-Agent": "AI-Helpdesk/1.0"}


def _clean_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript", "header", "footer"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    return " ".join(text.split())


def crawl_website(
    base_url: str,
    *,
    max_pages: int = 50,
    delay: float = 0.5,
    allowed_paths: Optional[Iterable[str]] = None,
    session: Optional[requests.Session] = None,
) -> List[Document]:
    """Crawl a website and return extracted documents.

    Parameters
    ----------
    base_url:
        The starting URL of the website.
    max_pages:
        Upper bound on how many pages to visit.
    delay:
        Delay in seconds between successive requests to avoid hammering the server.
    allowed_paths:
        Optional whitelist of path prefixes to restrict crawling.
    session:
        Optional ``requests.Session`` to reuse HTTP connections.
    """

    parsed_base = urlparse(base_url)
    allowed_netloc = parsed_base.netloc
    allowed_paths = tuple(allowed_paths or ())

    queue = deque([base_url])
    seen = set()
    documents: List[Document] = []
    http = session or requests.Session()

    while queue and len(documents) < max_pages:
        url = queue.popleft()
        if url in seen:
            continue
        seen.add(url)

        try:
            response = http.get(url, headers=DEFAULT_HEADERS, timeout=15)
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - network variability
            print(f"[crawl] Skipping {url}: {exc}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        text = _clean_text(soup)
        title = soup.title.string.strip() if soup.title else url

        documents.append(
            Document(
                id=url,
                source="web",
                content=text,
                metadata={"url": url, "title": title},
            )
        )

        if len(documents) >= max_pages:
            break

        for link in soup.find_all("a"):
            href = link.get("href")
            if not href:
                continue
            absolute = urljoin(url, href)
            parsed = urlparse(absolute)
            if parsed.netloc != allowed_netloc:
                continue
            if allowed_paths and not any(parsed.path.startswith(p) for p in allowed_paths):
                continue
            if parsed.fragment:
                absolute = absolute.split("#", 1)[0]
            if absolute not in seen:
                queue.append(absolute)

        time.sleep(max(0.0, delay))

    return documents
