from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from .config import get_settings
from .schemas import SourceItem

logger = logging.getLogger(__name__)


def _dedupe_sources(sources: List[SourceItem], limit: int) -> List[SourceItem]:
    seen = set()
    deduped: List[SourceItem] = []
    for s in sources:
        key = s.url.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(s)
        if len(deduped) >= limit:
            break
    return deduped


def _search_with_tavily(query: str, max_results: int, api_key: str) -> Tuple[List[SourceItem], str]:
    # Lazy import to keep optional dependency behavior clear.
    from tavily import TavilyClient  # type: ignore

    client = TavilyClient(api_key=api_key)
    resp = client.search(
        query=query,
        search_depth="basic",
        max_results=max_results,
        include_answer=False,
        include_raw_content=False,
    )
    sources: List[SourceItem] = []
    for r in resp.get("results", []) or []:
        url = r.get("url")
        title = r.get("title") or url or "Source"
        if url:
            sources.append(SourceItem(title=title, url=url))
    return sources, "tavily"


def _search_with_duckduckgo(query: str, max_results: int) -> Tuple[List[SourceItem], str]:
    # DuckDuckGoSearch is free and does not require an API key, but may be rate-limited in some envs.
    from duckduckgo_search import DDGS  # type: ignore

    sources: List[SourceItem] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            href = r.get("href") or r.get("url")
            title = r.get("title") or href or "Source"
            if href:
                sources.append(SourceItem(title=title, url=href))
    return sources, "duckduckgo"


# PUBLIC_INTERFACE
def search_recent_surat_events(query: str, max_results: Optional[int] = None) -> Tuple[List[SourceItem], str]:
    """Search the web for recent events in Surat and return sources plus the search backend name."""
    settings = get_settings()
    limit = max_results if max_results is not None else settings.search_max_results

    # Hint the search engine toward recency and relevance.
    augmented_query = f"{query.strip()} Surat events recent 2025 2024"

    if settings.tavily_api_key:
        try:
            sources, backend = _search_with_tavily(augmented_query, limit, settings.tavily_api_key)
            sources = _dedupe_sources(sources, limit=limit)
            if sources:
                return sources, backend
        except Exception:
            logger.exception("Tavily search failed; falling back to DuckDuckGo.")

    try:
        sources, backend = _search_with_duckduckgo(augmented_query, limit)
        sources = _dedupe_sources(sources, limit=limit)
        return sources, backend
    except Exception as e:
        logger.exception("DuckDuckGo search failed.")
        raise RuntimeError(
            "Web search is currently unavailable. Please try again later, or configure TAVILY_API_KEY."
        ) from e
