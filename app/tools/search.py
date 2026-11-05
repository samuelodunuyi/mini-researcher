"""Tavily web search: discovers candidate URLs. Results are candidates, never evidence."""

import logging
import time

from tavily import TavilyClient

from app.schemas import SearchHit

log = logging.getLogger(__name__)

SNIPPET_CHARS = 400


class SearchError(RuntimeError):
    pass


class TavilySearch:
    def __init__(self, api_key: str, max_results: int = 5, max_retries: int = 2, timeout_s: float = 30,
                 client=None, sleep=time.sleep):
        self._client = client or TavilyClient(api_key=api_key)
        self.max_results = max_results
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self._sleep = sleep

    def __call__(self, query: str) -> list[SearchHit]:
        response, attempt = None, 0
        while response is None:
            attempt += 1
            try:
                response = self._client.search(
                    query=query,
                    max_results=self.max_results,
                    search_depth="basic",
                    include_answer=False,
                    include_raw_content=False,
                    timeout=self.timeout_s,
                )
            except Exception as exc:  # tavily raises several unrelated exception types
                if attempt >= self.max_retries:
                    raise SearchError(f"search failed after {attempt} attempts: {exc}") from exc
                log.warning("search error (%s), retrying", type(exc).__name__)
                self._sleep(2 ** (attempt - 1))

        hits = []
        for r in response.get("results", []):
            url = r.get("url")
            if not url or not url.startswith(("http://", "https://")):
                continue
            hits.append(
                SearchHit(
                    title=r.get("title") or url,
                    url=url,
                    snippet=(r.get("content") or "")[:SNIPPET_CHARS],
                    score=r.get("score"),
                )
            )
        return hits
