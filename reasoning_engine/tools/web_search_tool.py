# ============================================================
# reasoning_engine/tools/web_search_tool.py
#
# PURPOSE: Search the live web when the knowledge base has no answer.
#
# WHEN IS THIS USED?
# The RAG system primarily answers from YOUR documents.
# But sometimes a query needs fresh or external data:
# - "What is the latest Python version?" (changes over time)
# - "What are competitors doing?" (not in your docs)
# - "What happened in the news today?" (not ingested yet)
#
# In these cases, the Conditional Router or Tool Executor
# can fall back to a live web search via the Serper API.
#
# SERPER API:
# Serper.dev provides Google Search results via a simple REST API.
# Free tier: 2,500 queries/month.
# Paid: ~$0.001 per query (very cheap).
# Sign up at: https://serper.dev/
#
# FLOW:
# User query → vector search (no good results) → web search → synthesize
# ============================================================

import logging
import requests
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class WebSearchTool:
    """
    Searches the web using the Serper API (Google Search).

    Used as a fallback when:
    - Vector store returns no results above the confidence threshold
    - The query explicitly requires up-to-date information
    - The planner includes a "web_search" step

    Usage:
        tool = WebSearchTool(api_key="your-serper-api-key")
        results = tool.run("Latest developments in RAG systems 2025")

        for result in results:
            print(result["title"])
            print(result["snippet"])
            print(result["url"])
    """

    # Serper API endpoint
    SEARCH_URL = "https://google.serper.dev/search"

    # Maximum characters to keep from each snippet
    # (keeps token count manageable when feeding results to the LLM)
    MAX_SNIPPET_LENGTH = 400

    def __init__(self, api_key: Optional[str] = None,
                 timeout_seconds: int = 10):
        """
        Args:
            api_key: Your Serper API key.
                     If None, web search is disabled (returns empty list).
            timeout_seconds: How long to wait for the API before giving up.
        """
        self.api_key = api_key
        self.timeout = timeout_seconds

        if not api_key:
            logger.warning(
                "WebSearchTool: No API key provided. "
                "Web search disabled. Set SERPER_API_KEY in your .env file."
            )

    def run(self, query: str,
            num_results: int = 5,
            search_type: str = "search") -> List[Dict[str, Any]]:
        """
        Search the web for the given query.

        Args:
            query: The search query string
            num_results: Number of results to return (max 10)
            search_type: "search" (general) or "news" (recent news)

        Returns:
            List of result dicts, each with:
            - "title": Page title
            - "snippet": Short excerpt from the page
            - "url": Full URL
            - "text": Combined title + snippet (used for LLM context)
            - "source": Always "web" (to distinguish from KB chunks)
            - "position": Rank in search results (1 = most relevant)
        """
        if not self.api_key:
            logger.warning("WebSearchTool.run(): No API key — returning empty results")
            return []

        logger.info(f"WebSearchTool: Searching for '{query[:80]}'")

        try:
            response = requests.post(
                self.SEARCH_URL,
                headers={
                    "X-API-KEY": self.api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "q": query,
                    "num": min(num_results, 10),    # Serper max is 10
                    "type": search_type,
                },
                timeout=self.timeout
            )

            # Raise an exception for HTTP error codes (4xx, 5xx)
            response.raise_for_status()

            data = response.json()
            results = self._parse_results(data)

            logger.info(f"WebSearchTool: Got {len(results)} results for '{query[:60]}'")
            return results

        except requests.exceptions.Timeout:
            logger.error(f"WebSearchTool: Request timed out after {self.timeout}s")
            return []

        except requests.exceptions.HTTPError as e:
            logger.error(f"WebSearchTool: HTTP error {e.response.status_code}: {e}")
            return []

        except requests.exceptions.ConnectionError:
            logger.error("WebSearchTool: Could not connect to Serper API")
            return []

        except Exception as e:
            logger.error(f"WebSearchTool: Unexpected error: {e}")
            return []

    def run_news(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search specifically for recent news articles.

        Useful for queries that require up-to-date information.
        Results are ordered by recency, not relevance.

        Args:
            query: News search query
            num_results: Number of news articles to return
        """
        return self.run(query, num_results=num_results, search_type="news")

    def format_for_llm(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results as a readable text block for the LLM.

        Converts the list of results into a structured string
        that the LLM can read and synthesize into an answer.

        Example output:
            [Web Result 1] Python 3.13 Released — python.org
            Python 3.13 adds new features including improved error messages...
            Source: https://python.org/news/3.13

            [Web Result 2] ...

        Args:
            results: Output from run() or run_news()

        Returns:
            Formatted string for inclusion in LLM prompt
        """
        if not results:
            return "No web search results found."

        lines = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            snippet = result.get("snippet", "")
            url = result.get("url", "")

            lines.append(f"[Web Result {i}] {title}")
            if snippet:
                lines.append(snippet)
            lines.append(f"Source: {url}")
            lines.append("")    # Blank line between results

        return "\n".join(lines)

    # ── PRIVATE METHODS ──────────────────────────────────────────

    def _parse_results(self, data: Dict) -> List[Dict[str, Any]]:
        """
        Parse the raw Serper API response into our standard format.

        Serper returns different structures for "organic" search vs "news".
        We normalize both into the same output format.
        """
        results = []

        # Organic search results (standard web search)
        organic = data.get("organic", [])
        for i, item in enumerate(organic, 1):
            title = item.get("title", "")
            snippet = item.get("snippet", "")

            # Truncate long snippets
            if len(snippet) > self.MAX_SNIPPET_LENGTH:
                snippet = snippet[:self.MAX_SNIPPET_LENGTH] + "..."

            results.append({
                "title":    title,
                "snippet":  snippet,
                "url":      item.get("link", ""),
                "text":     f"{title}: {snippet}",   # Combined for embedding
                "source":   "web",
                "position": i,
                "type":     "organic",
            })

        # News results (if search_type was "news")
        news = data.get("news", [])
        for i, item in enumerate(news, 1):
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            date = item.get("date", "")

            if len(snippet) > self.MAX_SNIPPET_LENGTH:
                snippet = snippet[:self.MAX_SNIPPET_LENGTH] + "..."

            results.append({
                "title":    title,
                "snippet":  f"[{date}] {snippet}" if date else snippet,
                "url":      item.get("link", ""),
                "text":     f"{title}: {snippet}",
                "source":   "web_news",
                "position": i,
                "type":     "news",
                "date":     date,
            })

        # Also include the "answerBox" if Serper returned one
        # (Google's featured snippet / direct answer)
        answer_box = data.get("answerBox", {})
        if answer_box:
            answer = answer_box.get("answer", answer_box.get("snippet", ""))
            if answer:
                results.insert(0, {
                    "title":    answer_box.get("title", "Direct Answer"),
                    "snippet":  answer,
                    "url":      answer_box.get("link", ""),
                    "text":     answer,
                    "source":   "web_answer_box",
                    "position": 0,
                    "type":     "answer_box",
                })

        return results

    @property
    def is_available(self) -> bool:
        """Check if the web search tool is configured and available."""
        return bool(self.api_key)
