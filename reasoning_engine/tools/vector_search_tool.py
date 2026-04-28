# ============================================================
# reasoning_engine/tools/vector_search_tool.py
#
# PURPOSE: Standalone tool wrapper for vector similarity search.
# This wraps the VectorStore's search() into a clean "tool" interface.
# ============================================================

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class VectorSearchTool:
    """
    Tool that searches the vector database for semantically similar chunks.

    BEGINNER CONCEPT - Why wrap VectorStore in a Tool?
    The VectorStore handles ALL database operations (insert, delete, search).
    This tool only exposes SEARCH to the reasoning engine.
    It also adds extra logic: re-ranking, deduplication, filtering.

    Usage:
        tool = VectorSearchTool(vector_store=store)
        results = tool.run("What is our refund policy?", top_k=5)
    """

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def run(self, query: str, top_k: int = 5,
            source_filter: Optional[str] = None,
            min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for chunks relevant to the query.

        Args:
            query: The search query
            top_k: Number of results to return
            source_filter: Only return results from this source file
            min_score: Minimum similarity score (0.0 - 1.0)

        Returns:
            List of chunk dicts sorted by relevance score
        """
        filter_dict = {}
        if source_filter:
            filter_dict["source"] = source_filter

        raw_results = self.vector_store.search(
            query=query,
            top_k=top_k * 2,           # Fetch extra so we can filter
            filter_dict=filter_dict or None
        )

        # Filter by minimum score
        filtered = [r for r in raw_results if r.get("score", 0) >= min_score]

        # Deduplicate: remove chunks with nearly identical text
        deduplicated = self._deduplicate(filtered)

        # Return only top_k after filtering
        return deduplicated[:top_k]

    def _deduplicate(self, chunks: List[Dict]) -> List[Dict]:
        """Remove chunks that have very similar text (overlap > 80%)."""
        seen_texts = []
        unique = []

        for chunk in chunks:
            text = chunk.get("text", "")
            # Check if this chunk is too similar to an already-seen chunk
            is_duplicate = any(
                self._similarity(text, seen) > 0.8
                for seen in seen_texts
            )
            if not is_duplicate:
                unique.append(chunk)
                seen_texts.append(text)

        return unique

    def _similarity(self, text1: str, text2: str) -> float:
        """Simple character-overlap similarity score (0.0 - 1.0)."""
        if not text1 or not text2:
            return 0.0
        # Use shorter text length as denominator
        shorter = min(len(text1), len(text2))
        # Count matching characters in same positions
        matches = sum(c1 == c2 for c1, c2 in zip(text1[:shorter], text2[:shorter]))
        return matches / shorter
