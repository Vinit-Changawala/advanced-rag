# ============================================================
# reasoning_engine/tools/sql_query_tool.py
#
# PURPOSE: Let the AI query the relational database using natural language.
#
# HOW IT WORKS (Text-to-SQL):
# User asks: "How many documents were processed last week?"
# LLM converts to SQL: "SELECT COUNT(*) FROM chunks WHERE created_at > ..."
# SQL runs → returns number → LLM explains the result
#
# BEGINNER CONCEPT - Why is this useful in RAG?
# Some questions have EXACT answers in structured data:
# "What is the average evaluation score this month?"
# "How many chunks are from PDF files?"
# Vector search can't answer these precisely — SQL can.
# ============================================================

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class SQLQueryTool:
    """
    Converts natural language questions to SQL queries and executes them.

    Usage:
        tool = SQLQueryTool(db=relational_db, llm_client=client)
        result = tool.run("How many chunks do we have from PDF files?")
        print(result["data"])   # The query results
        print(result["sql"])    # The generated SQL (for debugging)
    """

    # Safety: only allow SELECT queries (no DELETE, UPDATE, DROP)
    ALLOWED_STATEMENTS = {"SELECT", "WITH"}

    TEXT_TO_SQL_PROMPT = """You are a SQL expert. Convert the user's question to a SQL query.

Available tables:
- chunks(chunk_id, text, source, source_type, section_title, summary, keywords, chunk_index, created_at)
- answers(answer_id, query_id, query_text, answer_text, evaluation_score, latency_ms, timestamp)
- evaluations(eval_id, answer_id, relevance_score, accuracy_score, overall_score, timestamp)
- feedback(feedback_id, source, issue_type, details, original_query, timestamp)

Rules:
- Only write SELECT queries (no INSERT, UPDATE, DELETE, DROP)
- Return ONLY the SQL query, no explanation
- Use proper SQL syntax for PostgreSQL

Question: {question}

SQL:"""

    def __init__(self, relational_db=None, llm_client=None):
        self.db = relational_db
        self.llm_client = llm_client

    def run(self, question: str) -> Dict[str, Any]:
        """
        Run a natural language query against the database.

        Returns dict with "data", "sql", "success", "error"
        """
        # Step 1: Generate SQL from natural language
        sql = self._generate_sql(question)
        if not sql:
            return {"success": False, "error": "Could not generate SQL", "data": None}

        # Step 2: Safety check - only allow SELECT
        first_word = sql.strip().split()[0].upper()
        if first_word not in self.ALLOWED_STATEMENTS:
            logger.warning(f"Blocked unsafe SQL: {sql[:100]}")
            return {"success": False, "error": "Only SELECT queries allowed", "sql": sql}

        # Step 3: Execute the SQL
        try:
            results = self._execute_sql(sql)
            return {"success": True, "data": results, "sql": sql, "row_count": len(results)}
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            return {"success": False, "error": str(e), "sql": sql, "data": None}

    def _generate_sql(self, question: str) -> Optional[str]:
        """Use LLM to generate SQL from natural language question."""
        if not self.llm_client:
            return None

        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user",
                       "content": self.TEXT_TO_SQL_PROMPT.format(question=question)}],
            max_tokens=200,
            temperature=0.0     # Zero temperature for deterministic SQL generation
        )

        sql = response.choices[0].message.content.strip()
        # Clean up markdown code blocks if model added them
        sql = sql.replace("```sql", "").replace("```", "").strip()
        return sql

    def _execute_sql(self, sql: str):
        """Execute SQL and return results as list of dicts."""
        from sqlalchemy import text

        with self.db.engine.connect() as conn:
            result = conn.execute(text(sql))
            return [dict(row._mapping) for row in result]


# ============================================================
# reasoning_engine/tools/web_search_tool.py
#
# PURPOSE: Search the web for up-to-date information.
# Used when the knowledge base doesn't have the answer.
# ============================================================

import logging
import requests
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class WebSearchTool:
    """
    Searches the web using the Serper API (Google Search API).

    Used as a fallback when:
    - The vector database has no relevant chunks
    - The query needs very recent information
    - The confidence score is too low

    Usage:
        tool = WebSearchTool(api_key="your_serper_api_key")
        results = tool.run("Latest AI research 2025")
    """

    SERPER_URL = "https://google.serper.dev/search"

    def __init__(self, api_key: str = None):
        """
        Args:
            api_key: Serper API key (get one free at serper.dev)
        """
        self.api_key = api_key

    def run(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the web for the query.

        Returns:
            List of result dicts with "title", "snippet", "url"
        """
        if not self.api_key:
            logger.warning("WebSearchTool: No API key configured")
            return []

        try:
            response = requests.post(
                self.SERPER_URL,
                headers={
                    "X-API-KEY": self.api_key,
                    "Content-Type": "application/json"
                },
                json={"q": query, "num": num_results},
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            results = []

            for item in data.get("organic", []):
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "url": item.get("link", ""),
                    "source": "web",
                    "text": f"{item.get('title', '')}: {item.get('snippet', '')}",
                })

            logger.info(f"Web search returned {len(results)} results for: {query[:60]}")
            return results

        except requests.exceptions.Timeout:
            logger.error("Web search timed out")
            return []
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []
