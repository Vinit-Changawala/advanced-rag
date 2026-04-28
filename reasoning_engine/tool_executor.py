# ============================================================
# reasoning_engine/tool_executor.py
#
# PURPOSE: Execute individual plan steps (tools).
#
# BEGINNER CONCEPT - What is a "tool" in AI systems?
# Tools are functions the AI can "call" to get real-world data.
# Without tools, an LLM can only use its training knowledge.
# With tools, it can:
#   - Search our documents (vector_search)
#   - Query our database (sql_query)
#   - Search the web (web_search)
#
# The ToolExecutor is the "worker" that runs each tool
# and returns the results back to the Reasoning Engine.
# ============================================================

import logging
import time
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    Executes tools defined in the plan steps.

    Each tool is a registered function. When the planner says
    "execute vector_search", this class finds and runs that function.

    Usage:
        executor = ToolExecutor(vector_store=store, relational_db=db)
        result = executor.execute({
            "action": "vector_search",
            "search_query": "revenue growth Q3",
        })
    """

    def __init__(self, vector_store=None, relational_db=None,
                 llm_client=None):
        self.vector_store = vector_store
        self.relational_db = relational_db
        self.llm_client = llm_client

        # Tool registry: maps action name → method
        # This pattern is called a "strategy pattern"
        self._tools = {
            "vector_search": self._tool_vector_search,
            "sql_query":     self._tool_sql_query,
            "synthesize":    self._tool_synthesize,
            "validate":      self._tool_validate,
        }

    def execute(self, step: Dict[str, Any],
                context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute a single plan step.

        Args:
            step: A step dict from the planner, e.g.:
                  {"action": "vector_search", "search_query": "..."}
            context: Accumulated results from previous steps

        Returns:
            Result dict with "success", "data", "error", "latency_ms"
        """
        action = step.get("action")
        start = time.time()

        if action not in self._tools:
            return {
                "success": False,
                "data": None,
                "error": f"Unknown tool: {action}",
                "latency_ms": 0
            }

        logger.info(f"Executing tool: {action}")

        try:
            # Call the tool function
            result_data = self._tools[action](step, context or {})
            latency_ms = int((time.time() - start) * 1000)

            return {
                "success": True,
                "data": result_data,
                "error": None,
                "latency_ms": latency_ms,
                "action": action,
            }

        except Exception as e:
            latency_ms = int((time.time() - start) * 1000)
            logger.error(f"Tool {action} failed: {e}")
            return {
                "success": False,
                "data": None,
                "error": str(e),
                "latency_ms": latency_ms,
                "action": action,
            }

    def execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute all steps in a plan in order.

        Respects step dependencies: step 3 won't run until
        its dependency (step 2) has completed.

        Args:
            plan: Full plan dict from Planner.create_plan()

        Returns:
            Dict with all step results and final accumulated context
        """
        steps = plan.get("steps", [])
        context = {
            "query": plan.get("original_query", ""),
            "step_results": {},        # Stores results keyed by step_number
            "retrieved_chunks": [],    # All chunks found across all searches
        }

        for step in steps:
            step_num = step.get("step_number", 0)

            # Check dependencies: skip step if a dependency failed
            dependencies = step.get("depends_on", [])
            failed_deps = [
                d for d in dependencies
                if not context["step_results"].get(d, {}).get("success", False)
            ]
            if failed_deps:
                logger.warning(f"Skipping step {step_num}: dependencies {failed_deps} failed")
                context["step_results"][step_num] = {
                    "success": False,
                    "error": f"Dependency failed: {failed_deps}"
                }
                continue

            # Execute the step
            result = self.execute(step, context)
            context["step_results"][step_num] = result

            # Accumulate retrieved chunks into the context
            if result["success"] and result.get("data"):
                data = result["data"]
                if isinstance(data, list):
                    context["retrieved_chunks"].extend(data)
                elif isinstance(data, dict) and "chunks" in data:
                    context["retrieved_chunks"].extend(data["chunks"])

        return context

    # ── TOOL IMPLEMENTATIONS ──────────────────────────────────

    def _tool_vector_search(self, step: Dict, context: Dict) -> List[Dict]:
        """Search the vector database for relevant chunks."""
        if not self.vector_store:
            raise RuntimeError("VectorStore not configured in ToolExecutor")

        query = step.get("search_query", context.get("query", ""))
        top_k = step.get("top_k", 5)

        chunks = self.vector_store.search(query=query, top_k=top_k)
        logger.info(f"Vector search returned {len(chunks)} chunks for: {query[:60]}")
        return chunks

    def _tool_sql_query(self, step: Dict, context: Dict) -> Dict:
        """Execute a structured query against the relational database."""
        if not self.relational_db:
            raise RuntimeError("RelationalDB not configured in ToolExecutor")

        # In a real system, the LLM would generate the SQL.
        # Here we return stats as a safe default.
        stats = self.relational_db.get_answer_stats()
        return {"type": "stats", "data": stats}

    def _tool_synthesize(self, step: Dict, context: Dict) -> Dict:
        """
        Generate a final answer from all retrieved chunks.

        This is the "generation" step in RAG:
        Retrieved chunks → LLM → Final answer
        """
        if not self.llm_client:
            # Simple fallback: concatenate top chunk texts
            chunks = context.get("retrieved_chunks", [])
            if chunks:
                return {"answer": chunks[0].get("text", "No answer found."), "sources": []}
            return {"answer": "No relevant information found.", "sources": []}

        chunks = context.get("retrieved_chunks", [])
        query = context.get("query", "")

        if not chunks:
            return {"answer": "I could not find relevant information to answer your question.", "sources": []}

        import os as _os

        def _clean(p):
            n = _os.path.basename(p)
            return n[37:] if len(n) > 37 and n[36] == "_" else n

        # Build context with CLEAN filenames — no ugly /tmp/uuid_ paths shown to LLM
        context_text = "\n\n---\n\n".join([
            f"[Doc {i+1}: {_clean(c.get('source',''))}]\n{c.get('text','')}"
            for i, c in enumerate(chunks[:8])
        ])

        prompt = f"""You are answering a question based ONLY on the document excerpts below.
STRICT RULE: If the excerpts do not contain enough information to answer the question, you MUST respond with exactly:
"The uploaded documents do not contain information about this topic. Please upload relevant documents."
Do NOT use any outside knowledge. Do NOT answer from memory.

Question: {query}

Document excerpts:
{context_text}

Answer (based ONLY on the excerpts above, nothing else):"""

        response = self.llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.2
        )

        answer_text = response.choices[0].message.content.strip()
        sources = list({c.get("source", "") for c in chunks})

        return {
            "answer": answer_text,
            "sources": sources,
            "chunks_used": len(chunks),
            "tokens_used": response.usage.total_tokens,
        }

    def _tool_validate(self, step: Dict, context: Dict) -> Dict:
        """Quick validation check on the generated answer."""
        step_results = context.get("step_results", {})

        # Find the synthesize result
        for result in step_results.values():
            if result.get("action") == "synthesize" and result.get("success"):
                answer = result["data"].get("answer", "")
                # Simple heuristic checks
                has_content = len(answer) > 50
                not_error = "error" not in answer.lower()[:50]
                return {
                    "valid": has_content and not_error,
                    "checks": {
                        "has_content": has_content,
                        "not_error_message": not_error,
                    }
                }

        return {"valid": False, "reason": "No synthesize step found"}
