# ============================================================
# multi_agent_system/agent_2.py  —  Synthesis Agent
#
# PURPOSE: Take ALL information retrieved by Agent 1 (Research Agent)
#          and merge it into ONE clear, coherent answer.
#
# ANALOGY:
# Imagine a research team. Agent 1 (Researcher) goes to the library
# and brings back 10 books all containing relevant pages.
# Agent 2 (Synthesizer) reads ALL those pages and writes ONE
# clear report combining the key points, removing repetition,
# and resolving any contradictions.
#
# WHY IS THIS NECESSARY?
# Without synthesis:
#   "Chunk 1 says X, Chunk 2 says X again, Chunk 3 says Y..."
# With synthesis:
#   "The answer is X (supported by sources 1 & 2) and also Y."
# ============================================================

import logging
from typing import Dict, Any, List

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class SynthesisAgent(BaseAgent):
    """
    Agent 2: Information synthesizer and answer writer.

    Receives retrieved chunks from the Research Agent and produces
    a single well-structured answer. It handles:

    - Deduplication: If two chunks say the same thing, say it once
    - Contradiction resolution: If chunks disagree, note the conflict
    - Citation: Every claim is linked back to a source
    - Formatting: Produces a clean, readable response

    Usage:
        agent = SynthesisAgent(llm_client=openai_client)
        result = agent.run(
            task={"instruction": "Synthesize the information"},
            context={"query": "What is X?", "retrieved_chunks": [...]}
        )
        print(result["output"])   # The final synthesized answer
        print(result["sources"])  # List of source files used
    """

    # The system prompt shapes HOW the agent writes answers.
    # It is injected as a "system" message in the LLM call.
    SYSTEM_PROMPT = """You are an expert assistant that reads source documents and answers questions precisely.

Your rules:
1. Read the question carefully. Understand what the user ACTUALLY wants to know.
2. Write a direct, clear answer — like a knowledgeable human would explain it.
3. Use ONLY information from the provided source chunks.
4. Structure your answer:
   - Start with a 1-2 sentence direct answer to the question
   - Then give supporting details with bullet points if there are multiple facts
   - End with the source file name in parentheses
5. DO NOT copy-paste raw text from the source. Rewrite in your own words.
6. DO NOT repeat the same fact multiple times even if it appears in multiple chunks.
7. If the chunks don't contain the answer, say: "The uploaded documents do not contain information about this."
8. Keep the answer focused. If asked "What is X?", answer that — do not dump everything about X."""

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: OpenAI or Anthropic client object.
                       If None, falls back to extractive answer (first chunk).
        """
        super().__init__(
            name="Synthesis Agent",
            llm_client=llm_client,
            vector_store=None       # Synthesis Agent doesn't search — it writes
        )

    def run(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize retrieved chunks into a coherent answer.

        Args:
            task: {"instruction": "Synthesize information about X"}
            context: Must contain:
                     - "query": the original user question
                     - "retrieved_chunks": list of chunk dicts from Research Agent
                     - (optional) "improvement_notes": notes from Critique Agent for retry

        Returns:
            Dict with:
            - "success": bool
            - "output": the synthesized answer string
            - "sources": list of unique source file names
            - "chunks_used": count of chunks that went into the answer
            - "agent": agent name
        """
        chunks: List[Dict] = context.get("retrieved_chunks", [])
        query: str = context.get("query", task.get("instruction", ""))
        improvement_notes: str = context.get("improvement_notes", "")

        # ── GUARD: No chunks to synthesize ──────────────────────
        if not chunks:
            result = {
                "success": False,
                "output": (
                    "I was unable to find relevant information in the knowledge base "
                    "to answer your question. Please check that related documents "
                    "have been ingested into the system."
                ),
                "sources": [],
                "chunks_used": 0,
                "agent": self.name,
            }
            self.remember(task, result)
            return result

        logger.info(
            f"SynthesisAgent: Synthesizing {len(chunks)} chunks "
            f"for query: '{query[:60]}...'"
        )

        # ── USE LLM IF AVAILABLE ─────────────────────────────────
        if self.llm_client:
            answer, sources = self._llm_synthesize(query, chunks, improvement_notes)
        else:
            # Fallback: extractive — just use the most relevant chunk's text
            answer, sources = self._extractive_synthesize(query, chunks)

        result = {
            "success": True,
            "output": answer,
            "sources": sources,
            "chunks_used": len(chunks),
            "agent": self.name,
        }

        self.remember(task, result)
        logger.info(f"SynthesisAgent: Answer generated ({len(answer)} chars)")
        return result

    # ── PRIVATE METHODS ──────────────────────────────────────────

    def _llm_synthesize(self, query: str, chunks: List[Dict],
                        improvement_notes: str = "") -> tuple:
        """
        Use the LLM to write a synthesized answer from chunks.

        We format the chunks as a numbered list with source labels so
        the LLM can cite them properly.
        """
        # Build the formatted context block
        # We cap at 10 chunks to stay within the LLM context window
        import os as _os

        def _clean(p):
            n = _os.path.basename(p)
            return n[37:] if len(n) > 37 and n[36] == "_" else n

        context_parts = []
        for i, chunk in enumerate(chunks[:10], 1):
            source = _clean(chunk.get("source", "unknown"))
            text   = chunk.get("text", "")
            context_parts.append(f"[Excerpt {i} from {source}]\n{text}")

        context_block = "\n\n---\n\n".join(context_parts)

        improvement = f"IMPROVE ON THIS FEEDBACK: {improvement_notes}\n\n" if improvement_notes else ""

        prompt = f"""You are answering a question based ONLY on the document excerpts below.
STRICT RULE: If the excerpts do not contain enough information, respond ONLY with:
"The uploaded documents do not contain information about this topic. Please upload relevant documents."
Never use outside knowledge.

{improvement}Question: {query}

Document excerpts:
{context_block}

Answer (based ONLY on the excerpts, nothing else):"""

        # Call the LLM through the base class helper
        answer = self._call_llm(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.3,     # Slightly creative for natural writing
            max_tokens=1200
        )

        # Extract unique source file names
        sources = list({chunk.get("source", "") for chunk in chunks if chunk.get("source")})

        return answer, sources

    def _extractive_synthesize(self, query: str,
                                chunks: List[Dict]) -> tuple:
        """
        Fallback when no LLM is available.

        Simply returns the text of the most relevant chunk(s).
        Not as good as LLM synthesis but always works.
        """
        # Sort by score (highest first)
        sorted_chunks = sorted(chunks, key=lambda c: c.get("score", 0), reverse=True)

        # Take the top 3 chunks and concatenate them
        top_chunks = sorted_chunks[:3]
        answer_parts = []

        for chunk in top_chunks:
            source = chunk.get("source", "unknown")
            text = chunk.get("text", "")
            answer_parts.append(text)

        answer = "\n\n".join(answer_parts)
        sources = [c.get("source", "") for c in top_chunks if c.get("source")]

        return answer, sources
