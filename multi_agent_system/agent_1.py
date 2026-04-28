# ============================================================
# multi_agent_system/agent_1.py  —  Research Agent
#
# PURPOSE: Deeply search the knowledge base.
# This agent's ONLY job is finding relevant information.
# It runs multiple searches, from different angles,
# to make sure nothing important is missed.
# ============================================================

import logging
from typing import Dict, Any, List
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ResearchAgent(BaseAgent):
    """
    Agent 1: Deep retrieval specialist.

    Runs multiple vector searches using different query formulations
    to maximize recall (finding ALL relevant information).

    BEGINNER CONCEPT - Why multiple searches?
    If you search only for "revenue Q3", you might miss a chunk
    that says "third-quarter income figures".
    By searching multiple ways, we cast a wider net.
    """

    SYSTEM_PROMPT = """You are a research specialist with access to a knowledge base.
Your job is to find ALL information relevant to answering the user's question.
Be thorough - search from multiple angles.
Report exactly what you found, with sources. Do not add opinions."""

    def __init__(self, llm_client=None, vector_store=None):
        super().__init__(
            name="Research Agent",
            llm_client=llm_client,
            vector_store=vector_store
        )

    def run(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute deep research task.

        Strategy:
        1. Search with the original query
        2. Generate 2 alternative query formulations
        3. Search with each alternative
        4. Combine and deduplicate all results
        """
        original_query = context.get("query", task.get("instruction", ""))

        logger.info(f"ResearchAgent: Searching for '{original_query[:60]}'")

        all_chunks = []

        # Search 1: Original query
        if self.vector_store:
            chunks = self.vector_store.search(original_query, top_k=5)
            all_chunks.extend(chunks)
            logger.info(f"Search 1 returned {len(chunks)} chunks")

        # Search 2 & 3: Alternative query formulations (if LLM available)
        alternative_queries = self._generate_alternative_queries(original_query)
        for i, alt_query in enumerate(alternative_queries, 2):
            if self.vector_store:
                chunks = self.vector_store.search(alt_query, top_k=3)
                all_chunks.extend(chunks)
                logger.info(f"Search {i} (alt query) returned {len(chunks)} chunks")

        # Deduplicate by chunk_id
        seen_ids = set()
        unique_chunks = []
        for chunk in all_chunks:
            cid = chunk.get("chunk_id", chunk.get("text", "")[:50])
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique_chunks.append(chunk)

        # Sort by score (most relevant first)
        unique_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)

        result = {
            "success": True,
            "output": unique_chunks,
            "chunks_found": len(unique_chunks),
            "searches_run": 1 + len(alternative_queries),
            "agent": self.name,
        }

        self.remember(task, result)
        return result

    def _generate_alternative_queries(self, query: str) -> List[str]:
        """
        Generate alternative search queries WITHOUT an LLM call.
        
        Using the LLM here consumed 1 of our 2 req/min budget on Mistral free tier
        before we even got to synthesis. Simple keyword variants work well enough
        for retrieval and cost zero API calls.
        """
        q = query.strip().rstrip("?").lower()
        
        # Variant 1: strip common question words to get a noun phrase
        for prefix in ["what is ", "what are ", "how does ", "how do ",
                        "explain ", "describe ", "tell me about "]:
            if q.startswith(prefix):
                noun_phrase = q[len(prefix):]
                return [noun_phrase, f"{noun_phrase} definition overview"]
        
        # Variant 2: just append "definition" and "overview"
        return [f"{q} definition", f"{q} overview"]


# ============================================================
# multi_agent_system/agent_2.py  —  Synthesis Agent
#
# PURPOSE: Merge and clean up the research agent's findings.
# Takes ALL retrieved chunks and produces ONE coherent answer.
# ============================================================


class SynthesisAgent(BaseAgent):
    """
    Agent 2: Information synthesizer.

    Takes raw chunks from the Research Agent and:
    1. Removes duplicate information
    2. Resolves contradictions
    3. Writes a clear, coherent answer
    """

    SYSTEM_PROMPT = """You are an expert assistant that reads document excerpts and answers questions clearly.
Rules:
- Write a direct, plain-language answer — do NOT copy text verbatim
- Start with a 1-2 sentence direct answer, then bullet points if needed
- Do NOT include file names, paths, or source references in your answer text
- Do NOT repeat the same information
- Only use information present in the provided excerpts"""

    def __init__(self, llm_client=None):
        super().__init__(name="Synthesis Agent", llm_client=llm_client)

    def run(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize retrieved chunks into a coherent answer."""
        chunks = context.get("retrieved_chunks", [])
        query = context.get("query", "")

        if not chunks:
            return {
                "success": False,
                "output": "No relevant information found in the knowledge base.",
                "agent": self.name,
            }

        logger.info(f"SynthesisAgent: Synthesizing {len(chunks)} chunks")

        import os as _os
        def _cn(p):
            n = _os.path.basename(p)
            return n[37:] if len(n) > 37 and n[36] == "_" else n

        context_text = "\n\n---\n\n".join([
            f"[Excerpt {i+1}]\n{c.get('text', '')}"
            for i, c in enumerate(chunks[:10])
        ])

        prompt = f"""Question: {query}

Document excerpts:
{context_text}

STRICT RULE: Answer using ONLY the information in these excerpts.
If the excerpts do not contain enough information, respond with:
"The uploaded documents do not contain information about this topic. Please upload relevant documents."
Do NOT use outside knowledge or training memory.
Answer:"""

        answer = self._call_llm(prompt, system_prompt=self.SYSTEM_PROMPT, temperature=0.3)
        sources = list({c.get("source", "") for c in chunks if c.get("source")})

        result = {
            "success": True,
            "output": answer,
            "sources": sources,
            "chunks_used": len(chunks),
            "agent": self.name,
        }

        self.remember(task, result)
        return result


# ============================================================
# multi_agent_system/agent_3.py  —  Critique Agent
#
# PURPOSE: Review the synthesized answer for quality and accuracy.
# Acts as a "quality control" before sending to human validation.
# ============================================================

import json


class CritiqueAgent(BaseAgent):
    """
    Agent 3: Quality control validator.

    Reviews the synthesized answer and:
    1. Checks if it answers the original question
    2. Verifies claims are supported by source chunks
    3. Flags hallucinations (made-up content)
    4. Returns a score and optional improvement
    """

    SYSTEM_PROMPT = """You are a quality reviewer. Return ONLY valid JSON:
{"score": 8, "approved": true, "issues": [], "improved_answer": ""}

Rules:
- score: 1-10 (7+ = good enough)
- approved: true if score >= 7
- issues: list of strings (empty if none)
- improved_answer: empty string unless answer needs major rewrite
Return ONLY the JSON object, nothing else."""

    def __init__(self, llm_client=None):
        super().__init__(name="Critique Agent", llm_client=llm_client)

    def run(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Critique the synthesized answer."""
        # Find the synthesis result from context
        synthesis_output = task.get("answer_to_critique", "")
        query = context.get("query", "")
        chunks = context.get("retrieved_chunks", [])

        if not synthesis_output:
            return {"success": False, "output": None, "agent": self.name}

        logger.info("CritiqueAgent: Reviewing synthesized answer")

        # Build source context for fact-checking
        source_text = "\n".join([
            f"- {c.get('text', '')[:200]}"
            for c in chunks[:5]
        ])

        prompt = f"""Question asked: {query}

Answer to review:
{synthesis_output}

Source documents used:
{source_text}

Review this answer and return ONLY a JSON object as specified:"""

        try:
            raw = self._call_llm(
                prompt,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=800
            )

            # Clean and parse JSON - handle truncated responses
            raw = raw.replace("```json", "").replace("```", "").strip()
            # Try to repair truncated JSON by finding last complete field
            try:
                critique = json.loads(raw)
            except json.JSONDecodeError:
                # Attempt to extract what we can from partial JSON
                import re as _re
                score_m   = _re.search(r'"score"\s*:\s*(\d+)', raw)
                appr_m    = _re.search(r'"approved"\s*:\s*(true|false)', raw)
                score_val = int(score_m.group(1)) if score_m else 7
                appr_val  = (appr_m.group(1) == "true") if appr_m else (score_val >= 7)
                critique  = {"score": score_val, "approved": appr_val,
                             "issues": [], "improved_answer": ""}
                logger.warning(f"CritiqueAgent: partial JSON recovered, score={score_val}")

            result = {
                "success": True,
                "output": critique,
                "agent": self.name,
                "approved": critique.get("approved", False),
                "score": critique.get("score", 0),
            }

        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"CritiqueAgent parsing failed: {e}")
            result = {
                "success": False,
                "output": {"score": 5, "approved": False, "issues": [str(e)]},
                "agent": self.name,
                "approved": False,
                "score": 5,
            }

        self.remember(task, result)
        return result
