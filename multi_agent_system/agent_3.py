# ============================================================
# multi_agent_system/agent_3.py  —  Critique Agent
#
# PURPOSE: Review the synthesized answer and catch problems
#          BEFORE it reaches the user.
#
# ANALOGY:
# Every newspaper has an editor who reviews articles before publishing.
# The journalist (Agent 2) writes the story.
# The editor (Agent 3) checks: Are the facts right? Is it clear?
#          Is anything missing? Are quotes accurate?
# Only after editor approval does the story get published.
#
# WHAT DOES THE CRITIQUE AGENT CHECK?
# 1. Does the answer actually answer the question?
# 2. Is every fact traceable to a source chunk?
# 3. Is there any hallucinated (made-up) content?
# 4. Is the answer clear and well-structured?
# 5. Is anything important missing?
#
# The agent returns a structured JSON with scores and flags.
# If the score is too low, the Orchestrator asks Agent 2 to try again.
# ============================================================

import json
import logging
from typing import Dict, Any, List

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class CritiqueAgent(BaseAgent):
    """
    Agent 3: Quality control and validation specialist.

    Reviews the answer from the Synthesis Agent and returns
    a structured critique with scores and improvement suggestions.

    The critique drives the retry loop in the Orchestrator:
    If approved=False and score < 7, the Orchestrator asks
    the Synthesis Agent to rewrite, passing the issues as context.

    Usage:
        agent = CritiqueAgent(llm_client=openai_client)
        result = agent.run(
            task={"answer_to_critique": "The answer text here..."},
            context={
                "query": "Original question",
                "retrieved_chunks": [...]   # For fact-checking
            }
        )
        print(result["score"])       # e.g., 7
        print(result["approved"])    # True/False
        print(result["output"]["issues"])  # List of problems found
    """

    # Very strict system prompt — critics need to be precise, not nice
    SYSTEM_PROMPT = """You are a strict quality control reviewer for an AI system.
Your job is to find problems with AI-generated answers.

Be HONEST and CRITICAL. Do not be lenient.

You must return ONLY a valid JSON object with exactly these keys:
{
  "score": <integer 1-10>,
  "answers_question": <true or false>,
  "hallucination_detected": <true or false>,
  "issues": ["list of specific problems, empty list if none"],
  "missing_info": ["important topics the question asked about but the answer missed"],
  "approved": <true if score >= 7 and no hallucination, false otherwise>,
  "improved_answer": "<rewrite the answer fixing the issues, or empty string if approved>"
}

Score guide:
1-3: Wrong, misleading, or completely off-topic
4-6: Partially correct but has significant issues  
7-8: Good answer with minor issues
9-10: Excellent, comprehensive, accurate answer"""

    def __init__(self, llm_client=None):
        super().__init__(
            name="Critique Agent",
            llm_client=llm_client,
            vector_store=None       # Critics don't search — they judge
        )

    def run(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Critique the synthesized answer.

        Args:
            task: Must contain "answer_to_critique" — the answer text to review
            context: Must contain "query" and "retrieved_chunks" for fact-checking

        Returns:
            Dict with:
            - "success": bool
            - "output": full critique dict (score, issues, approved, etc.)
            - "score": numeric score 1-10
            - "approved": bool
            - "agent": agent name
        """
        answer_text: str = task.get("answer_to_critique", "")
        query: str = context.get("query", "")
        chunks: List[Dict] = context.get("retrieved_chunks", [])

        # ── GUARD: Nothing to critique ───────────────────────────
        if not answer_text:
            result = {
                "success": False,
                "output": {
                    "score": 0,
                    "approved": False,
                    "issues": ["No answer was provided to critique"],
                    "hallucination_detected": False,
                    "answers_question": False,
                    "missing_info": [],
                    "improved_answer": "",
                },
                "score": 0,
                "approved": False,
                "agent": self.name,
            }
            self.remember(task, result)
            return result

        logger.info(f"CritiqueAgent: Reviewing answer for query: '{query[:60]}'")

        # ── CRITIQUE WITH LLM ────────────────────────────────────
        if self.llm_client:
            critique = self._llm_critique(query, answer_text, chunks)
        else:
            # Fallback: basic heuristic critique (no LLM needed)
            critique = self._heuristic_critique(query, answer_text)

        result = {
            "success": True,
            "output": critique,
            "score": critique.get("score", 5),
            "approved": critique.get("approved", False),
            "agent": self.name,
        }

        self.remember(task, result)
        logger.info(
            f"CritiqueAgent: score={critique.get('score')}, "
            f"approved={critique.get('approved')}, "
            f"issues={len(critique.get('issues', []))}"
        )
        return result

    # ── PRIVATE METHODS ──────────────────────────────────────────

    def _llm_critique(self, query: str, answer: str,
                      chunks: List[Dict]) -> Dict[str, Any]:
        """Use the LLM to critique the answer against the source chunks."""

        # Build a short source summary for fact-checking
        # We only send up to 5 chunks to keep the prompt length manageable
        source_snippets = []
        for i, chunk in enumerate(chunks[:5], 1):
            source = chunk.get("source", "unknown")
            text = chunk.get("text", "")[:250]   # First 250 chars of each chunk
            source_snippets.append(f"[Source {i}: {source}]\n{text}")

        sources_block = "\n\n".join(source_snippets) if source_snippets else "No source chunks available."

        prompt = f"""Review this AI-generated answer for quality and accuracy.

Original Question: {query}

Answer to Review:
{answer}

Source Documents Used to Generate the Answer:
{sources_block}

Return ONLY the JSON critique as specified in your system instructions:"""

        try:
            raw_response = self._call_llm(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.0,      # Zero temperature: deterministic, strict judgement
                max_tokens=600
            )

            # Clean up the response — sometimes models add markdown fences
            cleaned = raw_response.strip()
            if cleaned.startswith("```"):
                # Remove ```json ... ``` wrapper
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]

            critique = json.loads(cleaned)

            # Validate and fill in any missing keys with safe defaults
            critique.setdefault("score", 5)
            critique.setdefault("approved", critique.get("score", 5) >= 7)
            critique.setdefault("issues", [])
            critique.setdefault("missing_info", [])
            critique.setdefault("hallucination_detected", False)
            critique.setdefault("answers_question", True)
            critique.setdefault("improved_answer", "")

            return critique

        except json.JSONDecodeError as e:
            logger.error(f"CritiqueAgent: Could not parse LLM JSON response: {e}")
            # Return a conservative "needs review" result
            return {
                "score": 5,
                "approved": False,
                "issues": [f"Critique parsing failed: {str(e)}"],
                "missing_info": [],
                "hallucination_detected": False,
                "answers_question": True,
                "improved_answer": "",
            }

        except Exception as e:
            logger.error(f"CritiqueAgent LLM call failed: {e}")
            return self._heuristic_critique(query, answer)

    def _heuristic_critique(self, query: str, answer: str) -> Dict[str, Any]:
        """
        Rule-based critique when LLM is not available.

        Checks simple heuristics:
        - Is the answer long enough?
        - Does it contain uncertainty phrases?
        - Does it repeat the question without answering?
        """
        issues = []
        score = 7   # Start optimistic

        answer_lower = answer.lower()

        # Check 1: Answer length
        if len(answer) < 50:
            issues.append("Answer is too short — likely incomplete")
            score -= 2

        # Check 2: Uncertainty / refusal phrases
        uncertainty_phrases = [
            "i don't know", "i cannot", "i'm not sure",
            "no information", "cannot find", "not available"
        ]
        if any(phrase in answer_lower for phrase in uncertainty_phrases):
            issues.append("Answer contains uncertainty — may not have found relevant info")
            score -= 1

        # Check 3: Answer is just the question repeated
        query_words = set(query.lower().split())
        answer_words = set(answer_lower.split())
        overlap = len(query_words & answer_words) / max(len(query_words), 1)
        if overlap > 0.8 and len(answer) < 200:
            issues.append("Answer may be repeating the question rather than answering it")
            score -= 2

        # Check 4: Generic non-answer phrases
        generic_phrases = ["as an ai", "i'm just an ai", "i am an ai assistant"]
        if any(phrase in answer_lower for phrase in generic_phrases):
            issues.append("Answer contains generic AI disclaimers instead of useful content")
            score -= 1

        score = max(1, min(10, score))   # Clamp between 1 and 10
        approved = score >= 7 and len(issues) == 0

        return {
            "score": score,
            "approved": approved,
            "issues": issues,
            "missing_info": [],
            "hallucination_detected": False,    # Can't detect without LLM
            "answers_question": score >= 5,
            "improved_answer": "",
        }
