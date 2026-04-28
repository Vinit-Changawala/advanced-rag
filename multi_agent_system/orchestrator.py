# ============================================================
# multi_agent_system/orchestrator.py
#
# PURPOSE: Coordinate all 3 agents to answer a complex query.
#
# BEGINNER CONCEPT - What does an orchestrator do?
# Imagine a restaurant kitchen:
# - Chef 1 (Research Agent): gathers all ingredients
# - Chef 2 (Synthesis Agent): cooks them into a dish
# - Chef 3 (Critique Agent): tastes the dish and approves it
# The Head Chef (Orchestrator): assigns tasks, ensures each
# chef gets what they need, and delivers the final plate.
#
# PIPELINE ORDER:
# Query → ResearchAgent → SynthesisAgent → CritiqueAgent → Result
# ============================================================

import logging
import time
from typing import Dict, Any

from .agent_1 import ResearchAgent, SynthesisAgent, CritiqueAgent

logger = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    """
    Coordinates the 3-agent pipeline for complex queries.

    Usage:
        orchestrator = MultiAgentOrchestrator(
            llm_client=openai_client,
            vector_store=store
        )
        result = orchestrator.run("Complex multi-part question here")
        print(result["final_answer"])
        print(result["critique"]["score"])
    """

    def __init__(self, llm_client=None, vector_store=None,
                 max_critique_retries: int = 2):
        """
        Args:
            llm_client: OpenAI client shared across all agents
            vector_store: Qdrant store for Research Agent
            max_critique_retries: If critique fails, retry synthesis N times
        """
        self.max_critique_retries = max_critique_retries

        # Instantiate all 3 agents
        self.research_agent = ResearchAgent(
            llm_client=llm_client,
            vector_store=vector_store
        )
        self.synthesis_agent = SynthesisAgent(llm_client=llm_client)
        self.critique_agent = CritiqueAgent(llm_client=llm_client)

        logger.info("MultiAgentOrchestrator: All 3 agents ready")

    def run(self, query: str,
            extra_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run the full 3-agent pipeline for a query.

        Args:
            query: The user's question
            extra_context: Any extra data to pass to agents

        Returns:
            Dict with final_answer, sources, critique, pipeline_log, latency_ms
        """
        start_time = time.time()
        pipeline_log = []   # Track each agent's output for debugging

        # Shared context dictionary passed between agents
        context = {
            "query": query,
            "retrieved_chunks": [],
            **(extra_context or {})
        }

        logger.info(f"Orchestrator: Starting pipeline for query: {query[:80]}")

        # ── STEP 1: Research Agent ─────────────────────────────
        logger.info("Orchestrator → Research Agent")
        research_task = {"instruction": f"Find all information relevant to: {query}"}

        research_result = self.research_agent.run(research_task, context)
        pipeline_log.append({"agent": "research", "result": research_result})

        if research_result.get("success"):
            # Move research findings into the shared context
            context["retrieved_chunks"] = research_result.get("output", [])
            logger.info(f"Research complete: {research_result.get('chunks_found', 0)} chunks")
        else:
            logger.warning("Research Agent failed")
            context["retrieved_chunks"] = []

        # ── STEP 2: Synthesis Agent ────────────────────────────
        logger.info("Orchestrator → Synthesis Agent")
        synthesis_task = {"instruction": f"Synthesize the retrieved information to answer: {query}"}

        synthesis_result = self.synthesis_agent.run(synthesis_task, context)
        pipeline_log.append({"agent": "synthesis", "result": synthesis_result})

        current_answer = synthesis_result.get("output", "No answer generated.")

        # ── STEP 3: Critique Agent (with retry loop) ───────────
        logger.info("Orchestrator → Critique Agent")
        final_answer = current_answer
        critique_output = {}

        for attempt in range(self.max_critique_retries + 1):
            critique_task = {
                "instruction": "Review this answer for quality and accuracy",
                "answer_to_critique": current_answer
            }

            critique_result = self.critique_agent.run(critique_task, context)
            pipeline_log.append({
                "agent": "critique",
                "attempt": attempt + 1,
                "result": critique_result
            })

            critique_output = critique_result.get("output", {})
            approved = critique_result.get("approved", False)
            score = critique_result.get("score", 0)

            logger.info(f"Critique attempt {attempt+1}: score={score}, approved={approved}")

            if approved or score >= 7:
                # Answer is good enough — use it
                # If the critique provided an improvement, use that instead
                improved = critique_output.get("improved_answer", "")
                final_answer = improved if improved else current_answer
                break

            elif attempt < self.max_critique_retries:
                # Try to improve the answer based on the critique's feedback
                logger.info("Answer not approved, attempting improvement...")
                issues = critique_output.get("issues", [])
                improvement_instruction = (
                    f"The previous answer had these issues: {issues}. "
                    f"Please write an improved version."
                )
                # Update context with the issues, then re-run synthesis
                context["improvement_notes"] = improvement_instruction
                synthesis_result = self.synthesis_agent.run(
                    {"instruction": improvement_instruction}, context
                )
                current_answer = synthesis_result.get("output", current_answer)

        # ── FINAL RESULT ───────────────────────────────────────
        latency_ms = int((time.time() - start_time) * 1000)
        sources = list({
            c.get("source", "") for c in context["retrieved_chunks"]
            if c.get("source")
        })

        return {
            "final_answer": final_answer,
            "sources": sources,
            "critique": critique_output,
            "critique_score": critique_output.get("score", 0),
            "approved": critique_output.get("approved", False),
            "chunks_used": len(context["retrieved_chunks"]),
            "pipeline_log": pipeline_log,
            "latency_ms": latency_ms,
        }
