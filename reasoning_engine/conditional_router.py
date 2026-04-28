# ============================================================
# reasoning_engine/conditional_router.py
#
# PURPOSE: Decide WHERE to send the query after the plan is made.
#
# THREE POSSIBLE ROUTES:
# 1. DIRECT ANSWER  → Simple query, just search + generate
# 2. MULTI-AGENT    → Complex query needing specialist agents
# 3. HUMAN REVIEW   → Sensitive/high-stakes query needs a human
#
# BEGINNER CONCEPT - What is routing?
# Imagine a call center:
# - "What are your hours?" → Automated system handles it (direct)
# - "I have a complex billing dispute" → Senior rep handles it (agent)
# - "Legal complaint" → Legal team handles it (human escalation)
#
# Our router does the same for AI queries.
# ============================================================

import logging
from typing import Dict, Any, Literal

logger = logging.getLogger(__name__)

# Type alias: route can only be one of these three strings
RouteType = Literal["direct", "multi_agent", "human_review"]


class ConditionalRouter:
    """
    Routes queries to the appropriate processing path.

    Usage:
        router = ConditionalRouter()
        route = router.route(plan, query)
        # route is one of: "direct", "multi_agent", "human_review"
    """

    # Keywords that suggest the query needs human review
    SENSITIVE_KEYWORDS = [
        "legal", "lawsuit", "sue", "medical advice", "diagnosis",
        "financial advice", "investment", "stock tip", "confidential",
        "personal data", "delete my data", "gdpr", "privacy",
        "offensive", "harmful", "dangerous"
    ]

    def __init__(self,
                 complexity_threshold: str = "moderate",
                 confidence_threshold: float = 0.6):
        """
        Args:
            complexity_threshold: Minimum complexity to use multi-agent
                                  ("simple" | "moderate" | "complex")
            confidence_threshold: Below this confidence → human review
        """
        self.complexity_threshold = complexity_threshold
        self.confidence_threshold = confidence_threshold

    def route(self, plan: Dict[str, Any], query: str,
              confidence: float = 1.0) -> Dict[str, Any]:
        """
        Determine the processing route for a query.

        Args:
            plan: The plan created by Planner
            query: Original user query
            confidence: System's confidence level (0.0 - 1.0)

        Returns:
            Dict with "route", "reason", and routing metadata
        """
        query_lower = query.lower()

        # ── RULE 1: Human Review for sensitive topics ──
        sensitive_match = [kw for kw in self.SENSITIVE_KEYWORDS if kw in query_lower]
        if sensitive_match:
            return {
                "route": "human_review",
                "reason": f"Sensitive keywords detected: {sensitive_match}",
                "priority": "high",
            }

        # ── RULE 2: Human Review for low confidence ──
        if confidence < self.confidence_threshold:
            return {
                "route": "human_review",
                "reason": f"Low system confidence: {confidence:.2f} < {self.confidence_threshold}",
                "priority": "medium",
            }

        # ── RULE 3: Multi-Agent for complex queries ──
        complexity = plan.get("complexity", "simple")
        requires_agents = plan.get("requires_agents", False)
        num_steps = len(plan.get("steps", []))

        complexity_levels = {"simple": 1, "moderate": 2, "complex": 3}
        plan_level = complexity_levels.get(complexity, 1)
        threshold_level = complexity_levels.get(self.complexity_threshold, 2)

        if requires_agents or plan_level >= threshold_level or num_steps > 4:
            return {
                "route": "multi_agent",
                "reason": f"Complex query: complexity={complexity}, steps={num_steps}",
                "agents_needed": plan_level,
            }

        # ── DEFAULT: Direct answer ──
        return {
            "route": "direct",
            "reason": "Simple query suitable for direct retrieval",
        }

    def should_escalate_to_human(self, evaluation_score: float,
                                  threshold: float = 7.0) -> bool:
        """
        After generating an answer, check if it should go to human review.

        If the evaluation score is too low, a human should review
        before sending the answer to the user.
        """
        return evaluation_score < threshold
