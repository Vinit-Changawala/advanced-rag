# ============================================================
# evaluation/latency_cost.py
#
# PURPOSE: Track how fast and cheap each query is.
#
# WHY TRACK THIS?
# AI systems can be EXPENSIVE. GPT-4o costs $5 per 1M tokens.
# If each query uses 5000 tokens, 1000 queries = $25.
# Tracking helps you optimize cost without hurting quality.
#
# LATENCY = How long it takes to get an answer.
# Anything over 10 seconds feels slow to users.
# Tracking shows you where the bottlenecks are.
# ============================================================

import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# OpenAI pricing per 1000 tokens (as of early 2025, update as needed)
# Check: https://openai.com/pricing
TOKEN_COSTS = {
    "gpt-4o":           {"input": 0.0025, "output": 0.01},     # per 1K tokens
    "gpt-4o-mini":      {"input": 0.00015, "output": 0.0006},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
}


class LatencyCostTracker:
    """
    Tracks response time and token usage for each query.

    Usage:
        tracker = LatencyCostTracker()

        with tracker.measure("query_123") as ctx:
            # ... do work ...
            ctx.add_tokens(model="gpt-4o", input_tokens=500, output_tokens=200)

        report = tracker.get_report("query_123")
        print(report["latency_ms"])    # e.g., 2340
        print(report["cost_usd"])      # e.g., 0.00325
    """

    def __init__(self):
        # In-memory store: {query_id: measurement_data}
        self._measurements: Dict[str, Dict] = {}

    class MeasurementContext:
        """Context manager for measuring a block of code."""

        def __init__(self, tracker, query_id: str):
            self.tracker = tracker
            self.query_id = query_id
            self.start_time = None
            self._tokens = []

        def __enter__(self):
            self.start_time = time.time()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed = time.time() - self.start_time
            self.tracker._measurements[self.query_id] = {
                "query_id": self.query_id,
                "latency_ms": int(elapsed * 1000),
                "tokens": self._tokens,
                "timestamp": datetime.now().isoformat(),
            }

        def add_tokens(self, model: str, input_tokens: int,
                       output_tokens: int = 0):
            """Record token usage for a specific model call."""
            self._tokens.append({
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            })

    def measure(self, query_id: str) -> "LatencyCostTracker.MeasurementContext":
        """Start measuring a query. Use as a context manager (with statement)."""
        return self.MeasurementContext(self, query_id)

    def get_report(self, query_id: str) -> Dict[str, Any]:
        """Get the full measurement report for a query."""
        data = self._measurements.get(query_id)
        if not data:
            return {"error": f"No measurement found for {query_id}"}

        # Calculate total cost from token usage
        total_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0

        for token_record in data.get("tokens", []):
            model = token_record["model"]
            input_t = token_record["input_tokens"]
            output_t = token_record["output_tokens"]

            total_input_tokens += input_t
            total_output_tokens += output_t

            if model in TOKEN_COSTS:
                costs = TOKEN_COSTS[model]
                # Cost = (tokens / 1000) * price_per_1k
                total_cost += (input_t / 1000) * costs["input"]
                total_cost += (output_t / 1000) * costs["output"]

        return {
            "query_id": query_id,
            "latency_ms": data["latency_ms"],
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "cost_usd": round(total_cost, 6),
            "timestamp": data["timestamp"],
        }

    def get_summary_stats(self) -> Dict[str, float]:
        """Get aggregate stats across all tracked queries."""
        if not self._measurements:
            return {}

        reports = [self.get_report(qid) for qid in self._measurements]
        valid = [r for r in reports if "error" not in r]

        if not valid:
            return {}

        avg_latency = sum(r["latency_ms"] for r in valid) / len(valid)
        total_cost = sum(r["cost_usd"] for r in valid)
        avg_tokens = sum(r["total_tokens"] for r in valid) / len(valid)

        return {
            "total_queries": len(valid),
            "avg_latency_ms": round(avg_latency, 1),
            "total_cost_usd": round(total_cost, 4),
            "avg_tokens_per_query": round(avg_tokens, 0),
            "avg_cost_per_query": round(total_cost / len(valid), 6),
        }


# ============================================================
# evaluation/feedback_loop.py
#
# PURPOSE: Send signals from evaluation back to the Reasoning Engine.
#
# THE FEEDBACK LOOP:
# Answer gets low score → Log the query + answer + issues
# Planner reads these logs → Adjusts future plans
# System improves over time automatically!
# ============================================================

import uuid


class FeedbackLoop:
    """
    Collects poor-performing answers and feeds them back to improve the system.

    BEGINNER CONCEPT - What is a feedback loop?
    Think of a thermostat:
    - It measures room temperature (evaluation)
    - If too cold, it turns on the heater (adjusts behavior)
    - Temperature rises back to target (improvement)

    Our feedback loop:
    - Measures answer quality (LLM Judge)
    - If too low, logs what went wrong (records the failure)
    - Planner reads these logs to avoid same mistakes (adjusts behavior)

    Usage:
        loop = FeedbackLoop(db=relational_db, threshold=7.0)
        loop.process(query, answer, evaluation_scores, source_chunks)
    """

    def __init__(self, relational_db=None, threshold: float = 7.0):
        self.db = relational_db
        self.threshold = threshold
        self._pending_feedback: List[Dict] = []   # In-memory buffer

    def process(self, query: str, answer: str,
                evaluation: Dict[str, Any],
                source_chunks: List[Dict] = None) -> Optional[Dict]:
        """
        Process evaluation results and record feedback if needed.

        Returns:
            Feedback record if score was below threshold, else None
        """
        overall_score = evaluation.get("overall", 10.0)

        if overall_score >= self.threshold:
            # Answer is good → no feedback needed
            return None

        # Identify what went wrong
        issue_type = self._classify_issue(evaluation)

        feedback = {
            "feedback_id": str(uuid.uuid4()),
            "source": "evaluation",
            "issue_type": issue_type,
            "details": evaluation.get("reasoning", ""),
            "original_query": query,
            "original_answer": answer[:500],
            "scores": {k: v for k, v in evaluation.items() if isinstance(v, (int, float))},
            "suggested_fix": self._suggest_fix(issue_type, evaluation),
            "timestamp": datetime.now().isoformat(),
        }

        # Save to database
        if self.db:
            try:
                self.db.save_feedback(feedback)
            except Exception as e:
                logger.error(f"Failed to save feedback: {e}")

        # Also keep in memory buffer (for quick access)
        self._pending_feedback.append(feedback)

        logger.info(
            f"Feedback recorded: score={overall_score:.1f}, "
            f"issue={issue_type}, query='{query[:60]}'"
        )

        return feedback

    def get_pending_feedback(self, limit: int = 50) -> List[Dict]:
        """Get recent low-quality answers for planner to learn from."""
        return self._pending_feedback[-limit:]

    def clear_pending(self):
        """Clear the in-memory buffer after planner has processed it."""
        self._pending_feedback = []

    def _classify_issue(self, evaluation: Dict) -> str:
        """Determine the main category of the problem."""
        scores = {
            "relevance": evaluation.get("relevance", 5),
            "accuracy": evaluation.get("accuracy", 5),
            "completeness": evaluation.get("completeness", 5),
            "clarity": evaluation.get("clarity", 5),
        }
        # The dimension with the lowest score is the main problem
        return min(scores, key=scores.get)

    def _suggest_fix(self, issue_type: str, evaluation: Dict) -> str:
        """Generate a suggestion for how to fix the issue."""
        suggestions = {
            "relevance": "Improve query parsing — the search is finding off-topic chunks",
            "accuracy": "Strengthen source filtering — check for hallucinations",
            "completeness": "Increase top_k — not enough chunks are being retrieved",
            "clarity": "Improve the synthesis prompt for clearer writing",
        }
        return suggestions.get(issue_type, "Review the full pipeline for this query type")
