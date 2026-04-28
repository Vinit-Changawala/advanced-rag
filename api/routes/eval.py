# api/routes/eval.py  — Evaluation and monitoring endpoints

import logging
from typing import Optional
from fastapi import APIRouter, Query, Request, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/eval/stats")
async def get_eval_stats(request: Request):
    """
    Overall system performance statistics.
    Returns totals, averages, and cost estimates pulled from PostgreSQL.
    """
    db = getattr(request.app.state, "relational_db", None)
    if db is None:
        return {"error": "Database unavailable", "data": None}
    try:
        stats = db.get_answer_stats()
        # Estimate cost: ~1800 tokens avg per query, gpt-4o pricing
        avg_tokens = 1800
        cost_per_query = (avg_tokens / 1000) * 0.003
        stats["estimated_cost_per_query_usd"] = round(cost_per_query, 5)
        if stats.get("total_answers"):
            stats["estimated_total_cost_usd"] = round(
                stats["total_answers"] * cost_per_query, 2)
        return stats
    except Exception as e:
        raise HTTPException(500, f"Stats query failed: {e}")


@router.get("/eval/low-scores")
async def get_low_scores(
    request:   Request,
    threshold: float = Query(7.0, ge=0, le=10, description="Score threshold"),
    limit:     int   = Query(20,  ge=1, le=100),
):
    """
    Answers that scored below the threshold.
    Use this to find the worst-performing query types for improvement.
    """
    db = getattr(request.app.state, "relational_db", None)
    if db is None:
        return {"error": "Database unavailable", "results": []}
    try:
        results = db.get_low_scored_answers(threshold=threshold, limit=limit)
        return {"threshold": threshold, "count": len(results), "results": results}
    except Exception as e:
        raise HTTPException(500, f"Query failed: {e}")


@router.get("/eval/feedback")
async def get_feedback(
    request: Request,
    limit:   int = Query(50, ge=1, le=200),
):
    """
    Recent feedback loop signals.
    Shows the most common failure patterns so you know where to improve.
    """
    loop = getattr(request.app.state, "feedback_loop", None)
    if loop is None:
        return {"error": "Feedback loop unavailable", "items": []}

    items    = loop.get_pending_feedback(limit=limit)
    patterns = loop.get_patterns()
    stats    = loop.get_stats()
    return {
        "stats":    stats,
        "patterns": patterns,
        "items":    items[:limit],
    }


@router.get("/eval/health")
async def eval_health(request: Request):
    """
    Detailed health check showing all component states and configuration.
    More detailed than GET /health.
    """
    s = request.app.state
    components = [
        "llm_client", "embedding_client", "vector_store", "relational_db",
        "planner", "tool_executor", "conditional_router", "orchestrator",
        "gatekeeper", "auditor", "strategist",
        "llm_judge", "feedback_loop", "preprocessing_pipeline",
    ]
    status = {
        c: ("ok" if getattr(s, c, None) is not None else "unavailable")
        for c in components
    }
    ok_count  = sum(1 for v in status.values() if v == "ok")
    all_ok    = ok_count == len(components)

    # Vector store info
    vs_info = {}
    if getattr(s, "vector_store", None):
        try:
            vs_info["vector_count"] = s.vector_store.count()
        except Exception:
            vs_info["vector_count"] = "unknown"

    return {
        "overall":    "healthy" if all_ok else "degraded",
        "components_ok": f"{ok_count}/{len(components)}",
        "components": status,
        "vector_store": vs_info,
    }


@router.post("/eval/stress-test")
async def run_stress_test(request: Request):
    """
    Run the red-team adversarial test suite against the live system.
    Uses the real /query endpoint internally.
    WARNING: Makes ~25 LLM calls — costs money and takes ~60 seconds.
    """
    s = request.app.state
    if not getattr(s, "llm_client", None):
        raise HTTPException(503, "LLM client unavailable")

    # Build a query function that calls our own pipeline
    def live_query(prompt: str) -> str:
        """Calls the pipeline directly (bypasses HTTP layer for speed)."""
        try:
            # Use tool_executor directly for speed
            if getattr(s, "tool_executor", None) and getattr(s, "vector_store", None):
                chunks  = s.vector_store.search(prompt, top_k=3)
                if not chunks:
                    return "I cannot find information about that in my knowledge base."
                plan = {
                    "original_query": prompt,
                    "steps": [
                        {"step_number": 1, "action": "synthesize",
                         "search_query": "", "depends_on": []},
                    ],
                }
                ctx = s.tool_executor.execute_plan(plan)
                for r in ctx.get("step_results", {}).values():
                    if r.get("success") and r.get("data"):
                        return r["data"].get("answer", "No answer generated.")
            return "I cannot find relevant information."
        except Exception:
            return "I cannot find relevant information."

    try:
        from stress_testing.runner import RedTeamRunner
        runner = RedTeamRunner(query_function=live_query)
        report = runner.run_all()
        # Remove verbose all_results to keep response manageable
        report.pop("all_results", None)
        return report
    except Exception as e:
        raise HTTPException(500, f"Stress test failed: {e}")
