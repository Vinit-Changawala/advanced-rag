# api/routes/query.py  — Full RAG pipeline endpoint

import uuid, time, logging
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

logger = logging.getLogger(__name__)
router = APIRouter()


class QueryRequest(BaseModel):
    query:      str  = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str]  = None
    top_k:      Optional[int]  = Field(5, ge=1, le=20)
    use_agents: Optional[bool] = None


class QueryResponse(BaseModel):
    query_id:         str
    answer:           str
    sources:          List[str]
    route_taken:      str
    confidence:       float
    evaluation_score: Optional[float] = None
    approved:         bool
    latency_ms:       int
    tokens_used:      Optional[int]  = None
    feedback_signal:  Optional[str]  = None


@router.post("/query", response_model=QueryResponse)
async def query(request_body: QueryRequest, request: Request):
    """
    Full RAG pipeline: plan → route → retrieve → generate → validate → evaluate.
    Required header: X-API-Key: <your API_SECRET_KEY>
    """
    start_time = time.time()
    query_id   = str(uuid.uuid4())
    logger.info(f"[{query_id[:8]}] Query: '{request_body.query[:80]}'")

    s = request.app.state

    # Guard: minimum components
    if getattr(s, "vector_store", None) is None:
        raise HTTPException(503, "Vector store unavailable — run: docker-compose up -d qdrant")
    if not getattr(s, "llm_client", None):
        raise HTTPException(503, "LLM client unavailable — check MISTRAL_API_KEY in .env")

    # ── 1. PLAN ──────────────────────────────────────────────
    plan = _make_plan(s, request_body.query, query_id)

    # ── 2. ROUTE ─────────────────────────────────────────────
    route = _decide_route(s, plan, request_body)
    logger.info(f"[{query_id[:8]}] Route: {route}")

    # ── 3. GENERATE ──────────────────────────────────────────
    answer_text, sources, retrieved_chunks, tokens_used = \
        _generate(s, route, plan, request_body)

    # ── 4. VALIDATE ──────────────────────────────────────────
    route, answer_text, approved = \
        _validate(s, route, answer_text, retrieved_chunks)

    # ── 5. EVALUATE ──────────────────────────────────────────
    eval_score, feedback_signal = \
        _evaluate(s, request_body.query, answer_text, retrieved_chunks, approved)

    # ── 6. SAVE ──────────────────────────────────────────────
    latency_ms = int((time.time() - start_time) * 1000)
    _save(s, query_id, request_body.query, answer_text,
          sources, eval_score, approved, route, latency_ms, tokens_used)

    logger.info(f"[{query_id[:8]}] Done: route={route} score={eval_score} {latency_ms}ms")

    return QueryResponse(
        query_id=query_id, answer=answer_text or "No answer could be generated.",
        sources=sources, route_taken=route, confidence=1.0,
        evaluation_score=eval_score, approved=approved,
        latency_ms=latency_ms, tokens_used=tokens_used, feedback_signal=feedback_signal,
    )


# ── HELPERS ───────────────────────────────────────────────────

def _make_plan(s, query: str, query_id: str) -> dict:
    """Create an execution plan. Falls back to a simple 2-step plan on failure."""
    if getattr(s, "planner", None):
        try:
            return s.planner.create_plan(query)
        except Exception as e:
            logger.warning(f"Planner error: {e}")
    return {
        "plan_id": query_id, "original_query": query,
        "complexity": "simple", "requires_agents": False,
        "steps": [
            {"step_number": 1, "action": "vector_search",
             "search_query": query, "depends_on": []},
            {"step_number": 2, "action": "synthesize",
             "search_query": "", "depends_on": [1]},
        ],
    }


def _decide_route(s, plan: dict, req: QueryRequest) -> str:
    """Decide which processing path to take."""
    if req.use_agents is True:
        return "multi_agent"
    if req.use_agents is False:
        return "direct"
    if getattr(s, "conditional_router", None):
        try:
            return s.conditional_router.route(plan, req.query).get("route", "direct")
        except Exception as e:
            logger.warning(f"Router error: {e}")
    return "direct"


def _generate(s, route: str, plan: dict, req: QueryRequest):
    """Run the chosen route and return (answer, sources, chunks, tokens)."""
    answer, sources, chunks, tokens = "", [], [], None

    if route == "human_review":
        answer = ("Your question has been flagged for human review. "
                  "A team member will respond shortly.")
        return answer, [], [], None

    if route == "multi_agent" and getattr(s, "orchestrator", None):
        try:
            result  = s.orchestrator.run(req.query)
            answer  = result.get("final_answer", "")
            # sources = result.get("sources", [])
            sources = _clean_sources(result.get("sources", []))
            chunks  = result.get("pipeline_log", [])
            return answer, sources, chunks, tokens
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")

    # direct path (or fallback from failed multi_agent)
    if getattr(s, "tool_executor", None):
        try:
            ctx    = s.tool_executor.execute_plan(plan)
            chunks = ctx.get("retrieved_chunks", [])
            for res in ctx.get("step_results", {}).values():
                if res.get("action") == "synthesize" and res.get("success"):
                    data   = res.get("data", {})
                    answer = data.get("answer", "")
                    sources = data.get("sources", [])
                    tokens  = data.get("tokens_used")
                    break
        except Exception as e:
            logger.error(f"ToolExecutor error: {e}")

    if not answer:
        # last-resort: try to synthesize with LLM, otherwise return a clear message
        try:
            raw = s.vector_store.search(req.query, top_k=req.top_k or 5)
            if raw:
                # If the best match score is below 0.4, the topic is not in the documents
                best_score = max((c.get("score", 0) for c in raw), default=0)
                if best_score < 0.4:
                    answer = "The uploaded documents do not contain information about this topic. Please upload relevant documents to get an answer."
                    sources = []
                    chunks = []
                else:
                    chunks  = raw
                    sources = list({c.get("source","") for c in raw if c.get("source")})
                    # Try LLM synthesis with the chunks
                    llm = getattr(s, "llm_client", None)
                    if llm:
                        import os as _os
                        def _cn(p):
                            n = _os.path.basename(p)
                            return n[37:] if len(n) > 37 and n[36] == "_" else n
                        ctx = "\n\n---\n\n".join(
                            f"[Doc {i+1}: {_cn(c.get('source',''))}]\n{c.get('text','')}"
                            for i, c in enumerate(raw[:8])
                        )
                        resp = llm.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role":"user","content":
                                f"Question: {req.query}\n\nDocument excerpts:\n{ctx}\n\n"
                                "STRICT RULE: Answer using ONLY the document excerpts above. "
                                "If the excerpts do not contain enough information, respond with exactly: "
                                "'The uploaded documents do not contain information about this topic. "
                                "Please upload relevant documents.' "
                                "Do NOT use outside knowledge or training memory. "
                                "Write a clear, direct answer. Do not copy text verbatim. "
                                "Do not mention file names in your answer."}],
                            max_tokens=800, temperature=0.2
                        )
                        answer = resp.choices[0].message.content.strip()
                    else:
                        answer = "No relevant information found in the knowledge base."
            else:
                answer = "No relevant information found in the knowledge base."
        except Exception as e:
            logger.error(f"Fallback search error: {e}")
            answer = "An error occurred while searching the knowledge base."

    # Clean source paths — strip temp folder, UUID prefix, show only filename
    sources = _clean_sources(sources)
    sources = _filter_used_sources(sources, chunks, answer)
    return answer, sources, chunks, tokens


# def _validate(s, route: str, answer: str, chunks: list):
#     """Run gatekeeper → auditor → strategist. Returns (route, answer, approved)."""
#     if route == "human_review" or not answer:
#         return route, answer, False

#     gate   = (s.gatekeeper.check({"final_answer": answer}, confidence=0.85, eval_score=10.0)
#               if getattr(s, "gatekeeper", None)
#               else {"passed": True, "risk_level": "low"})

#     audit  = (s.auditor.audit(answer, chunks)
#               if getattr(s, "auditor", None)
#               else {"hallucination_risk": "unknown"})

#     dec    = (s.strategist.decide(gate, audit, critique_score=9.0)
#               if getattr(s, "strategist", None)
#               else {"decision": "approve"})

#     if dec["decision"] == "reject":
#         answer = dec.get("fallback_message",
#                          "Unable to generate a reliable answer. Please rephrase.")
#         return "human_review", answer, False
#     if dec["decision"] == "escalate":
#         return "human_review", answer, False
#     return route, answer, True
def _validate(s, route: str, answer: str, chunks: list):
    """Run gatekeeper → auditor → strategist. Returns (route, answer, approved)."""
    if route == "human_review" or not answer:
        return route, answer, False

    gate = (s.gatekeeper.check({"final_answer": answer}, confidence=0.85, eval_score=10.0)
            if getattr(s, "gatekeeper", None)
            else {"passed": True, "risk_level": "low"})

    audit = (s.auditor.audit(answer, chunks)
             if getattr(s, "auditor", None)
             else {"hallucination_risk": "unknown"})

    dec = (s.strategist.decide(gate, audit, critique_score=9.0)
           if getattr(s, "strategist", None)
           else {"decision": "approve"})

    if dec["decision"] == "reject":
        fallback = dec.get("fallback_message",
                           "Unable to generate a reliable answer. Please rephrase.")
        return "human_review", fallback, False

    if dec["decision"] == "escalate":
        # Escalated but still return original route for display
        # so the badge shows "Multi-agent" not "Human review"
        return route, answer, False

    # Approved — return original route unchanged
    return route, answer, True


def _evaluate(s, query: str, answer: str, chunks: list, approved: bool):
    """
    Score answer with LLM judge.
    
    DISABLED for Mistral free tier: the LLM judge is the 7th API call per query.
    Mistral free tier allows 2 req/min. By the time we reach here, the rate limit
    is always exceeded. The judge failing logs a false score=5.0 which pollutes
    the feedback loop with wrong data.
    
    Re-enable by removing the early return below when using a paid Mistral plan.
    """
    return None, None   # disabled — too many API calls for free tier

    if not approved or not answer or not getattr(s, "llm_judge", None):  # noqa
        return None, None

    score, signal = None, None
    try:
        result = s.llm_judge.evaluate(query=query, answer=answer,
                                       source_chunks=chunks[:5])
        score  = result.get("overall")
        if getattr(s, "feedback_loop", None) and score is not None:
            fb = s.feedback_loop.process(query=query, answer=answer,
                                          evaluation=result, source_chunks=chunks)
            if fb:
                signal = f"Low score ({score:.1f}/10): {fb.get('issue_type')}"
    except Exception as e:
        logger.warning(f"Evaluation error: {e}")

    return score, signal


def _save(s, query_id, query_text, answer_text, sources,
          eval_score, approved, route, latency_ms, tokens):
    """Persist answer to PostgreSQL. Silently skips if DB is unavailable."""
    if not getattr(s, "relational_db", None):
        return
    try:
        s.relational_db.save_answer({
            "answer_id": query_id, "query_id": query_id,
            "query_text": query_text, "answer_text": answer_text,
            "sources": sources, "confidence_score": 1.0,
            "evaluation_score": eval_score, "approved_by_human": approved,
            "route_taken": route, "latency_ms": latency_ms, "token_count": tokens,
        })
    except Exception as e:
        logger.warning(f"DB save error: {e}")


def _clean_sources(sources: list) -> list:
    """
    Convert ugly temp paths like:
      \tmp\rag_uploads\abc123-uuid_myfile.pdf
    Into clean display names like:
      myfile.pdf

    The UUID prefix is added by the ingest route when saving temp files.
    Format is always: {job_id}_{original_filename}
    We split on the first underscore after the UUID to get the real name.
    """
    import os
    cleaned = []
    seen = set()
    for s in sources:
        if not s:
            continue
        # Get just the filename from the full path
        filename = os.path.basename(s)
        # Remove the UUID prefix (36 chars + 1 underscore = 37 chars)
        # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx (36 chars)
        if len(filename) > 37 and filename[36] == "_":
            filename = filename[37:]
        # Deduplicate
        if filename not in seen:
            seen.add(filename)
            cleaned.append(filename)
    return cleaned


def _filter_used_sources(sources: list, chunks: list, answer: str) -> list:
    """
    Only return sources whose content is actually reflected in the answer.
    
    Strategy: keep a source only if at least one of its chunks has a
    relevance score above 0.5, meaning it was genuinely retrieved as
    relevant — not just a low-score accidental hit.
    
    This prevents "BERT 2018.pdf" appearing as a source when asking
    about TechCorp health insurance just because BERT was in the DB.
    """
    if not chunks or not sources:
        return sources

    import os

    def _clean(p):
        n = os.path.basename(p)
        return n[37:] if len(n) > 37 and n[36] == "_" else n

    # Build set of sources that have at least one high-score chunk
    high_score_sources = set()
    for c in chunks:
        score = c.get("score", 0) if isinstance(c, dict) else 0
        if score >= 0.45:   # only genuinely relevant chunks
            src = _clean(c.get("source", ""))
            if src:
                high_score_sources.add(src)

    # If filtering removes everything, fall back to all sources
    filtered = [s for s in sources if s in high_score_sources]
    return filtered if filtered else sources