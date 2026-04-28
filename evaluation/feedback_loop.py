# ============================================================
# evaluation/feedback_loop.py
#
# PURPOSE: Collect bad answers and feed signals back to improve the system.
#
# THE FEEDBACK LOOP CONCEPT:
# This is one of the most powerful ideas in building AI systems.
# Instead of a one-way pipeline (query → answer), we create a CYCLE:
#
#   ┌─────────────────────────────────────────────────────┐
#   │  Query → RAG Pipeline → Answer → Evaluate           │
#   │                                       │             │
#   │         ◄──── Feedback ───────────────┘             │
#   │         (low score → log what went wrong)           │
#   │         (Planner reads logs → avoids same mistakes) │
#   └─────────────────────────────────────────────────────┘
#
# WHAT GETS LOGGED?
# Every answer that scores below the threshold gets a record:
# - What was the original query?
# - What answer was generated?
# - What was the score breakdown?
# - What TYPE of problem was it? (accuracy, relevance, completeness)
# - What's the suggested fix?
#
# The Planner reads these records and adjusts future behavior.
# Over time, the system learns its own weak spots.
# ============================================================

import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class FeedbackLoop:
    """
    Records low-quality answers and generates improvement signals.

    The feedback loop bridges evaluation (what went wrong?) to
    the reasoning engine (how to avoid it next time).

    Usage:
        loop = FeedbackLoop(relational_db=db, threshold=7.0)

        # After each evaluation:
        feedback = loop.process(
            query="What is our refund policy?",
            answer="The refund policy is 30 days...",
            evaluation={"overall": 5.2, "accuracy": 4.0, "relevance": 6.0, ...},
            source_chunks=[...]
        )

        # Get pending feedback for the Planner to read:
        signals = loop.get_pending_feedback()
    """

    def __init__(self,
                 relational_db=None,
                 threshold: float = 7.0,
                 max_memory_buffer: int = 200):
        """
        Args:
            relational_db: RelationalDB instance for persistent storage
            threshold: Answers scoring below this are logged as feedback
            max_memory_buffer: Max feedback items to keep in memory
        """
        self.db = relational_db
        self.threshold = threshold
        self.max_memory_buffer = max_memory_buffer

        # In-memory ring buffer: keeps recent feedback accessible without DB
        self._buffer: List[Dict] = []

        logger.info(f"FeedbackLoop initialized with threshold={threshold}")

    def process(self,
                query: str,
                answer: str,
                evaluation: Dict[str, Any],
                source_chunks: Optional[List[Dict]] = None) -> Optional[Dict]:
        """
        Process an evaluation result and record feedback if needed.

        Args:
            query: The original user query
            answer: The generated answer text
            evaluation: Output from LLMJudge.evaluate()
            source_chunks: The chunks that were used (for context)

        Returns:
            Feedback record dict if score was below threshold, else None
        """
        overall_score = evaluation.get("overall", 10.0)

        # Only record feedback for bad answers
        if overall_score >= self.threshold:
            logger.debug(f"Answer passed threshold ({overall_score:.1f} >= {self.threshold}) — no feedback")
            return None

        # Classify the main issue type
        issue_type = self._classify_issue(evaluation)

        # Generate a concrete suggestion for the Planner
        suggested_fix = self._suggest_fix(issue_type, evaluation, source_chunks)

        feedback_record = {
            "feedback_id":       str(uuid.uuid4()),
            "timestamp":         datetime.now().isoformat(),
            "source":            "evaluation",          # Who generated this feedback
            "issue_type":        issue_type,            # e.g., "accuracy", "relevance"
            "original_query":    query,
            "original_answer":   answer[:500],          # First 500 chars
            "scores": {
                "overall":       overall_score,
                "relevance":     evaluation.get("relevance", 0),
                "accuracy":      evaluation.get("accuracy", 0),
                "completeness":  evaluation.get("completeness", 0),
                "clarity":       evaluation.get("clarity", 0),
            },
            "details":           evaluation.get("reasoning", ""),
            "suggested_fix":     suggested_fix,
            "chunks_used":       len(source_chunks) if source_chunks else 0,
        }

        # Save to persistent database
        if self.db:
            try:
                self.db.save_feedback(feedback_record)
                logger.info(f"Feedback saved to DB: score={overall_score:.1f}, issue={issue_type}")
            except Exception as e:
                logger.error(f"Failed to save feedback to DB: {e}")

        # Add to in-memory buffer (drop oldest if buffer is full)
        self._buffer.append(feedback_record)
        if len(self._buffer) > self.max_memory_buffer:
            self._buffer = self._buffer[-self.max_memory_buffer:]

        logger.info(
            f"Feedback recorded: score={overall_score:.1f}, "
            f"issue_type={issue_type}, query='{query[:60]}'"
        )

        return feedback_record

    def get_pending_feedback(self, limit: int = 50) -> List[Dict]:
        """
        Get recent feedback records for the Planner to learn from.

        Called by the Planner before creating plans to check if there
        are known failure patterns it should avoid.

        Args:
            limit: Maximum number of records to return (most recent first)

        Returns:
            List of feedback record dicts
        """
        return self._buffer[-limit:][::-1]   # Most recent first

    def get_patterns(self) -> Dict[str, int]:
        """
        Analyze the feedback buffer for patterns.

        Returns a count of each issue_type so you can see
        which problems are most common.

        Example output:
            {"accuracy": 12, "completeness": 8, "relevance": 3, "clarity": 1}
        """
        pattern_counts: Dict[str, int] = {}
        for record in self._buffer:
            issue = record.get("issue_type", "unknown")
            pattern_counts[issue] = pattern_counts.get(issue, 0) + 1

        # Sort by count descending
        return dict(sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True))

    def clear_buffer(self):
        """
        Clear the in-memory buffer after the Planner has processed it.
        Call this after the Planner reads pending feedback.
        """
        count = len(self._buffer)
        self._buffer = []
        logger.info(f"FeedbackLoop buffer cleared ({count} records removed)")

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the feedback buffer."""
        if not self._buffer:
            return {"total_feedback_items": 0}

        scores = [r["scores"]["overall"] for r in self._buffer]
        return {
            "total_feedback_items":  len(self._buffer),
            "avg_failing_score":     round(sum(scores) / len(scores), 2),
            "min_score":             round(min(scores), 2),
            "max_score":             round(max(scores), 2),
            "issue_patterns":        self.get_patterns(),
        }

    # ── PRIVATE HELPERS ──────────────────────────────────────────

    def _classify_issue(self, evaluation: Dict[str, Any]) -> str:
        """
        Find the dimension with the LOWEST score.
        That's the main problem.

        Example: if accuracy=3 and everything else is 7+,
                 the main issue is "accuracy".
        """
        dimensions = {
            "relevance":    evaluation.get("relevance", 5.0),
            "accuracy":     evaluation.get("accuracy", 5.0),
            "completeness": evaluation.get("completeness", 5.0),
            "clarity":      evaluation.get("clarity", 5.0),
        }
        # Return the key with the minimum value
        return min(dimensions, key=dimensions.get)

    def _suggest_fix(self,
                     issue_type: str,
                     evaluation: Dict[str, Any],
                     source_chunks: Optional[List[Dict]]) -> str:
        """
        Generate a concrete suggestion for fixing the issue.

        These suggestions are read by the Planner to adjust future behavior.
        They are action-oriented, not just descriptions of the problem.
        """
        fix_suggestions = {
            "relevance": (
                "RETRIEVAL ISSUE: The search returned off-topic chunks. "
                "Suggestions: (1) Rewrite the search query to be more specific. "
                "(2) Add a keyword filter to vector search. "
                "(3) Increase top_k and filter by section_title."
            ),
            "accuracy": (
                "ACCURACY ISSUE: Facts in the answer may be wrong or hallucinated. "
                "Suggestions: (1) Enable Auditor fact-checking for this query type. "
                "(2) Increase min_score threshold for retrieved chunks. "
                "(3) Ask Critique Agent to do an extra accuracy pass."
            ),
            "completeness": (
                "COMPLETENESS ISSUE: The answer is missing important information. "
                "Suggestions: (1) Increase top_k from current value. "
                "(2) Run additional searches with broader queries. "
                "(3) Check if relevant documents have been ingested."
            ),
            "clarity": (
                "CLARITY ISSUE: The answer is poorly structured or hard to understand. "
                "Suggestions: (1) Update synthesis prompt to require bullet points or headers. "
                "(2) Ask Critique Agent to simplify language. "
                "(3) Set max_tokens higher to allow more structured output."
            ),
        }

        base_suggestion = fix_suggestions.get(
            issue_type,
            "Review the full pipeline for this query type."
        )

        # Add context about available chunks if completeness is the issue
        if issue_type == "completeness" and source_chunks is not None:
            base_suggestion += f" (Note: only {len(source_chunks)} chunks were retrieved)"

        return base_suggestion
