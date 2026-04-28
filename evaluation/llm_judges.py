# ============================================================
# evaluation/llm_judges.py
#
# PURPOSE: Use a powerful LLM to score our system's answers.
#
# BEGINNER CONCEPT - LLM-as-a-Judge:
# Evaluating AI answers is hard. You can't just check if it's
# equal to a "correct" answer because there are many valid ways
# to phrase a correct answer.
#
# The solution: Use a DIFFERENT, powerful LLM as the judge.
# We give it the question, the answer, and the sources,
# and ask it to rate the answer on multiple dimensions.
#
# This is like hiring a professional proofreader.
# ============================================================

import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class LLMJudge:
    """
    Uses GPT-4o to score answers on four dimensions.

    Dimensions (each scored 0-10):
    - Relevance: Does it answer the actual question?
    - Accuracy: Are the facts correct based on sources?
    - Completeness: Is the answer thorough?
    - Clarity: Is it easy to understand?

    Usage:
        judge = LLMJudge(llm_client=openai_client)
        scores = judge.evaluate(
            query="What is our refund policy?",
            answer="You can get a refund within 30 days...",
            source_chunks=[...]
        )
        print(scores["overall"])   # e.g., 8.5
    """

    JUDGE_PROMPT = """You are an expert evaluator of AI-generated answers.
Score the following answer based on the provided question and source documents.

Score each dimension from 0-10:
- relevance: Does the answer address the question asked?
- accuracy: Are facts correct and supported by source documents?
- completeness: Is the answer thorough without being excessive?
- clarity: Is it well-written and easy to understand?

Return ONLY a JSON object:
{{
  "relevance": <0-10>,
  "accuracy": <0-10>,
  "completeness": <0-10>,
  "clarity": <0-10>,
  "overall": <average of all 4>,
  "reasoning": "<brief explanation of scores>"
}}

Question: {question}

Answer to evaluate:
{answer}

Source documents used:
{sources}

JSON scores:"""

    def __init__(self, llm_client=None, model: str = "gpt-4o"):
        self.llm_client = llm_client
        self.model = model

    def evaluate(self, query: str, answer: str,
                 source_chunks: list = None) -> Dict[str, Any]:
        """
        Score an answer.

        Returns:
            Dict with relevance, accuracy, completeness, clarity, overall scores
        """
        if not self.llm_client:
            # Can't evaluate without an LLM — return neutral score
            return self._default_scores("No LLM client configured")

        sources_text = ""
        if source_chunks:
            sources_text = "\n\n".join([
                f"[{c.get('source', 'unknown')}]: {c.get('text', '')[:300]}"
                for c in source_chunks[:4]
            ])

        prompt = self.JUDGE_PROMPT.format(
            question=query,
            answer=answer[:2000],    # Limit answer length
            sources=sources_text[:3000]
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            scores = json.loads(response.choices[0].message.content)

            # Calculate overall if not provided by model
            if "overall" not in scores:
                dimensions = ["relevance", "accuracy", "completeness", "clarity"]
                dim_scores = [scores.get(d, 5.0) for d in dimensions]
                scores["overall"] = round(sum(dim_scores) / len(dim_scores), 2)

            logger.info(f"Evaluation complete: overall={scores.get('overall', 0):.1f}")
            return scores

        except Exception as e:
            logger.error(f"LLM Judge failed: {e}")
            return self._default_scores(str(e))

    def _default_scores(self, reason: str) -> Dict[str, Any]:
        """Return neutral scores when evaluation fails."""
        return {
            "relevance": 5.0,
            "accuracy": 5.0,
            "completeness": 5.0,
            "clarity": 5.0,
            "overall": 5.0,
            "reasoning": f"Evaluation unavailable: {reason}",
            "error": True,
        }


# ============================================================
# evaluation/precision_recall.py
#
# PURPOSE: Calculate retrieval quality metrics.
#
# BEGINNER CONCEPT - What is Precision and Recall?
#
# Imagine you search for "cats" and get 10 results.
# 7 are about cats, 3 are about dogs (irrelevant).
#
# PRECISION = How many of the returned results are RELEVANT?
#             7/10 = 0.70 → 70% of results were about cats ✓
#
# RECALL = How many of the TOTAL relevant documents did we find?
#          If there are 20 cat documents in the DB and we found 7:
#          7/20 = 0.35 → We only found 35% of all cat documents ✗
#
# Good RAG needs both: high precision AND high recall.
# ============================================================

from typing import List


class PrecisionRecallEvaluator:
    """
    Calculates retrieval quality metrics.

    Usage:
        evaluator = PrecisionRecallEvaluator()
        metrics = evaluator.calculate(
            retrieved_ids=["chunk_1", "chunk_2", "chunk_5"],
            relevant_ids=["chunk_1", "chunk_3", "chunk_5", "chunk_7"]
        )
        print(metrics["precision"])  # 0.67
        print(metrics["recall"])     # 0.50
        print(metrics["f1"])         # 0.57
    """

    def calculate(self, retrieved_ids: List[str],
                  relevant_ids: List[str]) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score.

        Args:
            retrieved_ids: List of chunk IDs our system returned
            relevant_ids: List of chunk IDs that are actually relevant
                         (you need labeled test data for this)

        Returns:
            Dict with precision, recall, f1 scores
        """
        retrieved_set = set(retrieved_ids)
        relevant_set = set(relevant_ids)

        # True Positives: chunks we retrieved that ARE relevant
        true_positives = retrieved_set & relevant_set

        # Precision = TP / (TP + FP)
        # FP (False Positives) = things we retrieved that are NOT relevant
        if retrieved_set:
            precision = len(true_positives) / len(retrieved_set)
        else:
            precision = 0.0

        # Recall = TP / (TP + FN)
        # FN (False Negatives) = relevant things we MISSED
        if relevant_set:
            recall = len(true_positives) / len(relevant_set)
        else:
            recall = 0.0

        # F1 = Harmonic mean of Precision and Recall
        # (balances both — a high F1 needs both to be high)
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        return {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "true_positives": len(true_positives),
            "retrieved_count": len(retrieved_set),
            "relevant_count": len(relevant_set),
        }
