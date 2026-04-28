# ============================================================
# evaluation/precision_recall.py
#
# PURPOSE: Measure HOW WELL our retrieval system finds the right chunks.
#
# BEGINNER CONCEPT — The two sides of retrieval quality:
#
# Imagine you're searching for 10 specific emails in your inbox.
# You run a search and get 8 results.
# - 6 of those 8 are the correct emails   ✅
# - 2 of those 8 are wrong emails         ❌
# - 4 correct emails were NOT found       ❌❌❌
#
# PRECISION  = 6/8  = 0.75  (75% of what we returned was right)
# RECALL     = 6/10 = 0.60  (we found 60% of all the right emails)
# F1         = harmonic mean of both (single balanced score)
#
# WHY BOTH MATTER:
# High precision, low recall  → Very accurate but misses lots of info
# High recall, low precision  → Finds everything but adds lots of noise
# High both (high F1)         → Perfect retrieval!
#
# In RAG, we NEED high recall — missing a key chunk means a wrong answer.
# We also need reasonable precision — too many irrelevant chunks confuse the LLM.
# ============================================================

import logging
from typing import List, Dict, Any, Set, Optional

logger = logging.getLogger(__name__)


class PrecisionRecallEvaluator:
    """
    Calculates retrieval quality metrics for the RAG system.

    Requires labeled test data: for each question, you need to know
    which chunk IDs are the "correct" ones to retrieve.
    This is called a "golden set" or "ground truth".

    Usage:
        evaluator = PrecisionRecallEvaluator()

        # You need to know the "correct" chunk IDs in advance
        metrics = evaluator.calculate(
            retrieved_ids=["chunk_3", "chunk_7", "chunk_12"],
            relevant_ids=["chunk_3", "chunk_7", "chunk_9", "chunk_15"]
        )
        print(metrics)
        # {'precision': 0.667, 'recall': 0.5, 'f1': 0.571, ...}
    """

    def calculate(self,
                  retrieved_ids: List[str],
                  relevant_ids: List[str]) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score.

        Args:
            retrieved_ids: The chunk IDs your system actually returned
            relevant_ids:  The chunk IDs that SHOULD have been returned
                          (your labeled "correct answers")

        Returns:
            Dict with precision, recall, f1, and count breakdowns
        """
        retrieved_set: Set[str] = set(retrieved_ids)
        relevant_set: Set[str] = set(relevant_ids)

        # True Positives: chunks we retrieved that ARE in the relevant set
        # The & operator on sets = intersection (items in BOTH sets)
        true_positives: Set[str] = retrieved_set & relevant_set

        # False Positives: chunks we retrieved that are NOT relevant
        false_positives: Set[str] = retrieved_set - relevant_set

        # False Negatives: relevant chunks we FAILED to retrieve
        false_negatives: Set[str] = relevant_set - retrieved_set

        # ── PRECISION = TP / (TP + FP) ──────────────────────────
        # "Of everything we returned, what fraction was correct?"
        if retrieved_set:
            precision = len(true_positives) / len(retrieved_set)
        else:
            precision = 0.0   # We returned nothing → precision undefined, treat as 0

        # ── RECALL = TP / (TP + FN) ──────────────────────────────
        # "Of all the correct answers, what fraction did we find?"
        if relevant_set:
            recall = len(true_positives) / len(relevant_set)
        else:
            recall = 0.0      # No correct answers defined → recall undefined, treat as 0

        # ── F1 = 2 * (P * R) / (P + R) ──────────────────────────
        # Harmonic mean — punishes systems that sacrifice one for the other.
        # A system with P=1.0, R=0.0 gets F1=0.0 (useless despite perfect precision).
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        metrics = {
            "precision":        round(precision, 4),
            "recall":           round(recall, 4),
            "f1":               round(f1, 4),
            "true_positives":   len(true_positives),
            "false_positives":  len(false_positives),
            "false_negatives":  len(false_negatives),
            "retrieved_count":  len(retrieved_set),
            "relevant_count":   len(relevant_set),
            # Extra detail: which specific chunks were missed/wrong
            "missed_chunks":    list(false_negatives),
            "wrong_chunks":     list(false_positives),
        }

        logger.info(
            f"Retrieval metrics: P={precision:.3f} R={recall:.3f} F1={f1:.3f} "
            f"(TP={len(true_positives)}, FP={len(false_positives)}, FN={len(false_negatives)})"
        )

        return metrics

    def calculate_at_k(self, retrieved_ids: List[str],
                       relevant_ids: List[str],
                       k: int) -> Dict[str, float]:
        """
        Calculate Precision@K and Recall@K.

        These metrics only consider the TOP K retrieved results.
        Useful because users typically only read the first few results.

        Args:
            retrieved_ids: All retrieved IDs (ordered by relevance score)
            relevant_ids: The correct IDs
            k: Only evaluate the top K results

        Example:
            P@5 asks: "Of the top 5 results, how many were correct?"
        """
        top_k_retrieved = retrieved_ids[:k]
        return {
            f"precision_at_{k}": self.calculate(top_k_retrieved, relevant_ids)["precision"],
            f"recall_at_{k}":    self.calculate(top_k_retrieved, relevant_ids)["recall"],
            f"f1_at_{k}":        self.calculate(top_k_retrieved, relevant_ids)["f1"],
        }

    def evaluate_test_suite(self,
                             test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run evaluation on a full test suite and calculate aggregate metrics.

        Args:
            test_cases: List of dicts, each with:
                        - "query": the question
                        - "retrieved_ids": what your system returned
                        - "relevant_ids": what it should have returned

        Returns:
            Aggregate metrics across all test cases
        """
        if not test_cases:
            return {"error": "No test cases provided"}

        all_metrics = []
        for tc in test_cases:
            m = self.calculate(
                retrieved_ids=tc.get("retrieved_ids", []),
                relevant_ids=tc.get("relevant_ids", [])
            )
            m["query"] = tc.get("query", "")
            all_metrics.append(m)

        # Aggregate: average each metric across all test cases
        precision_vals = [m["precision"] for m in all_metrics]
        recall_vals = [m["recall"] for m in all_metrics]
        f1_vals = [m["f1"] for m in all_metrics]

        return {
            "num_test_cases":   len(test_cases),
            "mean_precision":   round(sum(precision_vals) / len(precision_vals), 4),
            "mean_recall":      round(sum(recall_vals) / len(recall_vals), 4),
            "mean_f1":          round(sum(f1_vals) / len(f1_vals), 4),
            "per_case_metrics": all_metrics,
        }
