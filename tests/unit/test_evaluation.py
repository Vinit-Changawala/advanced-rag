# ============================================================
# tests/unit/test_evaluation.py
#
# Tests for:
# - LLMJudge         (scoring answers 0-10 across 4 dimensions)
# - PrecisionRecallEvaluator (retrieval quality metrics)
# - LatencyCostTracker (timing and cost recording)
# - FeedbackLoop      (collecting low-score signals)
# ============================================================

import time
import pytest
from unittest.mock import MagicMock


# ── LLMJudge ─────────────────────────────────────────────────

class TestLLMJudge:

    def test_returns_default_scores_without_llm(self):
        """Without LLM, returns neutral 5.0 scores for all dimensions."""
        from evaluation.llm_judges import LLMJudge

        judge = LLMJudge(llm_client=None)
        result = judge.evaluate("What is X?", "X is a thing.", [])

        for dim in ["relevance", "accuracy", "completeness", "clarity", "overall"]:
            assert dim in result
            assert isinstance(result[dim], float)
        assert result.get("error") is True

    def test_overall_is_average_of_four_dimensions(self):
        """overall score should be the average of the 4 dimension scores."""
        from evaluation.llm_judges import LLMJudge

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = """{
            "relevance": 8.0,
            "accuracy": 6.0,
            "completeness": 7.0,
            "clarity": 9.0,
            "overall": 7.5,
            "reasoning": "Good answer."
        }"""
        judge = LLMJudge(llm_client=mock_client)
        result = judge.evaluate("Question?", "Answer.", [])

        assert result["overall"] == 7.5
        assert result["relevance"] == 8.0

    def test_calculates_overall_if_not_returned(self):
        """If LLM omits 'overall', it should be calculated from dimensions."""
        from evaluation.llm_judges import LLMJudge

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = """{
            "relevance": 8.0,
            "accuracy": 8.0,
            "completeness": 8.0,
            "clarity": 8.0,
            "reasoning": "Nice."
        }"""
        judge = LLMJudge(llm_client=mock_client)
        result = judge.evaluate("Q?", "A.", [])

        assert "overall" in result
        assert result["overall"] == 8.0

    def test_falls_back_on_json_error(self):
        """If LLM returns malformed JSON, falls back to default scores."""
        from evaluation.llm_judges import LLMJudge

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = \
            "This is not JSON at all."
        judge = LLMJudge(llm_client=mock_client)
        result = judge.evaluate("Q?", "A.", [])

        assert result.get("error") is True
        assert result["overall"] == 5.0

    def test_evaluate_with_source_chunks(self):
        """Evaluation should work when source_chunks are provided."""
        from evaluation.llm_judges import LLMJudge

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = """{
            "relevance": 9.0, "accuracy": 9.0,
            "completeness": 9.0, "clarity": 9.0,
            "overall": 9.0, "reasoning": "Excellent."
        }"""
        judge = LLMJudge(llm_client=mock_client)

        chunks = [
            {"source": "doc.pdf", "text": "Relevant content here."},
            {"source": "doc.pdf", "text": "More content here."},
        ]
        result = judge.evaluate("Question?", "Answer.", chunks)
        assert result["overall"] == 9.0

        # Check that source chunks were included in the prompt
        call_args = mock_client.chat.completions.create.call_args
        prompt_content = call_args[1]["messages"][0]["content"]
        assert "Relevant content" in prompt_content


# ── PrecisionRecallEvaluator ──────────────────────────────────

class TestPrecisionRecallEvaluator:

    def setup_method(self):
        from evaluation.precision_recall import PrecisionRecallEvaluator
        self.evaluator = PrecisionRecallEvaluator()

    def test_perfect_retrieval(self):
        """Retrieving exactly the right chunks → P=1.0, R=1.0, F1=1.0."""
        metrics = self.evaluator.calculate(
            retrieved_ids=["a", "b", "c"],
            relevant_ids=["a", "b", "c"],
        )
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_zero_precision_when_nothing_correct(self):
        """Retrieved only wrong chunks → P=0.0."""
        metrics = self.evaluator.calculate(
            retrieved_ids=["x", "y", "z"],
            relevant_ids=["a", "b", "c"],
        )
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0

    def test_partial_retrieval(self):
        """Retrieved 2 out of 4 relevant → R=0.5."""
        metrics = self.evaluator.calculate(
            retrieved_ids=["a", "b", "x"],   # a,b correct; x wrong
            relevant_ids=["a", "b", "c", "d"],
        )
        # Precision: 2/3 = 0.667
        assert abs(metrics["precision"] - 0.667) < 0.01
        # Recall: 2/4 = 0.5
        assert metrics["recall"] == 0.5
        assert metrics["true_positives"] == 2
        assert metrics["false_positives"] == 1
        assert metrics["false_negatives"] == 2

    def test_empty_retrieved_list(self):
        """Empty retrieved list → P=0, R=0, F1=0."""
        metrics = self.evaluator.calculate(
            retrieved_ids=[],
            relevant_ids=["a", "b"],
        )
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0

    def test_empty_relevant_list(self):
        """Empty relevant list → P=0, R=0, F1=0."""
        metrics = self.evaluator.calculate(
            retrieved_ids=["a", "b"],
            relevant_ids=[],
        )
        assert metrics["precision"] == 0.0

    def test_f1_is_harmonic_mean(self):
        """F1 should be harmonic mean of P and R, not arithmetic mean."""
        # P=1.0, R=0.5 → arithmetic mean=0.75, harmonic mean=0.667
        metrics = self.evaluator.calculate(
            retrieved_ids=["a"],
            relevant_ids=["a", "b"],
        )
        assert abs(metrics["f1"] - 0.667) < 0.01

    def test_calculate_at_k(self):
        """calculate_at_k evaluates only the top K retrieved results."""
        metrics = self.evaluator.calculate_at_k(
            retrieved_ids=["a", "b", "x", "y", "z"],
            relevant_ids=["a", "b", "c"],
            k=2,
        )
        # Top 2: a, b → both correct → P@2 = 1.0
        assert metrics["precision_at_2"] == 1.0

    def test_evaluate_test_suite(self):
        """evaluate_test_suite aggregates metrics across multiple cases."""
        test_cases = [
            {"query": "Q1", "retrieved_ids": ["a", "b"], "relevant_ids": ["a", "b"]},
            {"query": "Q2", "retrieved_ids": ["x"],      "relevant_ids": ["a", "b"]},
        ]
        agg = self.evaluator.evaluate_test_suite(test_cases)

        assert agg["num_test_cases"] == 2
        assert 0.0 <= agg["mean_precision"] <= 1.0
        assert len(agg["per_case_metrics"]) == 2

    def test_returns_missed_and_wrong_chunks(self):
        """Result should include lists of missed and wrongly-retrieved chunks."""
        metrics = self.evaluator.calculate(
            retrieved_ids=["a", "wrong1"],
            relevant_ids=["a", "missed1"],
        )
        assert "missed1" in metrics["missed_chunks"]
        assert "wrong1"  in metrics["wrong_chunks"]


# ── LatencyCostTracker ────────────────────────────────────────

class TestLatencyCostTracker:

    def test_measures_latency(self):
        """Latency should be recorded and accessible after the context block."""
        from evaluation.latency_cost import LatencyCostTracker

        tracker = LatencyCostTracker()
        with tracker.measure("q1") as ctx:
            time.sleep(0.05)  # 50ms

        report = tracker.get_report("q1")
        assert report["latency_ms"] >= 40   # at least 40ms
        assert report["latency_ms"] < 500   # but not suspiciously long

    def test_records_token_usage(self):
        """Token usage added via ctx.add_tokens() should appear in the report."""
        from evaluation.latency_cost import LatencyCostTracker

        tracker = LatencyCostTracker()
        with tracker.measure("q2") as ctx:
            ctx.add_tokens(model="gpt-4o", input_tokens=500, output_tokens=200)

        report = tracker.get_report("q2")
        assert report["total_input_tokens"] == 500
        assert report["total_output_tokens"] == 200
        assert report["total_tokens"] == 700

    def test_calculates_cost(self):
        """Cost in USD should be calculated from token usage."""
        from evaluation.latency_cost import LatencyCostTracker

        tracker = LatencyCostTracker()
        with tracker.measure("q3") as ctx:
            ctx.add_tokens(model="gpt-4o-mini", input_tokens=1000, output_tokens=200)

        report = tracker.get_report("q3")
        assert report["cost_usd"] > 0.0
        assert report["cost_usd"] < 1.0   # Sanity check — shouldn't be expensive

    def test_returns_error_for_unknown_query_id(self):
        """Getting a report for an unmeasured query_id returns error dict."""
        from evaluation.latency_cost import LatencyCostTracker

        tracker = LatencyCostTracker()
        report = tracker.get_report("nonexistent-id")
        assert "error" in report

    def test_summary_stats_across_multiple_queries(self):
        """get_summary_stats aggregates across all tracked queries."""
        from evaluation.latency_cost import LatencyCostTracker

        tracker = LatencyCostTracker()
        for qid in ["qa", "qb", "qc"]:
            with tracker.measure(qid) as ctx:
                ctx.add_tokens("gpt-4o-mini", input_tokens=100, output_tokens=50)

        stats = tracker.get_summary_stats()
        assert stats["total_queries"] == 3
        assert stats["total_cost_usd"] > 0.0
        assert "avg_latency_ms" in stats


# ── FeedbackLoop ──────────────────────────────────────────────

class TestFeedbackLoop:

    def test_no_feedback_for_good_scores(self):
        """Answers scoring above threshold should NOT generate feedback."""
        from evaluation.feedback_loop import FeedbackLoop

        loop = FeedbackLoop(relational_db=None, threshold=7.0)
        result = loop.process(
            query="What is X?",
            answer="X is defined as...",
            evaluation={"overall": 8.5, "relevance": 8.0, "accuracy": 9.0,
                        "completeness": 8.5, "clarity": 8.5},
        )
        assert result is None

    def test_feedback_recorded_for_low_scores(self):
        """Answers below threshold should generate a feedback record."""
        from evaluation.feedback_loop import FeedbackLoop

        loop = FeedbackLoop(relational_db=None, threshold=7.0)
        result = loop.process(
            query="What is Y?",
            answer="Y is...",
            evaluation={"overall": 4.0, "relevance": 3.0, "accuracy": 5.0,
                        "completeness": 4.0, "clarity": 4.0, "reasoning": "Poor."},
        )
        assert result is not None
        assert "feedback_id" in result
        assert result["source"] == "evaluation"
        assert result["original_query"] == "What is Y?"

    def test_classifies_worst_dimension_as_issue_type(self):
        """issue_type should be the dimension with the lowest score."""
        from evaluation.feedback_loop import FeedbackLoop

        loop = FeedbackLoop(relational_db=None, threshold=7.0)
        result = loop.process(
            query="Q?", answer="A.",
            evaluation={"overall": 5.0, "relevance": 8.0, "accuracy": 2.0,
                        "completeness": 7.0, "clarity": 7.0},
        )
        # accuracy=2.0 is lowest → issue_type should be "accuracy"
        assert result["issue_type"] == "accuracy"

    def test_get_pending_feedback_returns_recent_items(self):
        """get_pending_feedback should return the most recent feedback."""
        from evaluation.feedback_loop import FeedbackLoop

        loop = FeedbackLoop(relational_db=None, threshold=7.0)
        bad_eval = {"overall": 3.0, "relevance": 3.0, "accuracy": 3.0,
                    "completeness": 3.0, "clarity": 3.0}
        for i in range(5):
            loop.process(f"Query {i}", "bad answer", bad_eval)

        pending = loop.get_pending_feedback(limit=3)
        assert len(pending) == 3

    def test_get_patterns_counts_issue_types(self):
        """get_patterns should count occurrences of each issue type."""
        from evaluation.feedback_loop import FeedbackLoop

        loop = FeedbackLoop(relational_db=None, threshold=7.0)
        # 3 accuracy failures
        for _ in range(3):
            loop.process("Q?", "A.", {"overall": 4.0, "relevance": 7.0, "accuracy": 2.0,
                                       "completeness": 7.0, "clarity": 7.0})
        # 1 clarity failure
        loop.process("Q?", "A.", {"overall": 4.0, "relevance": 7.0, "accuracy": 7.0,
                                   "completeness": 7.0, "clarity": 2.0})

        patterns = loop.get_patterns()
        assert patterns.get("accuracy", 0) == 3
        assert patterns.get("clarity",  0) == 1

    def test_suggested_fix_is_string(self):
        """Every feedback record should have a non-empty suggested_fix."""
        from evaluation.feedback_loop import FeedbackLoop

        loop = FeedbackLoop(relational_db=None, threshold=7.0)
        result = loop.process(
            "Q?", "A.",
            {"overall": 3.0, "relevance": 3.0, "accuracy": 3.0,
             "completeness": 3.0, "clarity": 3.0},
        )
        assert isinstance(result["suggested_fix"], str)
        assert len(result["suggested_fix"]) > 10

    def test_buffer_cleared_after_clear_pending(self):
        """clear_buffer should remove all items from in-memory buffer."""
        from evaluation.feedback_loop import FeedbackLoop

        loop = FeedbackLoop(relational_db=None, threshold=7.0)
        bad = {"overall": 3.0, "relevance": 3.0, "accuracy": 3.0,
               "completeness": 3.0, "clarity": 3.0}
        loop.process("Q?", "A.", bad)
        loop.process("Q2?", "A2.", bad)

        assert len(loop.get_pending_feedback()) == 2
        loop.clear_buffer()
        assert len(loop.get_pending_feedback()) == 0

    def test_saves_to_db_when_provided(self):
        """Feedback should be saved to relational DB when one is configured."""
        from evaluation.feedback_loop import FeedbackLoop

        mock_db = MagicMock()
        loop = FeedbackLoop(relational_db=mock_db, threshold=7.0)

        loop.process("Q?", "A.",
                     {"overall": 3.0, "relevance": 3.0, "accuracy": 3.0,
                      "completeness": 3.0, "clarity": 3.0})

        mock_db.save_feedback.assert_called_once()
