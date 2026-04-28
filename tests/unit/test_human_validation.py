# ============================================================
# tests/unit/test_human_validation.py
#
# Tests for all three human validation components:
# - Gatekeeper  (first-pass risk flag)
# - Auditor     (fact-checking against source chunks)
# - Strategist  (final approve/escalate/reject decision)
# ============================================================

import pytest
from unittest.mock import MagicMock


# ── Gatekeeper ────────────────────────────────────────────────

class TestGatekeeper:

    def setup_method(self):
        from human_validation.gatekeeper import Gatekeeper
        self.gk = Gatekeeper(
            min_confidence=0.7,
            min_answer_length=50,
            min_eval_score=7.0,
        )

    def _answer(self, text):
        return {"final_answer": text}

    def test_clean_answer_passes(self):
        """A well-formed, confident, neutral answer should pass all checks."""
        long_clean = (
            "Our refund policy allows returns within 30 days of purchase. "
            "Items must be in original condition. Contact support to initiate."
        )
        result = self.gk.check(self._answer(long_clean), confidence=0.9, eval_score=8.5)

        assert result["passed"] is True
        assert result["needs_review"] is False
        assert result["risk_level"] == "low"

    def test_short_answer_flagged(self):
        """An answer shorter than min_answer_length should be flagged."""
        result = self.gk.check(self._answer("Short."), confidence=0.9, eval_score=8.0)
        assert result["needs_review"] is True

    def test_low_confidence_flagged(self):
        """Confidence below min_confidence should trigger review."""
        long = "A" * 100
        result = self.gk.check(self._answer(long), confidence=0.3, eval_score=8.0)
        assert result["needs_review"] is True

    def test_low_eval_score_flagged(self):
        """Eval score below min_eval_score should trigger review."""
        long = "A" * 100
        result = self.gk.check(self._answer(long), confidence=0.9, eval_score=4.0)
        assert result["needs_review"] is True

    def test_uncertainty_phrase_flagged(self):
        """Answers containing 'I don't know' should be flagged."""
        text = "I don't know the answer to your question about this topic honestly."
        result = self.gk.check(self._answer(text), confidence=0.9, eval_score=8.0)
        assert result["needs_review"] is True

    def test_sensitive_topic_is_high_risk(self):
        """Legal or medical content should be flagged as high risk."""
        text = "Based on legal precedent, you should file a lawsuit against them immediately."
        text = text * 3  # Make it long enough to pass length check
        result = self.gk.check(self._answer(text), confidence=0.9, eval_score=8.0)

        assert result["risk_level"] == "high"
        assert result["needs_review"] is True

    def test_medical_content_is_high_risk(self):
        """Medical advice should be flagged as high risk."""
        text = ("The diagnosis indicates you should take medication at this dosage. "
                "Please consult a medical professional for proper treatment. " * 3)
        result = self.gk.check(self._answer(text), confidence=0.9, eval_score=8.0)
        assert result["risk_level"] == "high"

    def test_result_has_reasons_list(self):
        """Result should always include a 'reasons' list."""
        result = self.gk.check(self._answer("Test"), confidence=0.5, eval_score=5.0)
        assert "reasons" in result
        assert isinstance(result["reasons"], list)


# ── Auditor ───────────────────────────────────────────────────

class TestAuditor:

    def test_skips_audit_without_llm(self):
        """Without LLM client, returns audit_skipped=True."""
        from human_validation.auditor import Auditor

        auditor = Auditor(llm_client=None)
        result = auditor.audit("Some answer text.", [])

        assert result.get("audit_skipped") is True
        assert result["hallucination_risk"] == "unknown"

    def test_skips_on_empty_answer(self):
        """Empty answer text should return immediately with low risk."""
        from human_validation.auditor import Auditor

        auditor = Auditor(llm_client=None)
        result = auditor.audit("", [])
        assert result["hallucination_risk"] == "low"

    def test_calls_llm_with_answer_and_sources(self):
        """LLM should be called with both the answer and source chunks."""
        from human_validation.auditor import Auditor

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = """{
            "sentences": [{"text": "X is true.", "status": "SUPPORTED", "source_ref": "Source 1"}],
            "unsupported_count": 0,
            "hallucination_risk": "low",
            "overall_assessment": "All claims supported."
        }"""
        auditor = Auditor(llm_client=mock_client)

        chunks = [{"source": "doc.pdf", "text": "X is true because of Y."}]
        result = auditor.audit("X is true.", chunks)

        mock_client.chat.completions.create.assert_called_once()
        assert result["hallucination_risk"] == "low"
        assert result["unsupported_count"] == 0

    def test_falls_back_on_json_error(self):
        """If LLM returns bad JSON, falls back gracefully."""
        from human_validation.auditor import Auditor

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = \
            "Not valid JSON at all!"
        auditor = Auditor(llm_client=mock_client)

        result = auditor.audit("Some answer.", [])
        assert "error" in result or result["hallucination_risk"] == "unknown"

    def test_get_unsupported_sentences(self):
        """get_unsupported_sentences extracts only UNSUPPORTED sentences."""
        from human_validation.auditor import Auditor

        auditor = Auditor(llm_client=None)
        audit_report = {
            "sentences": [
                {"text": "Supported claim.", "status": "SUPPORTED"},
                {"text": "Made up claim.",   "status": "UNSUPPORTED"},
                {"text": "Inferred claim.",  "status": "INFERRED"},
            ]
        }
        unsupported = auditor.get_unsupported_sentences(audit_report)
        assert unsupported == ["Made up claim."]


# ── Strategist ────────────────────────────────────────────────

class TestStrategist:

    def setup_method(self):
        from human_validation.strategist import Strategist
        self.s = Strategist(min_critique_score=4.0, escalate_on_medium_risk=True)

    def _gate_ok(self):
        return {"risk_level": "low", "passed": True, "reasons": []}

    def _audit_ok(self):
        return {"hallucination_risk": "low", "unsupported_count": 0}

    def test_approves_clean_answer(self):
        """All-clear signals should result in 'approve'."""
        decision = self.s.decide(self._gate_ok(), self._audit_ok(), critique_score=8.0)
        assert decision["decision"] == "approve"

    def test_rejects_critically_low_critique_score(self):
        """Critique score below min_critique_score → 'reject'."""
        decision = self.s.decide(self._gate_ok(), self._audit_ok(), critique_score=2.0)
        assert decision["decision"] == "reject"
        assert "fallback_message" in decision

    def test_escalates_high_hallucination_risk(self):
        """High hallucination risk → 'escalate' with urgent priority."""
        audit = {"hallucination_risk": "high", "unsupported_count": 5}
        decision = self.s.decide(self._gate_ok(), audit, critique_score=8.0)
        assert decision["decision"] == "escalate"
        assert decision["priority"] == "urgent"

    def test_escalates_high_gate_risk(self):
        """High gatekeeper risk → 'escalate' with urgent priority."""
        gate = {"risk_level": "high", "passed": False,
                "reasons": ["Sensitive topic detected"]}
        decision = self.s.decide(gate, self._audit_ok(), critique_score=8.0)
        assert decision["decision"] == "escalate"
        assert decision["priority"] == "urgent"

    def test_escalates_medium_hallucination_risk(self):
        """Medium hallucination risk → 'escalate' with high priority."""
        audit = {"hallucination_risk": "medium", "unsupported_count": 1}
        decision = self.s.decide(self._gate_ok(), audit, critique_score=8.0)
        assert decision["decision"] == "escalate"

    def test_escalates_when_gate_not_passed(self):
        """Failed gatekeeper check → escalate."""
        gate = {"risk_level": "medium", "passed": False, "reasons": ["Short answer"]}
        decision = self.s.decide(gate, self._audit_ok(), critique_score=8.0)
        assert decision["decision"] == "escalate"

    def test_decision_contains_signals(self):
        """Every decision should include a 'signals' dict for debugging."""
        decision = self.s.decide(self._gate_ok(), self._audit_ok(), critique_score=8.0)
        assert "signals" in decision
        assert "critique_score" in decision["signals"]

    def test_fallback_message_is_helpful(self):
        """Rejection fallback message should be non-empty and helpful."""
        decision = self.s.decide(self._gate_ok(), self._audit_ok(), critique_score=2.0)
        msg = decision.get("fallback_message", "")
        assert len(msg) > 50
        # Should NOT reveal internal system errors
        assert "exception" not in msg.lower()
        assert "traceback" not in msg.lower()

    def test_format_escalation_note(self):
        """format_escalation_note should produce a readable note for reviewers."""
        gate = {"risk_level": "high", "passed": False, "reasons": ["Legal content"]}
        decision = self.s.decide(gate, self._audit_ok(), critique_score=8.0)
        note = self.s.format_escalation_note(
            decision,
            query="Should I sue my employer?",
            answer="Based on legal grounds you could file...",
        )
        assert "HUMAN REVIEW" in note
        assert "Should I sue" in note
        assert len(note) > 100
