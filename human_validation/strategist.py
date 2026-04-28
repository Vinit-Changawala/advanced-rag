# ============================================================
# human_validation/strategist.py
#
# PURPOSE: Make the FINAL routing decision based on all evidence.
#
# THE STRATEGIST IS THE DECISION-MAKER.
# After Gatekeeper flags risks and Auditor checks facts,
# the Strategist reviews EVERYTHING and decides:
#
#   ✅ APPROVE  → Answer is good, send it directly to the user
#   ⏳ ESCALATE → Put it in a human review queue first
#   ❌ REJECT   → Too risky, send a safe fallback message instead
#
# ANALOGY:
# A bank's fraud detection system:
# - Gatekeeper: Flagged this transaction as suspicious
# - Auditor: Found 2 things that don't match the customer's history
# - Strategist: "Block this transaction and call the customer"
#
# The Strategist has the FINAL say — it combines signals from
# all previous stages to make the best decision.
# ============================================================

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class Strategist:
    """
    Makes the final routing decision in the human validation pipeline.

    Inputs:
    - Gatekeeper result (risk_level, passed, reasons)
    - Auditor result (hallucination_risk, unsupported_count)
    - Critique score from the agent pipeline
    - Optional override rules from configuration

    Output: One of three decisions: "approve", "escalate", "reject"

    Usage:
        strategist = Strategist()

        decision = strategist.decide(
            gatekeeper_result={"risk_level": "medium", "passed": False},
            audit_result={"hallucination_risk": "low", "unsupported_count": 0},
            critique_score=8.5
        )

        if decision["decision"] == "approve":
            send_to_user(answer)
        elif decision["decision"] == "escalate":
            add_to_human_queue(answer, priority=decision["priority"])
        else:  # "reject"
            send_to_user(decision["fallback_message"])
    """

    # Safe fallback message shown to users when an answer is rejected.
    # This is better than showing a bad/hallucinated answer.
    SAFE_FALLBACK_MESSAGE = (
        "I wasn't able to generate a reliable answer to your question "
        "with the available information. A human expert will review "
        "your query and follow up with you shortly.\n\n"
        "In the meantime, you may want to:\n"
        "• Rephrase your question with more specific details\n"
        "• Contact support directly for urgent matters\n"
        "• Check our FAQ section for common questions"
    )

    def __init__(self,
                 min_critique_score: float = 4.0,
                 escalate_on_medium_risk: bool = False):
        """
        Args:
            min_critique_score: Below this → reject the answer outright
            escalate_on_medium_risk: If True, medium-risk answers go to human
                                     review instead of being sent directly
        """
        self.min_critique_score = min_critique_score
        self.escalate_on_medium_risk = escalate_on_medium_risk

    def decide(self,
               gatekeeper_result: Dict[str, Any],
               audit_result: Dict[str, Any],
               critique_score: float = 8.0,
               extra_signals: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make the final routing decision.

        Decision logic (in priority order):
        1. Critique score critically low → REJECT immediately
        2. High hallucination risk → ESCALATE (urgent)
        3. High gatekeeper risk level → ESCALATE (urgent)
        4. Medium risk (either source) → ESCALATE (normal priority)
        5. All checks passed → APPROVE

        Args:
            gatekeeper_result: Output from Gatekeeper.check()
            audit_result: Output from Auditor.audit()
            critique_score: Numeric score from Critique Agent (1-10)
            extra_signals: Any additional signals (user feedback, etc.)

        Returns:
            Dict with:
            - "decision": "approve" | "escalate" | "reject"
            - "reason": Human-readable explanation
            - "priority": "urgent" | "high" | "normal" | None
            - "fallback_message": Only present if decision == "reject"
            - "signals": Summary of all inputs that led to this decision
        """
        # Extract signals from inputs
        risk_level = gatekeeper_result.get("risk_level", "low")
        gatekeeper_passed = gatekeeper_result.get("passed", True)
        gatekeeper_reasons = gatekeeper_result.get("reasons", [])

        hallucination_risk = audit_result.get("hallucination_risk", "unknown")
        unsupported_count = audit_result.get("unsupported_count", 0)
        audit_skipped = audit_result.get("audit_skipped", False)

        # Collect all signals into a readable summary for debugging
        signals = {
            "gatekeeper_risk_level": risk_level,
            "gatekeeper_passed": gatekeeper_passed,
            "gatekeeper_reasons": gatekeeper_reasons,
            "hallucination_risk": hallucination_risk,
            "unsupported_claims": unsupported_count,
            "critique_score": critique_score,
            "audit_skipped": audit_skipped,
        }

        logger.info(f"Strategist deciding: {signals}")

        # ── RULE 1: REJECT if critique score is critically low ──────
        # A score this low means the answer is fundamentally wrong.
        # Better to show a fallback than a bad answer.
        if critique_score < self.min_critique_score:
            return {
                "decision": "reject",
                "reason": f"Critique score critically low: {critique_score}/10 (minimum: {self.min_critique_score})",
                "priority": None,
                "fallback_message": self.SAFE_FALLBACK_MESSAGE,
                "signals": signals,
            }

        # ── RULE 2: ESCALATE (urgent) for HIGH hallucination risk ───
        if hallucination_risk == "high":
            return {
                "decision": "escalate",
                "reason": f"High hallucination risk: {unsupported_count} unsupported claims detected",
                "priority": "urgent",
                "signals": signals,
            }

        # ── RULE 3: ESCALATE (urgent) for HIGH gatekeeper risk ──────
        if risk_level == "high":
            return {
                "decision": "escalate",
                "reason": f"High-risk content detected: {gatekeeper_reasons}",
                "priority": "urgent",
                "signals": signals,
            }

        # ── RULE 4: ESCALATE (normal) for MEDIUM risk ───────────────
        # has_medium_risk = (
        #     risk_level == "medium" or
        #     hallucination_risk == "medium" or
        #     not gatekeeper_passed
        # )
        has_medium_risk = (
            risk_level == "medium" or
            hallucination_risk == "medium"
            # Removed: "not gatekeeper_passed" — gatekeeper failing alone is not
            # enough to escalate; it only matters combined with other risk signals.
            # "unknown" hallucination risk (audit skipped) is also NOT medium risk.
            )

        if has_medium_risk and self.escalate_on_medium_risk:
            # Determine priority within medium escalations
            priority = "high" if hallucination_risk == "medium" else "normal"
            return {
                "decision": "escalate",
                "reason": (
                    f"Medium risk detected — queued for human review. "
                    f"Gate: {risk_level}, Hallucination: {hallucination_risk}"
                ),
                "priority": priority,
                "signals": signals,
            }

        # ── RULE 5: APPROVE — all checks passed ─────────────────────
        return {
            "decision": "approve",
            "reason": (
                f"All quality checks passed. "
                f"Score: {critique_score}/10, "
                f"Hallucination risk: {hallucination_risk}, "
                f"Gate: passed"
            ),
            "priority": None,
            "signals": signals,
        }

    def format_escalation_note(self, decision: Dict[str, Any],
                                query: str, answer: str) -> str:
        """
        Format a note for human reviewers explaining why this was escalated.

        This note appears in the human review queue alongside the answer.
        """
        signals = decision.get("signals", {})

        note_lines = [
            "=== HUMAN REVIEW REQUIRED ===",
            f"Priority: {decision.get('priority', 'normal').upper()}",
            f"Reason: {decision.get('reason', 'Unknown')}",
            "",
            f"Original Query: {query}",
            "",
            "Risk Signals:",
        ]

        for key, value in signals.items():
            if value and value != "low":    # Only show non-trivial signals
                note_lines.append(f"  • {key}: {value}")

        note_lines += [
            "",
            "Generated Answer (for your review):",
            answer[:500] + ("..." if len(answer) > 500 else ""),
            "",
            "Please review and either approve, edit, or reject this answer.",
        ]

        return "\n".join(note_lines)
