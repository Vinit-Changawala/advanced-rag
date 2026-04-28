# ============================================================
# human_validation/gatekeeper.py
#
# PURPOSE: First filter — decide if an answer needs human eyes.
#
# BEGINNER CONCEPT - Why a Gatekeeper?
# Not every answer should go straight to the user.
# The Gatekeeper is like a bouncer at a club:
# - Good answers → let them through immediately
# - Risky answers → put them in a queue for human review
#
# What makes an answer "risky"?
# - Low confidence score
# - Contains words like "I'm not sure" or "might be"
# - Very short (probably didn't find a good answer)
# - Topic is sensitive (legal, medical, financial)
# ============================================================

import logging
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Phrases that suggest the model is uncertain
UNCERTAINTY_PHRASES = [
    "i'm not sure", "i am not sure", "i don't know", "i do not know",
    "i cannot confirm", "i'm unable to", "uncertain", "unclear",
    "possibly", "it might be", "it may be", "perhaps",
    "i cannot find", "no information available", "not in my knowledge",
    "you should consult", "please verify"
]

# Topics that should always go to human review
SENSITIVE_TOPICS = [
    "legal", "lawsuit", "sue", "attorney", "court",
    "medical", "diagnosis", "treatment", "medication", "dosage",
    "financial advice", "invest", "stock", "portfolio",
    "confidential", "classified", "proprietary",
]


class Gatekeeper:
    """
    Filters answers based on quality and sensitivity.

    Usage:
        gatekeeper = Gatekeeper(min_confidence=0.7)
        decision = gatekeeper.check(answer, confidence=0.65)
        if decision["needs_review"]:
            send_to_human_queue(answer)
        else:
            send_to_user(answer)
    """

    def __init__(self, min_confidence: float = 0.7,
                 min_answer_length: int = 50,
                 min_eval_score: float = 7.0):
        """
        Args:
            min_confidence: Below this → flag for review
            min_answer_length: Shorter than this → flag (probably a bad answer)
            min_eval_score: Evaluation score below this → flag
        """
        self.min_confidence = min_confidence
        self.min_answer_length = min_answer_length
        self.min_eval_score = min_eval_score

    def check(self, answer: Dict[str, Any],
              confidence: float = 1.0,
              eval_score: float = 10.0) -> Dict[str, Any]:
        """
        Check if an answer should be sent directly or queued for review.

        Returns:
            Dict with "needs_review", "reason", "risk_level"
        """
        answer_text = answer.get("final_answer", answer.get("answer_text", ""))
        reasons = []
        risk_level = "low"

        # Check 1: Confidence score
        if confidence < self.min_confidence:
            reasons.append(f"Low confidence: {confidence:.2f}")
            risk_level = "medium"

        # Check 2: Evaluation score
        if eval_score < self.min_eval_score:
            reasons.append(f"Low eval score: {eval_score:.1f}")
            risk_level = "medium"

        # Check 3: Answer too short
        if len(answer_text) < self.min_answer_length:
            reasons.append(f"Answer too short: {len(answer_text)} chars")
            risk_level = "medium"

        # Check 4: Uncertainty phrases
        answer_lower = answer_text.lower()
        found_uncertainty = [p for p in UNCERTAINTY_PHRASES if p in answer_lower]
        if found_uncertainty:
            reasons.append(f"Uncertainty detected: {found_uncertainty[:2]}")
            risk_level = "medium"

        # Check 5: Sensitive topics (always high risk)
        found_sensitive = [t for t in SENSITIVE_TOPICS if t in answer_lower]
        if found_sensitive:
            reasons.append(f"Sensitive topic: {found_sensitive[:2]}")
            risk_level = "high"

        needs_review = len(reasons) > 0

        if needs_review:
            logger.info(f"Gatekeeper: Flagged for review. Risk={risk_level}. Reasons: {reasons}")
        else:
            logger.info("Gatekeeper: Answer passed all checks")

        return {
            "needs_review": needs_review,
            "reasons": reasons,
            "risk_level": risk_level,
            "passed": not needs_review,
        }


# ============================================================
# human_validation/auditor.py
#
# PURPOSE: Verify that each claim in the answer is traceable
#          back to a source document.
#
# WHY AUDITING?
# Hallucination (AI making up facts) is RAG's #1 problem.
# The Auditor traces each sentence back to its source.
# If a sentence can't be traced → it's flagged as potential hallucination.
# ============================================================


class Auditor:
    """
    Verifies citation accuracy and source traceability.

    Usage:
        auditor = Auditor(llm_client=client)
        report = auditor.audit(answer_text, source_chunks)
        print(report["hallucination_risk"])  # "low", "medium", "high"
    """

    AUDIT_PROMPT = """You are an fact-checker. Review each sentence in the answer and
check if it is supported by the provided source documents.

Answer to audit:
{answer}

Source documents:
{sources}

For each sentence, determine:
- SUPPORTED: Clearly stated in sources
- INFERRED: Reasonably implied by sources
- UNSUPPORTED: Cannot be verified from sources (potential hallucination)

Return a JSON:
{{
  "sentences": [{{"text": "...", "status": "SUPPORTED|INFERRED|UNSUPPORTED"}}],
  "unsupported_count": <int>,
  "hallucination_risk": "low|medium|high",
  "overall_assessment": "..."
}}"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def audit(self, answer_text: str, source_chunks: list) -> Dict[str, Any]:
        """
        Audit an answer against its source chunks.

        Returns a detailed audit report.
        """
        if not self.llm_client:
            return {"hallucination_risk": "unknown", "audit_skipped": True}

        sources_text = "\n\n".join([
            f"Source {i+1}: {c.get('text', '')[:300]}"
            for i, c in enumerate(source_chunks[:5])
        ])

        prompt = self.AUDIT_PROMPT.format(
            answer=answer_text,
            sources=sources_text
        )

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            import json
            raw = response.choices[0].message.content
            return json.loads(raw)
        except Exception as e:
            logger.error(f"Audit failed: {e}")
            return {"hallucination_risk": "unknown", "error": str(e)}


# ============================================================
# human_validation/strategist.py
#
# PURPOSE: Make the final decision: approve / improve / reject.
# The Strategist has the highest authority in human validation.
# ============================================================


class Strategist:
    """
    Makes the final routing decision based on gatekeeper + auditor results.

    Three possible decisions:
    - "approve":  Answer is good, send to user
    - "escalate": Add to human review queue
    - "reject":   Answer is too risky, return a safe fallback message
    """

    SAFE_FALLBACK = (
        "I wasn't able to find a confident answer to your question "
        "in the available documents. A human expert will review your "
        "query and get back to you shortly."
    )

    def decide(self, gatekeeper_result: Dict, audit_result: Dict,
               critique_score: float = 8.0) -> Dict[str, Any]:
        """
        Make the final routing decision.

        Returns:
            Dict with "decision", "reason", and optionally "fallback_message"
        """
        risk_level = gatekeeper_result.get("risk_level", "low")
        hallucination_risk = audit_result.get("hallucination_risk", "low")
        passed_gate = gatekeeper_result.get("passed", True)

        # High risk on either check → escalate to human
        if risk_level == "high" or hallucination_risk == "high":
            return {
                "decision": "escalate",
                "reason": f"High risk detected: gate={risk_level}, hallucination={hallucination_risk}",
                "priority": "urgent",
            }

        # Very low critique score → reject
        if critique_score < 4.0:
            return {
                "decision": "reject",
                "reason": f"Critique score too low: {critique_score}",
                "fallback_message": self.SAFE_FALLBACK,
            }

        # Medium risk → queue for human review (non-urgent)
        if not passed_gate or hallucination_risk == "medium":
            return {
                "decision": "escalate",
                "reason": "Medium risk, queued for human review",
                "priority": "normal",
            }

        # All checks passed → approve
        return {
            "decision": "approve",
            "reason": "All quality checks passed",
        }
