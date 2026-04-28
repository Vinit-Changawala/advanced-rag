# ============================================================
# human_validation/auditor.py
#
# PURPOSE: Trace every claim in an answer back to its source.
#
# THE HALLUCINATION PROBLEM:
# Language models can "hallucinate" — confidently state things
# that are completely made up, using fluent, convincing language.
# Example hallucination: "According to the Q3 report, revenue was
# $5.2 billion" — but the report never actually says that.
#
# THE AUDITOR'S JOB:
# Read every sentence of the answer.
# For each sentence, check: "Is this actually in the source chunks?"
# If NO → flag it as UNSUPPORTED (potential hallucination).
#
# AUDIT RESULT CATEGORIES:
# ✅ SUPPORTED    — Directly stated in a source chunk
# ⚠️  INFERRED    — Reasonably implied by sources (acceptable)
# ❌ UNSUPPORTED  — Cannot be verified (potential hallucination!)
# ============================================================

import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class Auditor:
    """
    Verifies that each claim in an answer is traceable to source documents.

    The Auditor is the second layer of human_validation (after the Gatekeeper).
    While the Gatekeeper looks at surface signals (length, keywords),
    the Auditor does DEEP fact-checking using the source chunks.

    Usage:
        auditor = Auditor(llm_client=openai_client)

        report = auditor.audit(
            answer_text="The revenue was $5 billion in Q3.",
            source_chunks=[{"text": "Q3 revenue reached $5 billion..."}]
        )

        print(report["hallucination_risk"])     # "low", "medium", "high"
        print(report["unsupported_count"])      # Number of unverifiable claims
        print(report["sentences"])             # Per-sentence breakdown
    """

    AUDIT_PROMPT = """You are a meticulous fact-checker.

Review each sentence in the ANSWER and classify it against the SOURCE DOCUMENTS.

Classification rules:
- SUPPORTED: The sentence's claim is explicitly stated in one of the sources
- INFERRED: The claim is a reasonable conclusion from what sources say
- UNSUPPORTED: The claim cannot be verified from the sources (potential hallucination)

Return ONLY a JSON object:
{{
  "sentences": [
    {{"text": "sentence here", "status": "SUPPORTED|INFERRED|UNSUPPORTED", "source_ref": "Source 1 or N/A"}}
  ],
  "unsupported_count": <integer>,
  "hallucination_risk": "low|medium|high",
  "overall_assessment": "brief summary of findings"
}}

Hallucination risk guide:
- "low"    = 0 unsupported sentences
- "medium" = 1-2 unsupported sentences  
- "high"   = 3+ unsupported sentences

ANSWER to audit:
{answer}

SOURCE DOCUMENTS:
{sources}

JSON audit result:"""

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: OpenAI client for LLM-powered fact-checking.
                       If None, returns a basic "unknown" assessment.
        """
        self.llm_client = llm_client

    def audit(self, answer_text: str,
              source_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Audit an answer against its source chunks.

        Args:
            answer_text: The synthesized answer to fact-check
            source_chunks: The chunks that were used to generate the answer

        Returns:
            Audit report dict with hallucination_risk, sentences breakdown, etc.
        """
        if not answer_text.strip():
            return {"hallucination_risk": "low", "unsupported_count": 0,
                    "sentences": [], "overall_assessment": "Empty answer"}

        # If no LLM is configured, skip detailed audit
        if not self.llm_client:
            logger.warning("Auditor: No LLM client — skipping detailed audit")
            return {
                "hallucination_risk": "unknown",
                "unsupported_count": 0,
                "sentences": [],
                "overall_assessment": "Audit skipped (no LLM configured)",
                "audit_skipped": True,
            }

        # Build the sources text block
        source_text_parts = []
        for i, chunk in enumerate(source_chunks[:6], 1):
            source = chunk.get("source", "unknown")
            text = chunk.get("text", "")[:400]    # Limit each chunk to 400 chars
            source_text_parts.append(f"[Source {i}: {source}]\n{text}")

        sources_block = "\n\n".join(source_text_parts) if source_text_parts else "No sources provided."

        prompt = self.AUDIT_PROMPT.format(
            answer=answer_text[:2000],     # Limit answer to 2000 chars
            sources=sources_block
        )

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",        # Use cheaper model for auditing
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.0,            # Zero temperature: deterministic fact-checking
                response_format={"type": "json_object"}
            )

            raw = response.choices[0].message.content
            report = json.loads(raw)

            # Ensure required keys exist
            report.setdefault("hallucination_risk", "unknown")
            report.setdefault("unsupported_count", 0)
            report.setdefault("sentences", [])
            report.setdefault("overall_assessment", "")

            logger.info(
                f"Auditor: hallucination_risk={report['hallucination_risk']}, "
                f"unsupported={report['unsupported_count']}"
            )
            return report

        except json.JSONDecodeError as e:
            logger.error(f"Auditor: JSON parse failed: {e}")
            return {
                "hallucination_risk": "unknown",
                "unsupported_count": 0,
                "sentences": [],
                "overall_assessment": f"Audit parsing error: {e}",
                "error": str(e),
            }
        except Exception as e:
            logger.error(f"Auditor: LLM call failed: {e}")
            return {
                "hallucination_risk": "unknown",
                "error": str(e),
                "audit_skipped": True,
            }

    def get_unsupported_sentences(self, audit_report: Dict[str, Any]) -> List[str]:
        """
        Extract just the unsupported sentences from an audit report.

        Useful for the Strategist to understand WHAT is problematic.
        """
        sentences = audit_report.get("sentences", [])
        return [
            s["text"] for s in sentences
            if s.get("status") == "UNSUPPORTED"
        ]
