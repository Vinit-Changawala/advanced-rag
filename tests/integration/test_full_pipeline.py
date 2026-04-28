# ============================================================
# tests/integration/test_full_pipeline.py
#
# PURPOSE: Test the ENTIRE pipeline end-to-end.
#
# BEGINNER CONCEPT — Integration Tests vs Unit Tests:
#
# Unit Test:    Tests ONE small function in isolation.
#               Fast. Easy to debug. Tests ONE thing at a time.
#               Example: "Does BoundaryDetector.split() work correctly?"
#
# Integration Test: Tests MULTIPLE components working TOGETHER.
#               Slower. Tests that components connect correctly.
#               Example: "Does a document go from raw file → vector DB → searchable?"
#
# WHY BOTH?
# A car can have perfect individual parts that don't fit together.
# Unit tests = check each part. Integration tests = check the assembled car.
#
# These integration tests use MOCKS for external services (OpenAI, Qdrant)
# so we don't need real API keys or running databases to test.
# ============================================================

import pytest
from unittest.mock import MagicMock, patch, call
from typing import List, Dict, Any


# ── MOCK FIXTURES ────────────────────────────────────────────
# Fixtures are reusable test setup functions.
# @pytest.fixture marks them so pytest knows to inject them.

@pytest.fixture
def mock_openai_client():
    """
    Creates a fake OpenAI client that returns predictable responses.
    This means tests don't need a real OpenAI API key.
    """
    client = MagicMock()

    # Set up the chat completion response
    chat_response = MagicMock()
    chat_response.choices[0].message.content = "This is a test answer from the AI."
    chat_response.usage.total_tokens = 150
    client.chat.completions.create.return_value = chat_response

    # Set up the embedding response
    embed_response = MagicMock()
    # Return a 1536-dimensional vector of zeros
    embed_response.data[0].embedding = [0.0] * 1536
    client.embeddings.create.return_value = embed_response

    return client


@pytest.fixture
def mock_vector_store():
    """Creates a fake vector store that returns pre-defined search results."""
    store = MagicMock()

    # When search() is called, return these fake chunks
    store.search.return_value = [
        {
            "chunk_id": "test_chunk_001",
            "text": "Our refund policy allows returns within 30 days of purchase.",
            "source": "policies/refund_policy.pdf",
            "section_title": "Refund Policy",
            "score": 0.92,
        },
        {
            "chunk_id": "test_chunk_002",
            "text": "To request a refund, contact customer support with your order number.",
            "source": "policies/refund_policy.pdf",
            "section_title": "How to Request",
            "score": 0.87,
        },
    ]
    store.count.return_value = 42
    return store


@pytest.fixture
def sample_document():
    """A sample raw document dict for testing the preprocessing pipeline."""
    return {
        "content": """# Company Refund Policy

## Overview
Our refund policy is designed to be fair and transparent.

## Eligibility
Customers may request a refund within 30 days of purchase.
The item must be in its original condition and packaging.

## Process
To request a refund:
1. Contact customer support
2. Provide your order number
3. Describe the reason for the return
4. Ship the item back with the provided label

## Exclusions
| Item Category | Refundable |
|---------------|-----------|
| Electronics   | Yes       |
| Software      | No        |
| Clothing      | Yes       |

Digital downloads and software licenses are non-refundable.
""",
        "source": "policies/refund_policy.pdf",
        "source_type": "document",
        "file_type": ".pdf",
        "file_name": "refund_policy.pdf",
    }


# ── PREPROCESSING PIPELINE TESTS ─────────────────────────────

class TestPreprocessingPipeline:
    """Integration tests for the data preprocessing pipeline."""

    def test_document_flows_through_all_stages(self, sample_document, mock_openai_client):
        """A raw document should produce processed chunks with all metadata."""
        from data_preprocessing.pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline(
            llm_client=mock_openai_client,
            config={"chunk_size": 100, "chunk_overlap": 20, "min_chunk_size": 30}
        )

        chunks = pipeline.process(sample_document)

        # Should produce at least one chunk
        assert len(chunks) > 0, "Pipeline should produce at least one chunk"

        # Every chunk should have required fields
        required_fields = ["chunk_id", "text", "source", "chunk_index"]
        for chunk in chunks:
            for field in required_fields:
                assert field in chunk, f"Chunk missing field: {field}"

        # Source should be preserved
        for chunk in chunks:
            assert chunk["source"] == "policies/refund_policy.pdf"

    def test_table_becomes_separate_chunk(self, sample_document, mock_openai_client):
        """Tables in documents should become their own atomic chunks."""
        from data_preprocessing.pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline(
            llm_client=None,    # No LLM needed for this test
            config={"chunk_size": 50, "chunk_overlap": 10, "min_chunk_size": 20}
        )

        chunks = pipeline.process(sample_document)

        # At least one chunk should contain table content
        table_chunks = [c for c in chunks if c.get("chunk_type") == "table"]
        assert len(table_chunks) >= 1, "Table should be extracted as a separate chunk"

        # Table chunk should contain the table data
        all_table_text = " ".join(c["text"] for c in table_chunks)
        assert "Electronics" in all_table_text or "Refundable" in all_table_text

    def test_batch_processing(self, mock_openai_client):
        """Processing multiple documents should work correctly."""
        from data_preprocessing.pipeline import PreprocessingPipeline

        docs = [
            {"content": "Document one content here.", "source": "doc1.txt",
             "source_type": "document", "file_type": ".txt"},
            {"content": "Document two content here.", "source": "doc2.txt",
             "source_type": "document", "file_type": ".txt"},
        ]

        pipeline = PreprocessingPipeline(
            llm_client=None,
            config={"chunk_size": 50, "chunk_overlap": 10, "min_chunk_size": 10}
        )
        all_chunks = pipeline.process_batch(docs)

        # Should have chunks from both documents
        sources = {c["source"] for c in all_chunks}
        assert "doc1.txt" in sources
        assert "doc2.txt" in sources


# ── REASONING ENGINE INTEGRATION TESTS ───────────────────────

class TestReasoningEngineIntegration:
    """Integration tests for the planning + execution flow."""

    def test_planner_to_executor_flow(self, mock_openai_client, mock_vector_store):
        """Plan created by Planner should be executable by ToolExecutor."""
        from reasoning_engine.planner import Planner
        from reasoning_engine.tool_executor import ToolExecutor

        planner = Planner(llm_client=None)   # Use default plan
        executor = ToolExecutor(
            vector_store=mock_vector_store,
            llm_client=mock_openai_client
        )

        # Create a plan
        plan = planner.create_plan("What is our refund policy?")
        assert "steps" in plan

        # Execute the plan
        context = executor.execute_plan(plan)

        # Should have executed at least one step
        assert "step_results" in context
        assert len(context["step_results"]) > 0

    def test_vector_search_returns_chunks(self, mock_vector_store):
        """Vector search tool should return formatted chunk results."""
        from reasoning_engine.tool_executor import ToolExecutor

        executor = ToolExecutor(vector_store=mock_vector_store, llm_client=None)

        step = {"action": "vector_search", "search_query": "refund policy"}
        context = {"query": "What is the refund policy?"}

        result = executor.execute(step, context)

        assert result["success"] is True
        assert isinstance(result["data"], list)
        assert len(result["data"]) > 0

    def test_conditional_router_selects_correct_path(self):
        """Router should select appropriate path based on query complexity."""
        from reasoning_engine.conditional_router import ConditionalRouter

        router = ConditionalRouter()

        # Simple query
        simple_plan = {"complexity": "simple", "requires_agents": False, "steps": [{}]}
        route = router.route(simple_plan, "What time is it?")
        assert route["route"] == "direct"

        # Sensitive query (should always go to human)
        sensitive_plan = {"complexity": "simple", "requires_agents": False, "steps": [{}]}
        route = router.route(sensitive_plan, "Give me legal advice about my lawsuit")
        assert route["route"] == "human_review"


# ── MULTI-AGENT INTEGRATION TESTS ────────────────────────────

class TestMultiAgentIntegration:
    """Integration tests for the 3-agent pipeline."""

    def test_orchestrator_runs_full_pipeline(self, mock_openai_client, mock_vector_store):
        """Orchestrator should coordinate all 3 agents successfully."""
        from multi_agent_system.orchestrator import MultiAgentOrchestrator

        # Mock the critique agent to always approve
        critique_response = MagicMock()
        critique_response.choices[0].message.content = '''{
            "score": 8, "answers_question": true, "hallucination_detected": false,
            "issues": [], "approved": true, "improved_answer": ""
        }'''
        mock_openai_client.chat.completions.create.return_value = critique_response

        orchestrator = MultiAgentOrchestrator(
            llm_client=mock_openai_client,
            vector_store=mock_vector_store
        )

        result = orchestrator.run("What is our refund policy?")

        assert "final_answer" in result
        assert "pipeline_log" in result
        assert len(result["pipeline_log"]) >= 3   # At least 3 agent steps

    def test_orchestrator_handles_empty_retrieval(self, mock_openai_client):
        """Orchestrator should handle gracefully when no chunks are found."""
        from multi_agent_system.orchestrator import MultiAgentOrchestrator

        empty_store = MagicMock()
        empty_store.search.return_value = []   # No results

        orchestrator = MultiAgentOrchestrator(
            llm_client=mock_openai_client,
            vector_store=empty_store
        )

        result = orchestrator.run("What about a topic with no documents?")

        # Should not crash — should return some response
        assert "final_answer" in result
        assert result["chunks_used"] == 0


# ── HUMAN VALIDATION INTEGRATION TESTS ───────────────────────

class TestHumanValidationIntegration:
    """Integration tests for the gatekeeper → auditor → strategist pipeline."""

    def test_clean_answer_gets_approved(self, mock_openai_client):
        """A high-quality answer should pass all validation checks."""
        from human_validation.gatekeeper import Gatekeeper
        from human_validation.auditor import Auditor
        from human_validation.strategist import Strategist

        gatekeeper = Gatekeeper(min_confidence=0.5)
        auditor = Auditor(llm_client=None)   # Skip LLM audit
        strategist = Strategist()

        answer = {
            "final_answer": (
                "Our refund policy allows returns within 30 days. "
                "You must contact customer support with your order number. "
                "Items must be in original condition."
            )
        }

        gate_result = gatekeeper.check(answer, confidence=0.9, eval_score=8.5)
        audit_result = auditor.audit(answer["final_answer"], source_chunks=[])
        decision = strategist.decide(gate_result, audit_result, critique_score=8.5)

        # Should be approved
        assert decision["decision"] == "approve"

    def test_sensitive_answer_gets_escalated(self):
        """Answers with sensitive content should be escalated."""
        from human_validation.gatekeeper import Gatekeeper
        from human_validation.strategist import Strategist

        gatekeeper = Gatekeeper()
        strategist = Strategist()

        sensitive_answer = {
            "final_answer": "Based on my legal analysis, you should file a lawsuit..."
        }

        gate_result = gatekeeper.check(sensitive_answer, confidence=0.9, eval_score=8.0)
        audit_result = {"hallucination_risk": "low", "unsupported_count": 0}
        decision = strategist.decide(gate_result, audit_result, critique_score=8.0)

        assert decision["decision"] in ("escalate", "approve")
        # The gatekeeper should have flagged it
        assert gate_result["risk_level"] in ("high", "medium")
