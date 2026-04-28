# ============================================================
# tests/conftest.py
#
# PURPOSE: Shared test fixtures available to ALL test files.
#
# BEGINNER CONCEPT — What is conftest.py?
# pytest automatically loads conftest.py before running tests.
# Any @pytest.fixture defined here is available in EVERY test
# file without needing to import it — pytest injects it automatically.
#
# Think of conftest.py as the "shared preparation room" for tests.
# You set up common tools here once, and all tests can use them.
#
# FIXTURES VS REGULAR FUNCTIONS:
# A regular function:  you call it yourself → result()
# A pytest fixture:    pytest calls it and injects the result
#                      into your test function as a parameter
#
# Example:
#   def test_something(mock_openai_client):  ← pytest injects this!
#       result = mock_openai_client.chat.completions.create(...)
# ============================================================

import pytest
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any


# ── LLM CLIENT FIXTURES ──────────────────────────────────────

@pytest.fixture
def mock_openai_client():
    """
    A fake LLM client for testing without real API calls.

    Works as a mock for BOTH OpenAI and MistralAdapter since both
    expose the same interface: client.chat.completions.create(...)

    Usage in tests:
        def test_something(mock_openai_client):
            agent = SynthesisAgent(llm_client=mock_openai_client)
    """
    client = MagicMock()

    # ── Chat Completions Mock ──
    # Returns a realistic-looking response object
    chat_response = MagicMock()
    chat_response.choices[0].message.content = (
        "This is a well-structured mock answer from the AI system. "
        "It provides relevant information based on the source documents. "
        "(Source: test_document.pdf)"
    )
    chat_response.usage.total_tokens = 250
    chat_response.usage.prompt_tokens = 200
    chat_response.usage.completion_tokens = 50
    client.chat.completions.create.return_value = chat_response

    # ── Embeddings Mock ──
    # Returns a 1536-dimensional zero vector (matches text-embedding-3-small)
    embed_response = MagicMock()
    embed_response.data[0].embedding = [0.01] * 1536   # Small non-zero values
    client.embeddings.create.return_value = embed_response

    return client


@pytest.fixture
def mock_openai_client_json():
    """
    Like mock_openai_client but returns valid JSON strings.
    Used for agents/judges that parse JSON from LLM responses.
    """
    client = MagicMock()

    # Returns a valid evaluation JSON
    eval_json = '''{
        "relevance": 8.0,
        "accuracy": 7.5,
        "completeness": 8.0,
        "clarity": 9.0,
        "overall": 8.1,
        "reasoning": "The answer is relevant and well-structured."
    }'''

    response = MagicMock()
    response.choices[0].message.content = eval_json
    response.usage.total_tokens = 100
    client.chat.completions.create.return_value = response

    return client


# ── DATABASE FIXTURES ─────────────────────────────────────────

@pytest.fixture
def mock_vector_store():
    """
    A fake Qdrant vector store for testing search and storage.

    Pre-configured with realistic return values so tests can
    verify that the system correctly uses search results.
    """
    store = MagicMock()

    # Sample chunks that the store "contains"
    sample_chunks = [
        {
            "chunk_id":      "chunk_001",
            "text":          "Our refund policy allows returns within 30 days of purchase.",
            "source":        "policies/refund_policy.pdf",
            "source_type":   "document",
            "section_title": "Refund Policy Overview",
            "summary":       "30-day return window for purchases.",
            "keywords":      ["refund", "return", "30 days", "purchase"],
            "score":         0.94,
        },
        {
            "chunk_id":      "chunk_002",
            "text":          "To request a refund, contact support@company.com with your order ID.",
            "source":        "policies/refund_policy.pdf",
            "source_type":   "document",
            "section_title": "How to Request a Refund",
            "summary":       "Contact support with order ID to request refund.",
            "keywords":      ["contact", "support", "order", "email"],
            "score":         0.89,
        },
        {
            "chunk_id":      "chunk_003",
            "text":          "Digital downloads and software licenses are non-refundable.",
            "source":        "policies/refund_policy.pdf",
            "source_type":   "document",
            "section_title": "Exclusions",
            "summary":       "Software licenses cannot be refunded.",
            "keywords":      ["digital", "software", "non-refundable"],
            "score":         0.82,
        },
    ]

    store.search.return_value = sample_chunks
    store.count.return_value = 42
    store.upsert.return_value = "chunk_001"
    store.upsert_batch.return_value = None
    store.embed_text.return_value = [0.01] * 1536

    return store


@pytest.fixture
def mock_relational_db():
    """
    A fake PostgreSQL database for testing metadata storage.
    """
    db = MagicMock()

    # Return realistic stats
    db.get_answer_stats.return_value = {
        "total_answers": 150,
        "avg_score": 7.8,
        "avg_latency_ms": 2400.0,
    }

    # Return a list of low-scoring answers for feedback loop tests
    db.get_low_scored_answers.return_value = [
        {
            "answer_id": "ans_001",
            "query_text": "What is our data retention policy?",
            "answer_text": "I could not find information about data retention.",
            "overall_score": 3.5,
            "judge_reasoning": "Answer did not address the question.",
        }
    ]

    db.save_chunk_metadata.return_value = None
    db.save_answer.return_value = None
    db.save_evaluation.return_value = None
    db.save_feedback.return_value = None

    return db


# ── DOCUMENT FIXTURES ─────────────────────────────────────────

@pytest.fixture
def sample_raw_document():
    """
    A realistic raw document dict for testing the preprocessing pipeline.
    Contains headings, paragraphs, and a table.
    """
    return {
        "content": """# Employee Handbook — Remote Work Policy

## Overview
This document outlines our remote work policy for all employees.
All staff are eligible to work remotely up to 3 days per week.

## Eligibility Requirements
To work remotely, employees must meet the following criteria:
- Completed at least 6 months with the company
- Maintained a satisfactory performance review
- Have a dedicated workspace at home

## Equipment Allowance

| Item          | Allowance  | Reimbursement |
|---------------|-----------|----------------|
| Monitor       | 1 unit    | Up to $300     |
| Chair         | 1 unit    | Up to $200     |
| Internet      | Monthly   | Up to $50/mo   |

## Approval Process
Submit a remote work request through the HR portal.
Your manager will review and respond within 3 business days.
If approved, the policy is effective from the following Monday.

## Contact
For questions, contact hr@company.com or call ext. 4455.
""",
        "source":      "hr/remote_work_policy.pdf",
        "source_type": "document",
        "file_type":   ".pdf",
        "file_name":   "remote_work_policy.pdf",
        "file_size_bytes": 45678,
    }


@pytest.fixture
def sample_chunks():
    """
    Pre-processed chunks ready for storage in the vector database.
    """
    return [
        {
            "chunk_id":              "hr/remote_work_policy.pdf::chunk_0",
            "text":                  "All staff are eligible to work remotely up to 3 days per week.",
            "source":                "hr/remote_work_policy.pdf",
            "source_type":           "document",
            "section_title":         "Overview",
            "chunk_index":           0,
            "total_chunks":          4,
            "chunk_type":            "text",
            "summary":               "Remote work allowed up to 3 days per week for all staff.",
            "keywords":              ["remote", "work", "3 days", "staff"],
            "hypothetical_questions": ["How many days can I work remotely?"],
        },
        {
            "chunk_id":              "hr/remote_work_policy.pdf::chunk_1",
            "text":                  "To work remotely, employees must have completed at least 6 months with the company.",
            "source":                "hr/remote_work_policy.pdf",
            "source_type":           "document",
            "section_title":         "Eligibility Requirements",
            "chunk_index":           1,
            "total_chunks":          4,
            "chunk_type":            "text",
            "summary":               "Minimum 6 months tenure required for remote work.",
            "keywords":              ["eligibility", "6 months", "tenure"],
            "hypothetical_questions": ["How long do I need to work before going remote?"],
        },
    ]


# ── QUERY FIXTURES ────────────────────────────────────────────

@pytest.fixture
def sample_query():
    """A simple query dict for testing the reasoning engine."""
    return {
        "query_id":   "test-query-001",
        "query_text": "What is our remote work policy?",
        "session_id": None,
        "top_k":      5,
    }


@pytest.fixture
def sample_evaluation():
    """A realistic evaluation result from the LLM Judge."""
    return {
        "relevance":     8.0,
        "accuracy":      7.5,
        "completeness":  8.5,
        "clarity":       9.0,
        "overall":       8.25,
        "reasoning":     "The answer directly addresses the question with accurate information.",
    }


# ── PYTEST CONFIGURATION ──────────────────────────────────────

def pytest_configure(config):
    """
    Custom pytest configuration.
    Runs before test collection begins.
    """
    # Register custom markers so pytest doesn't warn about unknown markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with -m 'not slow')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests requiring external services"
    )
    config.addinivalue_line(
        "markers", "stress: marks adversarial stress tests"
    )


def pytest_collection_modifyitems(config, items):
    """
    Automatically add markers to tests based on their location.
    Tests in tests/integration/ get the 'integration' marker.
    Tests in tests/stress/ get the 'stress' marker.
    """
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        if "stress" in str(item.fspath):
            item.add_marker(pytest.mark.stress)
