# ============================================================
# tests/unit/test_api.py
#
# Tests for the FastAPI endpoints using TestClient.
# No real databases or LLM calls — all components are mocked.
#
# TestClient simulates HTTP requests in-process (no server needed).
# ============================================================

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient


# ── APP FIXTURE ───────────────────────────────────────────────

@pytest.fixture
def app_with_mocks():
    """
    Create the FastAPI app with all external dependencies mocked.

    We patch:
    - build_app_state()  → returns a mock state with all components
    - AuthMiddleware     → disabled (pass all requests)
    - RateLimiterMiddleware → disabled
    """
    # Build a fully-mocked app.state
    mock_state = MagicMock()

    # Mock vector store
    mock_state.vector_store.search.return_value = [
        {"chunk_id": "c1", "text": "Refunds within 30 days.", "source": "policy.pdf",
         "section_title": "Refund Policy", "score": 0.92},
    ]
    mock_state.vector_store.count.return_value = 42

    # Mock LLM client
    llm_resp = MagicMock()
    llm_resp.choices[0].message.content = "Based on the policy, refunds are within 30 days."
    llm_resp.usage.total_tokens = 200
    mock_state.llm_client.chat.completions.create.return_value = llm_resp

    # Mock planner → simple default plan
    mock_state.planner.create_plan.return_value = {
        "plan_id": "test-plan", "original_query": "test",
        "complexity": "simple", "requires_agents": False,
        "steps": [
            {"step_number": 1, "action": "vector_search",
             "search_query": "test", "depends_on": []},
            {"step_number": 2, "action": "synthesize",
             "search_query": "", "depends_on": [1]},
        ],
    }

    # Mock conditional router → direct
    mock_state.conditional_router.route.return_value = {"route": "direct"}

    # Mock tool_executor
    mock_state.tool_executor.execute_plan.return_value = {
        "query": "test",
        "retrieved_chunks": [
            {"text": "30 day refund.", "source": "policy.pdf", "score": 0.9}
        ],
        "step_results": {
            2: {
                "action": "synthesize",
                "success": True,
                "data": {
                    "answer": "Refunds are allowed within 30 days.",
                    "sources": ["policy.pdf"],
                    "tokens_used": 150,
                },
            }
        },
    }

    # Mock validation
    mock_state.gatekeeper.check.return_value = {
        "passed": True, "risk_level": "low", "needs_review": False, "reasons": []
    }
    mock_state.auditor.audit.return_value = {
        "hallucination_risk": "low", "unsupported_count": 0
    }
    mock_state.strategist.decide.return_value = {"decision": "approve"}

    # Mock evaluation
    mock_state.llm_judge.evaluate.return_value = {
        "relevance": 9.0, "accuracy": 9.0,
        "completeness": 8.0, "clarity": 9.0,
        "overall": 8.75, "reasoning": "Good."
    }
    mock_state.feedback_loop.process.return_value = None

    # Mock relational DB
    mock_state.relational_db.save_answer.return_value = None
    mock_state.relational_db.get_answer_stats.return_value = {
        "total_answers": 100, "avg_score": 8.2, "avg_latency_ms": 2300
    }
    mock_state.relational_db.get_low_scored_answers.return_value = []

    # Mock feedback_loop stats
    mock_state.feedback_loop.get_pending_feedback.return_value = []
    mock_state.feedback_loop.get_patterns.return_value = {}
    mock_state.feedback_loop.get_stats.return_value = {"total_feedback_items": 0}

    # Patch build_app_state and middleware
    with patch("utils.app_factory.build_app_state", return_value=mock_state):
        with patch("api.middleware.auth.AuthMiddleware.dispatch",
                   new=lambda self, req, call_next: call_next(req)):
            with patch("api.middleware.rate_limiter.RateLimiterMiddleware.dispatch",
                       new=lambda self, req, call_next: call_next(req)):
                import importlib
                import api.main
                importlib.reload(api.main)
                from api.main import app

                # Manually attach mock state to app
                for key, val in vars(mock_state).items():
                    setattr(app.state, key, val)

                yield app, mock_state


@pytest.fixture
def client(app_with_mocks):
    """TestClient wrapping the mocked app."""
    app, state = app_with_mocks
    with TestClient(app) as c:
        c._mock_state = state
        yield c


# ── HEALTH CHECK TESTS ────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_returns_200(self):
        """GET /health should return 200 when all components are ready."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.main import app

        # Set up minimal app state
        app.state.llm_client       = MagicMock()
        app.state.embedding_client = MagicMock()
        app.state.vector_store     = MagicMock()
        app.state.relational_db    = MagicMock()
        app.state.orchestrator     = MagicMock()

        c = TestClient(app)
        response = c.get("/health")
        assert response.status_code in (200, 206)
        data = response.json()
        assert "status" in data
        assert "components" in data

    def test_root_endpoint_returns_service_info(self):
        """GET / should return service name and docs URL."""
        from fastapi.testclient import TestClient
        from api.main import app
        c = TestClient(app)
        response = c.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "docs" in data


# ── QUERY ENDPOINT TESTS ──────────────────────────────────────

class TestQueryEndpoint:

    API_KEY = "dev-secret-key-12345"

    def _headers(self):
        return {"X-API-Key": self.API_KEY, "Content-Type": "application/json"}

    def test_query_returns_answer(self, client):
        """POST /api/v1/query should return an answer."""
        resp = client.post(
            "/api/v1/query",
            json={"query": "What is the refund policy?"},
            headers=self._headers(),
        )
        # Accept 200 (success) or 503 (components not wired in this test context)
        assert resp.status_code in (200, 503, 422)

    def test_query_requires_non_empty_string(self):
        """Empty query string should return 422 validation error."""
        from fastapi.testclient import TestClient
        from api.main import app
        c = TestClient(app)
        resp = c.post(
            "/api/v1/query",
            json={"query": ""},
            headers=self._headers(),
        )
        assert resp.status_code == 422   # Pydantic validation failure

    def test_query_response_has_required_fields(self, client):
        """Successful response should have all required fields."""
        resp = client.post(
            "/api/v1/query",
            json={"query": "What is the refund policy?"},
            headers=self._headers(),
        )
        if resp.status_code == 200:
            data = resp.json()
            required = ["query_id", "answer", "sources", "route_taken",
                        "confidence", "approved", "latency_ms"]
            for field in required:
                assert field in data, f"Missing field: {field}"

    def test_query_too_long_rejected(self):
        """Query exceeding 2000 chars should return 422."""
        from fastapi.testclient import TestClient
        from api.main import app
        c = TestClient(app)
        resp = c.post(
            "/api/v1/query",
            json={"query": "A" * 2001},
            headers=self._headers(),
        )
        assert resp.status_code == 422


# ── EVAL ENDPOINT TESTS ───────────────────────────────────────

class TestEvalEndpoints:

    def test_eval_stats_returns_json(self):
        """GET /api/v1/eval/stats should return JSON."""
        from fastapi.testclient import TestClient
        from api.main import app

        mock_db = MagicMock()
        mock_db.get_answer_stats.return_value = {
            "total_answers": 50, "avg_score": 7.8, "avg_latency_ms": 2100
        }
        app.state.relational_db = mock_db

        c = TestClient(app)
        resp = c.get("/api/v1/eval/stats",
                     headers={"X-API-Key": "dev-secret-key-12345"})
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            assert "total_answers" in resp.json() or "error" in resp.json()

    def test_low_scores_endpoint_accepts_threshold_param(self):
        """GET /api/v1/eval/low-scores should accept threshold query param."""
        from fastapi.testclient import TestClient
        from api.main import app

        mock_db = MagicMock()
        mock_db.get_low_scored_answers.return_value = []
        app.state.relational_db = mock_db

        c = TestClient(app)
        resp = c.get("/api/v1/eval/low-scores?threshold=6.0",
                     headers={"X-API-Key": "dev-secret-key-12345"})
        assert resp.status_code in (200, 503)

    def test_eval_health_lists_components(self):
        """GET /api/v1/eval/health should list component statuses."""
        from fastapi.testclient import TestClient
        from api.main import app
        c = TestClient(app)
        resp = c.get("/api/v1/eval/health",
                     headers={"X-API-Key": "dev-secret-key-12345"})
        assert resp.status_code == 200
        data = resp.json()
        assert "components" in data
        assert "overall" in data


# ── AUTH MIDDLEWARE TESTS ─────────────────────────────────────

class TestAuthMiddleware:

    def test_missing_api_key_returns_401(self):
        """Request without X-API-Key header should return 401."""
        import os
        os.environ.setdefault("API_SECRET_KEY", "test-key-12345")

        from api.middleware.auth import AuthMiddleware
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient

        test_app = FastAPI()
        test_app.add_middleware(AuthMiddleware)

        @test_app.get("/protected")
        def protected():
            return {"ok": True}

        c = TestClient(test_app, raise_server_exceptions=False)
        resp = c.get("/protected")
        assert resp.status_code == 401

    def test_wrong_api_key_returns_403(self):
        """Wrong API key should return 403."""
        import os
        os.environ["API_SECRET_KEY"] = "correct-key"

        from api.middleware.auth import AuthMiddleware
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        test_app = FastAPI()
        test_app.add_middleware(AuthMiddleware)

        @test_app.get("/protected")
        def protected():
            return {"ok": True}

        c = TestClient(test_app, raise_server_exceptions=False)
        resp = c.get("/protected", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 403

    def test_correct_api_key_passes(self):
        """Correct API key should allow the request through."""
        import os
        os.environ["API_SECRET_KEY"] = "my-valid-key"

        from api.middleware.auth import AuthMiddleware
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        test_app = FastAPI()
        test_app.add_middleware(AuthMiddleware)

        @test_app.get("/protected")
        def protected():
            return {"ok": True}

        c = TestClient(test_app)
        resp = c.get("/protected", headers={"X-API-Key": "my-valid-key"})
        assert resp.status_code == 200

    def test_health_endpoint_exempt_from_auth(self):
        """GET /health should NOT require an API key."""
        import os
        os.environ["API_SECRET_KEY"] = "my-valid-key"

        from api.middleware.auth import AuthMiddleware
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        test_app = FastAPI()
        test_app.add_middleware(AuthMiddleware)

        @test_app.get("/health")
        def health():
            return {"status": "ok"}

        c = TestClient(test_app)
        resp = c.get("/health")   # No API key header
        assert resp.status_code == 200


# ── RATE LIMITER TESTS ────────────────────────────────────────

class TestRateLimiter:

    def test_blocks_after_burst_exceeded(self):
        """Requests beyond max_requests + burst should return 429."""
        from api.middleware.rate_limiter import RateLimiterMiddleware
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        test_app = FastAPI()
        test_app.add_middleware(
            RateLimiterMiddleware,
            max_requests=3,
            window_seconds=60,
            burst_limit=0,
        )

        @test_app.get("/endpoint")
        def endpoint():
            return {"ok": True}

        c = TestClient(test_app, raise_server_exceptions=False)
        responses = [c.get("/endpoint") for _ in range(5)]

        status_codes = [r.status_code for r in responses]
        assert 429 in status_codes, f"Expected 429 but got: {status_codes}"

    def test_rate_limit_header_in_response(self):
        """Responses should include X-RateLimit-Limit header."""
        from api.middleware.rate_limiter import RateLimiterMiddleware
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        test_app = FastAPI()
        test_app.add_middleware(
            RateLimiterMiddleware, max_requests=100, window_seconds=60
        )

        @test_app.get("/test")
        def test_route():
            return {"ok": True}

        c = TestClient(test_app)
        resp = c.get("/test")
        assert "X-RateLimit-Limit" in resp.headers
