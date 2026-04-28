# ============================================================
# tests/unit/test_planner.py
# ============================================================

import pytest
from unittest.mock import MagicMock, patch
from reasoning_engine.planner import Planner
from reasoning_engine.conditional_router import ConditionalRouter


class TestPlanner:
    """Tests for the Planner class."""

    def test_default_plan_returned_without_llm(self):
        """Without an LLM client, should return a simple default plan."""
        planner = Planner(llm_client=None)
        plan = planner.create_plan("What is the weather today?")

        assert "steps" in plan
        assert len(plan["steps"]) >= 1
        assert "plan_id" in plan
        assert plan["original_query"] == "What is the weather today?"

    def test_plan_has_required_fields(self):
        """Every plan should have the required fields."""
        planner = Planner(llm_client=None)
        plan = planner.create_plan("Test query")

        required_fields = ["plan_id", "steps", "complexity", "requires_agents"]
        for field in required_fields:
            assert field in plan, f"Plan missing required field: {field}"

    def test_each_step_has_action(self):
        """Every step in the plan should have an 'action' field."""
        planner = Planner(llm_client=None)
        plan = planner.create_plan("Tell me about Python")

        for step in plan["steps"]:
            assert "action" in step, "Each step must have an 'action'"
            assert "step_number" in step, "Each step must have a 'step_number'"

    def test_llm_plan_parsing(self):
        """Should correctly parse LLM-returned JSON plan."""
        # Create a mock LLM client that returns a known response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '''{
            "complexity": "moderate",
            "requires_agents": false,
            "steps": [
                {"step_number": 1, "action": "vector_search",
                 "description": "Search", "search_query": "test", "depends_on": []}
            ],
            "estimated_chunks_needed": 3
        }'''
        mock_client.chat.completions.create.return_value = mock_response

        planner = Planner(llm_client=mock_client)
        # Call _llm_plan directly so the mock JSON is used without going
        # through the response_format wrapper that could alter it
        plan = planner._llm_plan("Test", None)

        assert plan["complexity"] == "moderate"
        assert len(plan["steps"]) == 1


class TestConditionalRouter:
    """Tests for the ConditionalRouter class."""

    def setup_method(self):
        self.router = ConditionalRouter()

    def _make_simple_plan(self, complexity="simple", requires_agents=False, steps=2):
        return {
            "complexity": complexity,
            "requires_agents": requires_agents,
            "steps": [{"step_number": i} for i in range(steps)]
        }

    def test_simple_query_routes_direct(self):
        """Simple, non-sensitive queries should go direct."""
        plan = self._make_simple_plan("simple")
        result = self.router.route(plan, "What is the capital of France?")
        assert result["route"] == "direct"

    def test_complex_query_routes_to_agents(self):
        """Complex queries should route to multi-agent system."""
        plan = self._make_simple_plan("complex", requires_agents=True, steps=6)
        result = self.router.route(plan, "Compare all our products and summarize differences")
        assert result["route"] == "multi_agent"

    def test_sensitive_topic_routes_to_human(self):
        """Sensitive topics should always go to human review."""
        plan = self._make_simple_plan("simple")
        result = self.router.route(plan, "Give me legal advice about my lawsuit")
        assert result["route"] == "human_review"

    def test_low_confidence_routes_to_human(self):
        """Low confidence scores should trigger human review."""
        plan = self._make_simple_plan("simple")
        result = self.router.route(plan, "Normal query", confidence=0.3)
        assert result["route"] == "human_review"

    def test_escalation_threshold(self):
        """should_escalate_to_human should return True below threshold."""
        assert self.router.should_escalate_to_human(5.0, threshold=7.0) is True
        assert self.router.should_escalate_to_human(9.0, threshold=7.0) is False


# ============================================================
# tests/unit/test_agents.py
# ============================================================

from unittest.mock import MagicMock
from multi_agent_system.agent_1 import ResearchAgent, SynthesisAgent, CritiqueAgent


class TestResearchAgent:
    """Tests for the Research Agent."""

    def test_runs_without_vector_store(self):
        """Agent should handle missing vector store gracefully."""
        agent = ResearchAgent(llm_client=None, vector_store=None)
        result = agent.run(
            task={"instruction": "Find info about X"},
            context={"query": "What is X?"}
        )
        # Should succeed but return empty chunks
        assert result["success"] is True
        assert result["output"] == []

    def test_deduplicates_results(self):
        """Agent should not return duplicate chunks."""
        mock_store = MagicMock()
        # Return same chunk twice (simulating duplicates)
        mock_store.search.return_value = [
            {"chunk_id": "chunk_1", "text": "Hello world", "score": 0.9},
            {"chunk_id": "chunk_1", "text": "Hello world", "score": 0.9},  # duplicate
        ]

        agent = ResearchAgent(llm_client=None, vector_store=mock_store)
        result = agent.run(
            task={"instruction": "Find something"},
            context={"query": "something"}
        )

        assert result["chunks_found"] == 1, "Duplicates should be removed"

    def test_memory_stores_task_result(self):
        """Agent should remember completed tasks."""
        agent = ResearchAgent(llm_client=None, vector_store=None)
        task = {"instruction": "Find X"}
        context = {"query": "X"}

        agent.run(task, context)
        assert len(agent.memory) == 1


class TestCritiqueAgent:
    """Tests for the Critique Agent."""

    def test_returns_default_on_missing_llm(self):
        """Should handle missing LLM gracefully."""
        agent = CritiqueAgent(llm_client=None)
        result = agent.run(
            task={"answer_to_critique": "Some answer"},
            context={"query": "Some question", "retrieved_chunks": []}
        )
        # Should fail gracefully, not crash
        assert "success" in result
        assert "agent" in result

    def test_returns_score(self):
        """Critique result should always contain a score."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = '''{
            "score": 8,
            "answers_question": true,
            "hallucination_detected": false,
            "issues": [],
            "approved": true,
            "improved_answer": ""
        }'''

        agent = CritiqueAgent(llm_client=mock_client)
        result = agent.run(
            task={"answer_to_critique": "A good answer"},
            context={"query": "A question", "retrieved_chunks": []}
        )

        assert result["score"] == 8
        assert result["approved"] is True
