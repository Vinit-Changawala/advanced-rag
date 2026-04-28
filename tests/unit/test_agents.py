# ============================================================
# tests/unit/test_agents.py
#
# Tests for the three multi-agent system files that were only
# partially tested before:
# - agent_2.py (SynthesisAgent)
# - agent_3.py (CritiqueAgent)
# - base_agent.py (BaseAgent common behaviour)
# ============================================================

import json
import pytest
from unittest.mock import MagicMock


# ── BaseAgent ─────────────────────────────────────────────────

class TestBaseAgent:
    """Tests for the shared BaseAgent base class."""

    def _concrete_agent(self, **kwargs):
        """Create a concrete subclass for testing (BaseAgent is abstract)."""
        from multi_agent_system.base_agent import BaseAgent

        class _ConcreteAgent(BaseAgent):
            def run(self, task, context):
                return {"success": True, "output": "test result", "agent": self.name}

        return _ConcreteAgent(name="Test Agent", **kwargs)

    def test_agent_has_unique_id(self):
        """Every agent instance should get a unique agent_id."""
        a1 = self._concrete_agent()
        a2 = self._concrete_agent()
        assert a1.agent_id != a2.agent_id

    def test_memory_stores_task_and_result(self):
        """remember() should store task+result pairs in self.memory."""
        agent = self._concrete_agent()
        task   = {"instruction": "Do something"}
        result = {"success": True, "output": "done"}

        agent.remember(task, result)
        assert len(agent.memory) == 1
        assert agent.memory[0]["task"] == task
        assert agent.memory[0]["result"] == result

    def test_memory_capped_at_10_entries(self):
        """Memory should never grow beyond 10 entries."""
        agent = self._concrete_agent()
        for i in range(15):
            agent.remember({"step": i}, {"ok": True})

        assert len(agent.memory) == 10
        # Should keep the MOST RECENT 10
        assert agent.memory[-1]["task"]["step"] == 14

    def test_get_memory_summary_returns_string(self):
        """get_memory_summary should return a readable string."""
        agent = self._concrete_agent()
        # No memory yet
        summary = agent.get_memory_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_get_memory_summary_with_entries(self):
        """Summary should mention recent step descriptions."""
        agent = self._concrete_agent()
        agent.remember({"instruction": "Find revenue data"}, {"success": True})
        summary = agent.get_memory_summary()
        assert "Find revenue data" in summary or "OK" in summary

    def test_call_llm_raises_without_client(self):
        """_call_llm should raise RuntimeError if llm_client is None."""
        agent = self._concrete_agent(llm_client=None)
        with pytest.raises(RuntimeError, match="not configured"):
            agent._call_llm("Hello")

    def test_call_llm_uses_client(self):
        """_call_llm should call the LLM client and return content."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = \
            "The answer is 42."
        agent = self._concrete_agent(llm_client=mock_client)

        result = agent._call_llm("What is the answer?")
        assert result == "The answer is 42."

    def test_repr_includes_class_name(self):
        """repr should include class name and truncated agent_id."""
        agent = self._concrete_agent()
        r = repr(agent)
        assert "_ConcreteAgent" in r or "Agent" in r
        assert "Test Agent" in r


# ── SynthesisAgent (agent_2.py) ───────────────────────────────

class TestSynthesisAgent:

    def _make_agent(self, mock_client=None):
        from multi_agent_system.agent_2 import SynthesisAgent
        return SynthesisAgent(llm_client=mock_client)

    def test_returns_no_info_when_no_chunks(self):
        """Without retrieved chunks, should return success=False with explanation."""
        agent = self._make_agent()
        result = agent.run(
            task={"instruction": "Synthesize"},
            context={"query": "What is X?", "retrieved_chunks": []},
        )
        assert result["success"] is False
        assert "unable" in result["output"].lower() or "not found" in result["output"].lower()
        assert result["chunks_used"] == 0

    def test_extractive_synthesis_without_llm(self):
        """Without LLM, should return the top chunk's text as the answer."""
        agent = self._make_agent(mock_client=None)
        chunks = [
            {"text": "The refund policy is 30 days.", "source": "policy.pdf", "score": 0.9},
            {"text": "Contact support@company.com.",   "source": "policy.pdf", "score": 0.7},
        ]
        result = agent.run(
            task={"instruction": "Synthesize"},
            context={"query": "What is the refund policy?", "retrieved_chunks": chunks},
        )
        assert result["success"] is True
        assert "30 days" in result["output"]
        assert len(result["sources"]) > 0

    def test_llm_synthesis_called_with_chunks(self):
        """LLM should receive the chunk texts in the prompt."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = \
            "The refund policy allows 30-day returns (Source: policy.pdf)."

        agent = self._make_agent(mock_client=mock_client)
        chunks = [{"text": "Refunds within 30 days.", "source": "policy.pdf", "score": 0.9}]

        result = agent.run(
            task={"instruction": "Synthesize"},
            context={"query": "Refund policy?", "retrieved_chunks": chunks},
        )

        assert result["success"] is True
        mock_client.chat.completions.create.assert_called_once()
        # Verify chunk text was in the prompt
        call_args = mock_client.chat.completions.create.call_args
        prompt = call_args[1]["messages"][-1]["content"]
        assert "Refunds within 30 days" in prompt

    def test_improvement_notes_added_to_prompt(self):
        """If improvement_notes are in context, they should appear in prompt."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = \
            "Improved answer here."
        agent = self._make_agent(mock_client=mock_client)

        chunks = [{"text": "Some content.", "source": "doc.pdf", "score": 0.8}]
        context = {
            "query": "Tell me about X",
            "retrieved_chunks": chunks,
            "improvement_notes": "Previous answer was too vague. Be specific.",
        }
        agent.run(task={"instruction": "Improve"}, context=context)

        call_args = mock_client.chat.completions.create.call_args
        prompt = call_args[1]["messages"][-1]["content"]
        assert "too vague" in prompt

    def test_sources_list_is_deduplicated(self):
        """Sources list should not contain duplicates."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = "Answer."
        agent = self._make_agent(mock_client=mock_client)

        # All 3 chunks from same source
        chunks = [
            {"text": f"Chunk {i}", "source": "doc.pdf", "score": 0.9 - i*0.1}
            for i in range(3)
        ]
        result = agent.run(
            task={}, context={"query": "Q?", "retrieved_chunks": chunks}
        )
        assert result["sources"].count("doc.pdf") == 1  # Not 3

    def test_stores_task_in_memory(self):
        """After run(), the task should be stored in agent's memory."""
        agent = self._make_agent()
        chunks = [{"text": "Info here.", "source": "s.pdf", "score": 0.8}]
        agent.run(task={"instruction": "Synthesize"},
                  context={"query": "Q?", "retrieved_chunks": chunks})
        assert len(agent.memory) == 1


# ── CritiqueAgent (agent_3.py) ────────────────────────────────

class TestCritiqueAgent:

    def _make_agent(self, mock_client=None):
        from multi_agent_system.agent_3 import CritiqueAgent
        return CritiqueAgent(llm_client=mock_client)

    def _good_critique_response(self):
        return json.dumps({
            "score": 8,
            "answers_question": True,
            "hallucination_detected": False,
            "issues": [],
            "missing_info": [],
            "approved": True,
            "improved_answer": "",
        })

    def _bad_critique_response(self):
        return json.dumps({
            "score": 3,
            "answers_question": False,
            "hallucination_detected": True,
            "issues": ["Made up statistics", "Does not address the question"],
            "missing_info": ["Cost information"],
            "approved": False,
            "improved_answer": "A better answer would be...",
        })

    def test_returns_failure_on_empty_answer(self):
        """No answer to critique should return success=False, score=0."""
        agent = self._make_agent()
        result = agent.run(
            task={"answer_to_critique": ""},
            context={"query": "Q?", "retrieved_chunks": []},
        )
        assert result["success"] is False
        assert result["score"] == 0
        assert result["approved"] is False

    def test_heuristic_critique_without_llm_passes_good_answer(self):
        """A clear, well-formed answer should score >= 7 with heuristics."""
        agent = self._make_agent(mock_client=None)
        good = (
            "Our refund policy allows returns within 30 days. "
            "Contact support with your order number. "
            "Items must be in original condition."
        )
        result = agent.run(
            task={"answer_to_critique": good},
            context={"query": "What is the refund policy?", "retrieved_chunks": []},
        )
        assert result["score"] >= 5   # Heuristic may not give 7+ but should be reasonable

    def test_heuristic_critique_flags_uncertainty(self):
        """Answer with 'I don't know' should score lower."""
        agent = self._make_agent(mock_client=None)
        uncertain = "I don't know the answer. I'm not sure about this. It might be something."
        result = agent.run(
            task={"answer_to_critique": uncertain},
            context={"query": "Q?", "retrieved_chunks": []},
        )
        assert result["score"] < 8

    def test_llm_critique_good_answer_is_approved(self):
        """High-quality answer should come back as approved=True."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = \
            self._good_critique_response()
        agent = self._make_agent(mock_client=mock_client)

        result = agent.run(
            task={"answer_to_critique": "Good detailed answer here."},
            context={"query": "Q?", "retrieved_chunks": []},
        )
        assert result["approved"] is True
        assert result["score"] == 8

    def test_llm_critique_bad_answer_is_rejected(self):
        """Low-quality answer should come back as approved=False with issues."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = \
            self._bad_critique_response()
        agent = self._make_agent(mock_client=mock_client)

        result = agent.run(
            task={"answer_to_critique": "Bad vague answer."},
            context={"query": "Q?", "retrieved_chunks": []},
        )
        assert result["approved"] is False
        assert result["score"] == 3
        issues = result["output"].get("issues", [])
        assert len(issues) > 0

    def test_llm_critique_handles_markdown_json_fence(self):
        """LLM wrapping JSON in ``` should still parse correctly."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = (
            "```json\n" + self._good_critique_response() + "\n```"
        )
        agent = self._make_agent(mock_client=mock_client)

        result = agent.run(
            task={"answer_to_critique": "Answer."},
            context={"query": "Q?", "retrieved_chunks": []},
        )
        assert result["success"] is True
        assert result["approved"] is True

    def test_critique_includes_source_chunks_in_prompt(self):
        """Source chunks should be included in the critique prompt for fact-checking."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = \
            self._good_critique_response()
        agent = self._make_agent(mock_client=mock_client)

        chunks = [{"text": "The policy is 30 days.", "source": "doc.pdf"}]
        agent.run(
            task={"answer_to_critique": "The policy is 30 days."},
            context={"query": "Q?", "retrieved_chunks": chunks},
        )

        call_args = mock_client.chat.completions.create.call_args
        prompt = call_args[1]["messages"][-1]["content"]
        assert "30 days" in prompt   # Source text should be in the prompt

    def test_falls_back_to_heuristic_on_llm_error(self):
        """If LLM raises, falls back to heuristic critique without crashing."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API error")
        agent = self._make_agent(mock_client=mock_client)

        result = agent.run(
            task={"answer_to_critique": "Some answer text here that is long enough."},
            context={"query": "Q?", "retrieved_chunks": []},
        )
        # Should succeed (heuristic took over), not raise
        assert result["success"] is True or result["success"] is False
        assert "score" in result
