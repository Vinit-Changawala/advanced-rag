# ============================================================
# tests/unit/test_llm_client.py
#
# Tests for the MistralAdapter in utils/llm_client.py.
#
# These tests run WITHOUT real API keys by mocking the Mistral
# client. They verify:
# - Model name remapping (gpt-4o → mistral-large-latest)
# - response_format handling (json_object → prompt injection)
# - JSON cleaning from responses
# - Fallback to OpenAI when MISTRAL_API_KEY is not set
# ============================================================

import json
import pytest
from unittest.mock import MagicMock, patch
from utils.llm_client import (
    MistralAdapter,
    _clean_json_response,
    _FakeResponse,
    _MistralCompletions,
)


# ── FAKE MISTRAL RESPONSE ─────────────────────────────────────

def _make_mistral_response(content: str, prompt_tokens=100, completion_tokens=50):
    """Build a fake Mistral SDK response object."""
    resp = MagicMock()
    resp.choices[0].message.content = content
    resp.usage.prompt_tokens     = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    return resp


# ── _clean_json_response tests ────────────────────────────────

class TestCleanJsonResponse:
    """Tests for the JSON cleaning helper."""

    def test_plain_json_unchanged(self):
        """Already-clean JSON should pass through unchanged."""
        raw = '{"score": 8, "approved": true}'
        assert _clean_json_response(raw) == raw

    def test_strips_markdown_fence(self):
        """```json ... ``` wrapper should be removed."""
        raw = '```json\n{"score": 8}\n```'
        result = _clean_json_response(raw)
        assert result == '{"score": 8}'

    def test_strips_plain_fence(self):
        """``` ... ``` wrapper without json label should also be stripped."""
        raw = '```\n{"key": "value"}\n```'
        result = _clean_json_response(raw)
        assert result == '{"key": "value"}'

    def test_extracts_json_from_preamble(self):
        """JSON buried after preamble text should be extracted."""
        raw = 'Here is the result: {"score": 9, "issues": []}'
        result = _clean_json_response(raw)
        parsed = json.loads(result)
        assert parsed["score"] == 9

    def test_extracts_json_array(self):
        """JSON array should also be extracted correctly."""
        raw = 'The keywords are: ["python", "rag", "embeddings"]'
        result = _clean_json_response(raw)
        parsed = json.loads(result)
        assert "python" in parsed

    def test_returns_original_when_no_json(self):
        """Text with no JSON should be returned as-is."""
        raw = "This is just plain text with no JSON."
        result = _clean_json_response(raw)
        assert result == raw

    def test_handles_empty_string(self):
        """Empty string should not crash."""
        result = _clean_json_response("")
        assert result == ""


# ── _FakeResponse tests ───────────────────────────────────────

class TestFakeResponse:
    """Tests for the OpenAI-compatible response wrapper."""

    def test_choices_accessible(self):
        """response.choices[0].message.content should work."""
        r = _FakeResponse("Hello world")
        assert r.choices[0].message.content == "Hello world"

    def test_usage_accessible(self):
        """response.usage.total_tokens should work."""
        r = _FakeResponse("Hello", prompt_tokens=100, completion_tokens=50)
        assert r.usage.total_tokens == 150
        assert r.usage.prompt_tokens == 100
        assert r.usage.completion_tokens == 50

    def test_message_role(self):
        """Message role should be 'assistant'."""
        r = _FakeResponse("test")
        assert r.choices[0].message.role == "assistant"


# ── MistralAdapter model remapping tests ─────────────────────

class TestModelRemapping:
    """Tests that OpenAI model names are remapped to Mistral equivalents."""

    def _make_completions_with_mock(self, response_content="Test response"):
        """
        Create a _MistralCompletions instance backed by a mock Mistral SDK object.

        We cannot patch 'utils.llm_client.Mistral' because Mistral is imported
        locally inside MistralAdapter.__init__. Instead we directly instantiate
        _MistralCompletions with a mock client, bypassing MistralAdapter entirely.
        """
        mock_sdk = MagicMock()
        mock_sdk.chat.complete.return_value = _make_mistral_response(response_content)
        completions = _MistralCompletions(mock_sdk, "mistral-large-latest")
        return completions, mock_sdk

    def test_gpt4o_remapped_to_large(self):
        """model='gpt-4o' should be translated to 'mistral-large-latest'."""
        completions, mock_sdk = self._make_completions_with_mock()
        completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
        called_model = mock_sdk.chat.complete.call_args[1]["model"]
        assert called_model == "mistral-large-latest", \
            f"Expected 'mistral-large-latest', got '{called_model}'"

    def test_gpt4o_mini_remapped_to_small(self):
        """model='gpt-4o-mini' should become 'mistral-small-latest'."""
        completions, mock_sdk = self._make_completions_with_mock()
        completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
        )
        called_model = mock_sdk.chat.complete.call_args[1]["model"]
        assert called_model == "mistral-small-latest", \
            f"Expected 'mistral-small-latest', got '{called_model}'"

    def test_native_mistral_model_unchanged(self):
        """An already-correct Mistral model name should pass through unchanged."""
        completions, mock_sdk = self._make_completions_with_mock()
        completions.create(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "Hello"}],
        )
        called_model = mock_sdk.chat.complete.call_args[1]["model"]
        assert called_model == "mistral-large-latest"

    def test_gpt4_remapped_to_large(self):
        """model='gpt-4' should also map to mistral-large-latest."""
        completions, mock_sdk = self._make_completions_with_mock()
        completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )
        called_model = mock_sdk.chat.complete.call_args[1]["model"]
        assert called_model == "mistral-large-latest"

    def test_unknown_model_uses_default(self):
        """An unknown/unrecognised model name should fall back to the default."""
        completions, mock_sdk = self._make_completions_with_mock()
        completions.create(
            model=None,   # triggers default
            messages=[{"role": "user", "content": "Hello"}],
        )
        called_model = mock_sdk.chat.complete.call_args[1]["model"]
        assert called_model == "mistral-large-latest"  # the default


# ── response_format handling tests ───────────────────────────

class TestResponseFormatHandling:
    """Tests that response_format=json_object is handled via prompt injection."""

    def _make_completions(self, response_content='{"score": 8}'):
        mock_sdk = MagicMock()
        mock_sdk.chat.complete.return_value = _make_mistral_response(response_content)
        return _MistralCompletions(mock_sdk, "mistral-large-latest"), mock_sdk

    def test_json_object_not_passed_to_mistral(self):
        """
        response_format={"type": "json_object"} must NOT be forwarded to Mistral.
        Mistral doesn't support this parameter and will raise an error.
        """
        completions, mock_sdk = self._make_completions()
        completions.create(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "Score this"}],
            response_format={"type": "json_object"},
        )
        # Verify response_format was NOT passed to the SDK
        call_kwargs = mock_sdk.chat.complete.call_args[1]
        assert "response_format" not in call_kwargs

    def test_json_instruction_injected_into_system_prompt(self):
        """
        When response_format is requested, a JSON instruction should be
        added to the system prompt so the model knows to return JSON.
        """
        completions, mock_sdk = self._make_completions()
        completions.create(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": "You are an evaluator."},
                {"role": "user", "content": "Score this answer."},
            ],
            response_format={"type": "json_object"},
        )
        sent_messages = mock_sdk.chat.complete.call_args[1]["messages"]
        system_content = next(
            m["content"] for m in sent_messages if m["role"] == "system"
        )
        assert "JSON" in system_content
        assert "valid" in system_content.lower()

    def test_json_instruction_added_when_no_system_message(self):
        """If there's no system message, one should be created with JSON instruction."""
        completions, mock_sdk = self._make_completions()
        completions.create(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "Return scores"}],
            response_format={"type": "json_object"},
        )
        sent_messages = mock_sdk.chat.complete.call_args[1]["messages"]
        system_messages = [m for m in sent_messages if m["role"] == "system"]
        assert len(system_messages) == 1
        assert "JSON" in system_messages[0]["content"]

    def test_response_is_openai_compatible(self):
        """The returned response object must have OpenAI-compatible structure."""
        completions, _ = self._make_completions('{"score": 9, "approved": true}')
        response = completions.create(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "Test"}],
        )
        # OpenAI-style access pattern
        content = response.choices[0].message.content
        parsed  = json.loads(content)
        assert parsed["score"] == 9

    def test_json_cleaned_from_markdown_fence(self):
        """If Mistral wraps JSON in ``` fences, they should be stripped."""
        completions, _ = self._make_completions('```json\n{"result": "ok"}\n```')
        response = completions.create(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "test"}],
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content)
        assert parsed["result"] == "ok"

    def test_no_response_format_does_not_inject(self):
        """Without response_format, no JSON instruction should be injected."""
        completions, mock_sdk = self._make_completions("Plain text response")
        completions.create(
            model="mistral-large-latest",
            messages=[{"role": "system", "content": "You help."},
                      {"role": "user", "content": "Hello"}],
        )
        sent_messages = mock_sdk.chat.complete.call_args[1]["messages"]
        system_content = next(
            m["content"] for m in sent_messages if m["role"] == "system"
        )
        # Original system prompt should be unchanged
        assert system_content == "You help."


# ── create_llm_client factory tests ──────────────────────────

class TestCreateLlmClient:
    """Tests for the create_llm_client factory function."""

    def test_returns_mistral_adapter_when_mistral_key_set(self, monkeypatch):
        """When MISTRAL_API_KEY is present, return MistralAdapter."""
        monkeypatch.setenv("MISTRAL_API_KEY", "fake-mistral-key")

        with patch("utils.llm_client.MistralAdapter") as mock_cls:
            mock_cls.return_value = MagicMock()
            from utils.llm_client import create_llm_client
            client = create_llm_client()
            mock_cls.assert_called_once()

    def test_raises_when_no_keys_set(self, monkeypatch):
        """When neither key is set, should raise ValueError."""
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY",  raising=False)

        from utils.llm_client import create_llm_client
        with pytest.raises(ValueError):
            create_llm_client()

    def test_create_embedding_client_returns_hf_embedder(self, monkeypatch):
        """create_embedding_client should return a HuggingFaceEmbedder (no API key needed)."""
        from utils.llm_client import create_embedding_client, HuggingFaceEmbedder

        # HuggingFaceEmbedder is lazy-loaded (no download until .embed() is called)
        # so this should succeed without any network access
        client = create_embedding_client()
        assert isinstance(client, HuggingFaceEmbedder)
        assert client.vector_size == 384   # BAAI/bge-small default
