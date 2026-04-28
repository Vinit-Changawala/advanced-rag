# ============================================================
# utils/llm_client.py
#
# PURPOSE: A wrapper that makes Mistral's API look EXACTLY like
#          OpenAI's API so the rest of the code needs zero changes.
#
# WHY THIS APPROACH?
# Every file in this project calls:
#     llm_client.chat.completions.create(model=..., messages=...)
# That is the OpenAI interface.
#
# Mistral's real interface is slightly different:
#     mistral_client.chat.complete(model=..., messages=...)
#
# Instead of editing 15+ files, we write ONE adapter class here
# that translates OpenAI-style calls → Mistral-style calls.
# The rest of the project never needs to know the difference.
#
# SETUP:
# You need TWO keys in your .env:
#   MISTRAL_API_KEY=...   → for all LLM chat (answers, summaries, judges)
#   OPENAI_API_KEY=...    → ONLY for embeddings (text-embedding-3-small)
#
# HOW TO USE IN api/main.py:
#   from utils.llm_client import create_llm_client
#   llm_client = create_llm_client()   # auto-detects which provider to use
# ============================================================

import os
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# ── RESPONSE WRAPPER ──────────────────────────────────────────
# The rest of the code reads:  response.choices[0].message.content
# and:                         response.usage.total_tokens
# We build a tiny object that looks exactly like an OpenAI response.

class _FakeUsage:
    """Mimics OpenAI response.usage object."""
    def __init__(self, prompt_tokens=0, completion_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class _FakeMessage:
    """Mimics OpenAI response.choices[0].message object."""
    def __init__(self, content: str):
        self.content = content
        self.role = "assistant"


class _FakeChoice:
    """Mimics OpenAI response.choices[0] object."""
    def __init__(self, content: str):
        self.message = _FakeMessage(content)
        self.finish_reason = "stop"


class _FakeResponse:
    """
    Mimics a full OpenAI ChatCompletion response object.

    The code does: response.choices[0].message.content
    This class makes that work even though the data came from Mistral.
    """
    def __init__(self, content: str, prompt_tokens=0, completion_tokens=0):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(prompt_tokens, completion_tokens)


# ── MISTRAL CHAT COMPLETIONS ADAPTER ─────────────────────────

class _MistralCompletions:
    """
    Mimics OpenAI's client.chat.completions object.
    Exposes a .create() method with the same signature as OpenAI.
    """

    # Maps OpenAI model names → equivalent Mistral model names.
    # This means files that hardcode "gpt-4o" or "gpt-4o-mini" will
    # automatically use the correct Mistral model without any code changes.
    MODEL_MAP = {
        "gpt-4o":              "mistral-large-latest",   # Most capable
        "gpt-4o-mini":         "mistral-small-latest",   # Fast and cheap
        "gpt-4":               "mistral-large-latest",
        "gpt-4-turbo":         "mistral-large-latest",
        "gpt-3.5-turbo":       "open-mistral-7b",        # Free tier
        "gpt-3.5-turbo-0125":  "open-mistral-7b",
    }

    def __init__(self, mistral_client, default_model: str):
        self._client = mistral_client
        self._default_model = default_model

    def create(self,
               model: str = None,
               messages: List[Dict] = None,
               max_tokens: int = 1000,
               temperature: float = 0.3,
               response_format: Optional[Dict] = None,
               **kwargs) -> _FakeResponse:
        """
        Drop-in replacement for openai_client.chat.completions.create().

        KEY DIFFERENCE — response_format handling:
        OpenAI supports response_format={"type": "json_object"} natively.
        Mistral does NOT support this parameter directly.

        Our solution: if json_object is requested, we automatically add
        "Return only valid JSON, no other text." to the system prompt.
        The model then behaves the same way.
        """
        model = model or self._default_model

        # Remap OpenAI model names to Mistral equivalents.
        # This means code that says model="gpt-4o" still works — we translate it.
        if model in self.MODEL_MAP:
            mistral_model = self.MODEL_MAP[model]
            logger.debug(f"Model remapped: {model} → {mistral_model}")
            model = mistral_model

        # Deep-copy messages so we don't mutate the caller's list
        msgs = [dict(m) for m in (messages or [])]

        # Handle JSON mode: inject instruction instead of using response_format param
        wants_json = (
            response_format is not None and
            response_format.get("type") == "json_object"
        )

        if wants_json:
            # Find existing system message and append JSON instruction,
            # OR prepend a new system message if none exists
            sys_msg = next((m for m in msgs if m.get("role") == "system"), None)
            json_instruction = (
                "\n\nIMPORTANT: Your response must be ONLY valid JSON. "
                "No markdown, no code fences, no explanation before or after. "
                "Start with { and end with }."
            )
            if sys_msg:
                sys_msg["content"] += json_instruction
            else:
                msgs.insert(0, {
                    "role": "system",
                    "content": "You are a helpful assistant." + json_instruction
                })

            logger.debug("JSON mode: injected JSON instruction into system prompt")

        # Call Mistral API with automatic retry for 429 (rate limit) and 503 (server error)
        import time as _time

        last_err = None
        for attempt in range(4):   # up to 4 attempts: 0, 1, 2, 3
            if attempt > 0:
                wait = attempt * 35   # 35s, 70s, 105s — stays inside 2 req/min window
                logger.warning(f"Mistral {last_err.status_code if hasattr(last_err,'status_code') else '?'} on attempt {attempt}, waiting {wait}s...")
                _time.sleep(wait)
            try:
                response = self._client.chat.complete(
                    model=model,
                    messages=msgs,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                content_str = response.choices[0].message.content or ""

                if wants_json:
                    content_str = _clean_json_response(content_str)

                prompt_tokens     = getattr(response.usage, "prompt_tokens",     0) or 0
                completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0

                return _FakeResponse(content_str, prompt_tokens, completion_tokens)

            except Exception as e:
                last_err = e
                err_str = str(e)
                if "429" in err_str or "rate_limited" in err_str or                    "503" in err_str or "unreachable_backend" in err_str:
                    continue   # retry
                # Any other error → raise immediately
                logger.error(f"Mistral chat.complete() failed: {e}")
                raise

        logger.error(f"Mistral chat.complete() failed after 4 attempts: {last_err}")
        raise last_err


class _MistralChat:
    """Mimics OpenAI's client.chat object. Has a .completions attribute."""
    def __init__(self, mistral_client, default_model: str):
        self.completions = _MistralCompletions(mistral_client, default_model)


class MistralAdapter:
    """
    Full adapter that makes a Mistral client look like an OpenAI client.

    The rest of the codebase uses:
        llm_client.chat.completions.create(...)
    This class provides exactly that interface.

    Usage:
        adapter = MistralAdapter(api_key="your-mistral-key")
        # Now use exactly like OpenAI:
        response = adapter.chat.completions.create(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "Hello"}]
        )
        print(response.choices[0].message.content)
    """

    def __init__(self,
                 api_key: str,
                 default_model: str = "mistral-large-latest"):
        """
        Args:
            api_key: Your Mistral API key from console.mistral.ai
            default_model: Which Mistral model to use.
                          Options:
                          - "mistral-large-latest"  → Most capable, like GPT-4o
                          - "mistral-small-latest"  → Faster, cheaper, like GPT-4o-mini
                          - "open-mistral-7b"       → Free tier, less capable
        """
        try:
            from mistralai import Mistral
            self._mistral = Mistral(api_key=api_key)
            logger.info(f"MistralAdapter initialized with model: {default_model}")
        except ImportError:
            raise ImportError(
                "mistralai package not installed.\n"
                "Run: pip install mistralai"
            )

        self.chat = _MistralChat(self._mistral, default_model)
        self.default_model = default_model


# ── FACTORY FUNCTION ─────────────────────────────────────────

def create_llm_client(provider: str = None):
    """
    Create the correct LLM client based on environment variables.

    Auto-detects which provider to use:
    - If MISTRAL_API_KEY is set → uses Mistral
    - If only OPENAI_API_KEY is set → uses OpenAI

    You can also force a provider by passing it explicitly:
        create_llm_client("mistral")
        create_llm_client("openai")

    Returns:
        Either a MistralAdapter or a real OpenAI client —
        both have the same interface.
    """
    mistral_key = os.environ.get("MISTRAL_API_KEY", "")
    openai_key  = os.environ.get("OPENAI_API_KEY", "")

    # Determine provider
    if provider is None:
        provider = "mistral" if mistral_key else "openai"

    if provider == "mistral":
        if not mistral_key:
            raise ValueError(
                "MISTRAL_API_KEY not found in environment.\n"
                "Add it to your .env file: MISTRAL_API_KEY=your-key-here"
            )
        model = os.environ.get("MISTRAL_MODEL", "mistral-large-latest")
        logger.info(f"LLM provider: Mistral ({model})")
        return MistralAdapter(api_key=mistral_key, default_model=model)

    elif provider == "openai":
        if not openai_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment.\n"
                "Add it to your .env file: OPENAI_API_KEY=your-key-here"
            )
        import openai
        logger.info("LLM provider: OpenAI (gpt-4o)")
        return openai.OpenAI(api_key=openai_key)

    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'mistral' or 'openai'.")


def create_embedding_client():
    """
    Create a FREE local embedding client using HuggingFace sentence-transformers.

    Uses BAAI/bge-small-en-v1.5 — a top-ranked open-source embedding model.
    Runs 100% on your computer. No API key. No cost. No internet needed after first download.

    FIRST RUN: Downloads the model (~130 MB) from HuggingFace automatically.
    SUBSEQUENT RUNS: Uses the cached model from ~/.cache/huggingface/

    Produces 384-dimensional vectors.
    NOTE: Qdrant collection is auto-created with 384 dims to match.

    Install requirement: pip install sentence-transformers

    Returns:
        HuggingFaceEmbedder instance with an .embed(text) method.
    """
    model_name = os.environ.get("HF_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    logger.info(f"Embedding client: HuggingFace {model_name} (free, local)")
    return HuggingFaceEmbedder(model_name=model_name)


class HuggingFaceEmbedder:
    """
    Wraps sentence-transformers to produce embeddings.

    This is the embedding client used by VectorStore.embed_text().
    It exposes the same .embed(text) → list[float] interface that
    VectorStore expects.

    BEGINNER CONCEPT — What are sentence-transformers?
    They convert a sentence into a list of numbers (a vector).
    Similar sentences produce similar vectors.
    That is how semantic search works: "How do I return a product?"
    and "What is your refund policy?" end up with similar vectors,
    so searching for one also finds the other.

    BAAI/bge-small-en-v1.5 is one of the best free models:
    - Ranked #1 on the MTEB leaderboard for its size class
    - Only 130 MB
    - Fast on CPU — no GPU needed
    - Produces 384-dimensional vectors
    """

    # Dimension produced by BAAI/bge-small-en-v1.5
    # If you change the model, update this too.
    DIMENSIONS = {
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5":  768,
        "BAAI/bge-large-en-v1.5": 1024,
        "all-MiniLM-L6-v2":       384,
        "all-mpnet-base-v2":      768,
    }

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None   # lazy-load: only download when first used
        # Eagerly validate sentence-transformers is installed at startup
        # so failures surface immediately, not silently during background ingestion
        try:
            import sentence_transformers  # noqa — just checking it exists
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )

    @property
    def model(self):
        """Load the model on first use (lazy loading saves startup time)."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed.\n"
                    "Run: pip install sentence-transformers\n"
                    "The model (~130 MB) downloads automatically on first use."
                )
            logger.info(f"Loading embedding model: {self.model_name} ...")
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Embedding model loaded. Dimensions: {self.vector_size}")
        return self._model

    @property
    def vector_size(self) -> int:
        """Number of dimensions this model produces."""
        return self.DIMENSIONS.get(self.model_name, 384)

    def embed(self, text: str) -> list:
        """
        Convert a text string into a vector (list of floats).

        Args:
            text: Any text string

        Returns:
            List of floats, length = self.vector_size

        BGE models work best with a query prefix for retrieval tasks.
        When encoding queries (not documents), prefix with "Represent this
        sentence for searching relevant passages: "
        We add this automatically.
        """
        # BGE models recommend this prefix for query encoding
        if self.model_name.startswith("BAAI/bge") and not text.startswith("Represent"):
            encode_text = f"Represent this sentence for searching relevant passages: {text}"
        else:
            encode_text = text

        vector = self.model.encode(encode_text, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: list) -> list:
        """
        Embed multiple texts at once (faster than calling embed() in a loop).

        Args:
            texts: List of text strings

        Returns:
            List of vectors, one per input text
        """
        if self.model_name.startswith("BAAI/bge"):
            prefixed = [
                f"Represent this sentence for searching relevant passages: {t}"
                for t in texts
            ]
        else:
            prefixed = texts

        vectors = self.model.encode(prefixed, normalize_embeddings=True,
                                     batch_size=32, show_progress_bar=False)
        return [v.tolist() for v in vectors]


# ── PRIVATE HELPERS ──────────────────────────────────────────

def _clean_json_response(text: str) -> str:
    """
    Clean a model response to extract pure JSON.

    Even when instructed to return only JSON, models sometimes
    wrap it in ```json ... ``` or add a preamble sentence.
    This function strips all of that.
    """
    text = text.strip()

    # Remove markdown code fences: ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        inner_lines = lines[1:]
        if inner_lines and inner_lines[-1].strip() == "```":
            inner_lines = inner_lines[:-1]
        text = "\n".join(inner_lines).strip()

    # Find the first { or [ and last } or ] to extract just the JSON
    start_brace = text.find("{")
    start_bracket = text.find("[")

    if start_brace == -1 and start_bracket == -1:
        # No JSON found — return as-is and let the caller handle the error
        return text

    # Pick whichever comes first
    if start_brace == -1:
        start = start_bracket
        end_char = "]"
    elif start_bracket == -1:
        start = start_brace
        end_char = "}"
    else:
        start = min(start_brace, start_bracket)
        end_char = "}" if start == start_brace else "]"

    # Find the matching closing brace/bracket from the end
    end = text.rfind(end_char)
    if end == -1:
        return text

    extracted = text[start:end + 1]

    # Validate it's actually parseable JSON
    try:
        json.loads(extracted)
        return extracted
    except json.JSONDecodeError:
        # If cleaning broke it, return original text
        return text
