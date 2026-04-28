# ============================================================
# utils/app_factory.py
#
# PURPOSE: Create and wire every component the application needs.
#
# BEGINNER CONCEPT — Why a factory?
# Our app has many parts: LLM client, embedding client, vector
# store, relational DB, planner, orchestrator, etc.
# Each part needs references to other parts to work.
#
# Without a factory you'd scatter all this wiring across many
# files and it becomes impossible to track. A factory puts all
# of it in ONE place.
#
# ANALOGY: Think of this as the "assembly line" that takes all
# the individual parts and connects them together into a working
# machine before the doors open (before the first HTTP request).
#
# HOW IT IS USED:
#   In api/main.py startup event:
#       from utils.app_factory import build_app_state
#       app.state = build_app_state()
#
# Every route then accesses components via:
#       request.app.state.orchestrator
#       request.app.state.vector_store
#       etc.
# ============================================================

import os
import logging
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── APP STATE DATACLASS ────────────────────────────────────────
# A dataclass is like a plain container / struct.
# It holds all the shared components so routes can access them
# via request.app.state.XXX

@dataclass
class AppState:
    """
    Holds every shared component the application uses.

    All fields default to None so the app can start even if
    some components fail to initialise (e.g. DB is down).
    Routes should check for None before using a component.
    """
    # ── LLM clients ────────────────────────────────────
    llm_client:         object = None   # MistralAdapter or OpenAI — for chat
    embedding_client:   object = None   # Always OpenAI — for embeddings only

    # ── Databases ──────────────────────────────────────
    vector_store:       object = None   # Qdrant — semantic search
    relational_db:      object = None   # PostgreSQL — metadata, logs, stats

    # ── Reasoning pipeline ─────────────────────────────
    planner:            object = None   # Breaks queries into steps
    tool_executor:      object = None   # Runs individual plan steps
    conditional_router: object = None   # Routes to direct/agents/human

    # ── Multi-agent system ─────────────────────────────
    orchestrator:       object = None   # Coordinates the 3 agents

    # ── Preprocessing pipeline ─────────────────────────
    preprocessing_pipeline: object = None  # Document → chunks → metadata

    # ── Human validation ───────────────────────────────
    gatekeeper:         object = None
    auditor:            object = None
    strategist:         object = None

    # ── Evaluation ─────────────────────────────────────
    llm_judge:          object = None
    feedback_loop:      object = None
    latency_tracker:    object = None


def build_app_state() -> AppState:
    """
    Build and wire every application component.

    Called once at startup. Returns an AppState instance with all
    components initialised and connected to each other.

    If a component fails to start, it is logged and set to None.
    This means the app stays up even if (say) Qdrant is down —
    only features that need Qdrant will fail, not the whole server.
    """
    state = AppState()

    # ── STEP 1: Create LLM clients ────────────────────────────
    logger.info("Initialising LLM clients...")
    state.llm_client, state.embedding_client = _init_llm_clients()

    # ── STEP 2: Connect to databases ──────────────────────────
    logger.info("Connecting to databases...")
    state.vector_store  = _init_vector_store(state.embedding_client)
    state.relational_db = _init_relational_db()

    # ── STEP 3: Build preprocessing pipeline ──────────────────
    logger.info("Building preprocessing pipeline...")
    state.preprocessing_pipeline = _init_preprocessing(state.llm_client)

    # ── STEP 4: Build reasoning engine ────────────────────────
    logger.info("Building reasoning engine...")
    state.planner           = _init_planner(state.llm_client)
    state.tool_executor     = _init_tool_executor(state.vector_store,
                                                   state.relational_db,
                                                   state.llm_client)
    state.conditional_router = _init_router()

    # ── STEP 5: Build multi-agent system ──────────────────────
    logger.info("Building multi-agent orchestrator...")
    state.orchestrator = _init_orchestrator(state.llm_client, state.vector_store)

    # ── STEP 6: Build human validation ────────────────────────
    logger.info("Building human validation pipeline...")
    state.gatekeeper  = _init_gatekeeper()
    state.auditor     = _init_auditor(state.llm_client)
    state.strategist  = _init_strategist()

    # ── STEP 7: Build evaluation components ───────────────────
    logger.info("Building evaluation components...")
    state.llm_judge      = _init_llm_judge(state.llm_client)
    state.feedback_loop  = _init_feedback_loop(state.relational_db)
    state.latency_tracker = _init_latency_tracker()

    _log_startup_summary(state)
    return state


# ── PRIVATE INIT FUNCTIONS ────────────────────────────────────
# Each function creates one component and handles its own errors.
# Returning None is acceptable — the route will skip that component.

def _init_llm_clients():
    """Create the LLM client (Mistral or OpenAI) and the embedding client."""
    from utils.llm_client import create_llm_client, create_embedding_client

    llm_client = None
    embedding_client = None

    try:
        llm_client = create_llm_client()
        logger.info(f"  LLM client: {type(llm_client).__name__}")
    except Exception as e:
        logger.error(f"  LLM client FAILED: {e}")

    try:
        embedding_client = create_embedding_client()
        logger.info("  Embedding client: OpenAI text-embedding-3-small")
    except Exception as e:
        logger.error(f"  Embedding client FAILED: {e}")

    return llm_client, embedding_client


def _init_vector_store(embedding_client):
    """Connect to Qdrant vector database."""
    try:
        from database.vector_store import VectorStore
        store = VectorStore(
            host=os.environ.get("QDRANT_HOST", "localhost"),
            port=int(os.environ.get("QDRANT_PORT", "6333")),
            collection_name=os.environ.get("QDRANT_COLLECTION_NAME", "rag_chunks"),
            embedding_client=embedding_client,
        )
        count = store.count()
        logger.info(f"  Vector store: Qdrant connected ({count} vectors)")
        return store
    except Exception as e:
        logger.error(f"  Vector store FAILED: {e}")
        return None


def _init_relational_db():
    """Connect to PostgreSQL relational database."""
    try:
        from database.relational_db import RelationalDB
        db_url = os.environ.get(
            "DATABASE_URL",
            "postgresql://raguser:ragpassword@localhost:5432/ragdb"
        )
        db = RelationalDB(connection_string=db_url)
        stats = db.get_answer_stats()
        logger.info(f"  Relational DB: PostgreSQL connected "
                    f"({stats.get('total_answers', 0)} answers stored)")
        return db
    except Exception as e:
        logger.error(f"  Relational DB FAILED: {e}")
        return None


def _init_preprocessing(llm_client):
    """Build the preprocessing pipeline."""
    try:
        from data_preprocessing.pipeline import PreprocessingPipeline
        import yaml, pathlib

        # Load chunk settings from config/settings.yaml
        config_path = pathlib.Path(__file__).parent.parent / "config" / "settings.yaml"
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                raw = yaml.safe_load(f)
                config = raw.get("preprocessing", {})

        pipeline = PreprocessingPipeline(llm_client=llm_client, config=config)
        logger.info("  Preprocessing pipeline: ready")
        return pipeline
    except Exception as e:
        logger.error(f"  Preprocessing pipeline FAILED: {e}")
        return None


def _init_planner(llm_client):
    try:
        from reasoning_engine.planner import Planner
        planner = Planner(llm_client=llm_client)
        logger.info("  Planner: ready")
        return planner
    except Exception as e:
        logger.error(f"  Planner FAILED: {e}")
        return None


def _init_tool_executor(vector_store, relational_db, llm_client):
    try:
        from reasoning_engine.tool_executor import ToolExecutor
        executor = ToolExecutor(
            vector_store=vector_store,
            relational_db=relational_db,
            llm_client=llm_client,
        )
        logger.info("  Tool executor: ready")
        return executor
    except Exception as e:
        logger.error(f"  Tool executor FAILED: {e}")
        return None


def _init_router():
    try:
        from reasoning_engine.conditional_router import ConditionalRouter
        import yaml, pathlib

        config_path = pathlib.Path(__file__).parent.parent / "config" / "settings.yaml"
        router = ConditionalRouter()
        logger.info("  Conditional router: ready")
        return router
    except Exception as e:
        logger.error(f"  Router FAILED: {e}")
        return None


def _init_orchestrator(llm_client, vector_store):
    try:
        from multi_agent_system.orchestrator import MultiAgentOrchestrator
        orchestrator = MultiAgentOrchestrator(
            llm_client=llm_client,
            vector_store=vector_store,
        )
        logger.info("  Multi-agent orchestrator: ready (3 agents)")
        return orchestrator
    except Exception as e:
        logger.error(f"  Orchestrator FAILED: {e}")
        return None


def _init_gatekeeper():
    try:
        from human_validation.gatekeeper import Gatekeeper
        gk = Gatekeeper(
            min_confidence=float(os.environ.get("MIN_CONFIDENCE", "0.6")),
            min_eval_score=float(os.environ.get("EVAL_THRESHOLD", "7.0")),
        )
        logger.info("  Gatekeeper: ready")
        return gk
    except Exception as e:
        logger.error(f"  Gatekeeper FAILED: {e}")
        return None


def _init_auditor(llm_client):
    try:
        from human_validation.auditor import Auditor
        auditor = Auditor(llm_client=llm_client)
        logger.info("  Auditor: ready")
        return auditor
    except Exception as e:
        logger.error(f"  Auditor FAILED: {e}")
        return None


def _init_strategist():
    try:
        from human_validation.strategist import Strategist
        # strategist = Strategist()
        strategist = Strategist(escalate_on_medium_risk=False)
        logger.info("  Strategist: ready")
        return strategist
    except Exception as e:
        logger.error(f"  Strategist FAILED: {e}")
        return None


def _init_llm_judge(llm_client):
    try:
        from evaluation.llm_judges import LLMJudge
        judge = LLMJudge(llm_client=llm_client)
        logger.info("  LLM judge: ready")
        return judge
    except Exception as e:
        logger.error(f"  LLM judge FAILED: {e}")
        return None


def _init_feedback_loop(relational_db):
    try:
        from evaluation.feedback_loop import FeedbackLoop
        loop = FeedbackLoop(
            relational_db=relational_db,
            threshold=float(os.environ.get("EVAL_THRESHOLD", "7.0")),
        )
        logger.info("  Feedback loop: ready")
        return loop
    except Exception as e:
        logger.error(f"  Feedback loop FAILED: {e}")
        return None


def _init_latency_tracker():
    try:
        from evaluation.latency_cost import LatencyCostTracker
        tracker = LatencyCostTracker()
        logger.info("  Latency tracker: ready")
        return tracker
    except Exception as e:
        logger.error(f"  Latency tracker FAILED: {e}")
        return None


def _log_startup_summary(state: AppState):
    """Print a clean startup summary showing what started successfully."""
    fields = {
        "LLM client":       state.llm_client,
        "Embedding client": state.embedding_client,
        "Vector store":     state.vector_store,
        "Relational DB":    state.relational_db,
        "Preprocessing":    state.preprocessing_pipeline,
        "Planner":          state.planner,
        "Tool executor":    state.tool_executor,
        "Router":           state.conditional_router,
        "Orchestrator":     state.orchestrator,
        "Gatekeeper":       state.gatekeeper,
        "Auditor":          state.auditor,
        "Strategist":       state.strategist,
        "LLM judge":        state.llm_judge,
        "Feedback loop":    state.feedback_loop,
    }

    ok  = [name for name, val in fields.items() if val is not None]
    bad = [name for name, val in fields.items() if val is None]

    logger.info("=" * 50)
    logger.info(f"Startup complete: {len(ok)}/{len(fields)} components ready")
    if bad:
        logger.warning(f"Failed components: {', '.join(bad)}")
    logger.info("=" * 50)
