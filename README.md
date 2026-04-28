# Advanced RAG System

A production-grade Retrieval Augmented Generation system with multi-agent validation, human review, adversarial stress testing, and a continuous feedback loop.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                   Reasoning Engine                       │
│   Planner ──► Tool Executor ──► Conditional Router      │
└────────┬───────────────────────────────┬────────────────┘
         │                               │
    [direct]                      [multi_agent]
         │                               │
         ▼                               ▼
  Vector + SQL               ┌───────────────────────┐
    Search                   │   Multi-Agent System   │
         │                   │  Agent 1 (Research)    │
         └──────────────────►│  Agent 2 (Synthesis)   │
                             │  Agent 3 (Critique)    │
                             └───────────┬────────────┘
                                         │
                                         ▼
                             ┌───────────────────────┐
                             │   Human Validation     │
                             │  Gatekeeper            │
                             │  Auditor               │
                             │  Strategist            │
                             └───────────┬────────────┘
                                         │
                                         ▼
                             ┌───────────────────────┐
                             │     Evaluation         │
                             │  LLM Judges            │
                             │  Precision & Recall    │
                             │  Latency & Cost        │
                             └───────────┬────────────┘
                                         │
                            ◄────────────┘  Feedback Loop
                         (low scores → signals back to Planner)

Data Layer:
   Data Sources ──► Preprocessing Pipeline ──► Qdrant + PostgreSQL
   Stress Testing ─────────────────────────────► Red Team Reports
```

---

## Project Structure

```
advanced-rag/
├── config/                     # YAML configuration files
│   ├── settings.yaml           # Global settings (models, DB, API)
│   ├── agents_config.yaml      # Agent prompts and tool access
│   └── eval_config.yaml        # Evaluation thresholds and rubrics
│
├── data_sources/               # File loaders for all document types
│   ├── document_loader.py      # PDF, DOCX, TXT, Markdown
│   ├── code_loader.py          # Python, JS, Java, SQL, etc.
│   ├── image_loader.py         # OCR + Vision AI descriptions
│   └── spreadsheet_loader.py   # Excel, CSV with schema detection
│
├── data_preprocessing/         # 3-stage ingestion pipeline
│   ├── pipeline.py             # Orchestrates all 3 stages
│   ├── restructuring/          # Stage 1: Parse & clean
│   ├── chunking/               # Stage 2: Split intelligently
│   └── metadata_creation/      # Stage 3: Summaries, keywords, HyDE questions
│
├── database/                   # Persistence layer
│   ├── vector_store.py         # Qdrant — semantic similarity search
│   ├── relational_db.py        # PostgreSQL — metadata, logs, stats
│   ├── schemas.py              # Pydantic data models
│   └── migrations/             # SQL schema evolution scripts
│
├── reasoning_engine/           # Query planning and execution
│   ├── planner.py              # Breaks complex queries into steps
│   ├── tool_executor.py        # Runs tools from the plan
│   ├── conditional_router.py   # Routes to direct/agents/human
│   └── tools/                  # Vector search, SQL, web search
│
├── multi_agent_system/         # 3-agent pipeline for complex queries
│   ├── agent_1.py              # Research Agent — deep retrieval
│   ├── agent_2.py              # Synthesis Agent — combine & write
│   ├── agent_3.py              # Critique Agent — validate quality
│   └── orchestrator.py         # Coordinates agent pipeline
│
├── human_validation/           # Pre-delivery quality gates
│   ├── gatekeeper.py           # Flags low-confidence answers
│   ├── auditor.py              # Fact-checks against source chunks
│   └── strategist.py           # Final approve/escalate/reject decision
│
├── evaluation/                 # Continuous quality measurement
│   ├── llm_judges.py           # LLM-as-judge scoring (0-10)
│   ├── precision_recall.py     # Retrieval quality metrics
│   ├── latency_cost.py         # Performance and cost tracking
│   └── feedback_loop.py        # Routes failures back to Planner
│
├── stress_testing/             # Adversarial red-team tests
│   ├── runner.py               # Orchestrates all test suites
│   ├── biased_opinions.py      # Political bias, stereotyping
│   ├── information_evaluation.py # Hallucination detection
│   └── prompt_injection.py     # Jailbreak & injection attacks
│
├── api/                        # FastAPI REST interface
│   ├── main.py                 # App entry point, middleware
│   ├── routes/                 # Endpoints: /ingest /query /eval
│   └── middleware/             # Auth (API keys), rate limiting
│
├── tests/                      # Three-tier test suite
│   ├── unit/                   # Fast isolated unit tests
│   ├── integration/            # End-to-end pipeline tests
│   └── stress/                 # Red-team test validation
│
├── notebooks/                  # Jupyter analysis notebooks
│   ├── chunking_experiments.ipynb
│   └── eval_analysis.ipynb
│
├── Dockerfile                  # Multi-stage production container
├── docker-compose.yml          # Full stack (app + postgres + qdrant + redis)
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Test configuration
├── Makefile                    # Developer shortcuts
└── .env.example                # Environment variable template
```

---

## Quick Start

### Option A — Docker (Recommended)

```bash
# 1. Clone and enter the project
git clone <your-repo> advanced-rag
cd advanced-rag

# 2. Set up environment variables
cp .env.example .env
# Edit .env and fill in OPENAI_API_KEY and other values

# 3. Start everything with one command
make setup

# 4. Open the API docs
open http://localhost:8000/docs
```

### Option B — Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Fill in your values

# 3. Start only the databases via Docker
docker-compose up -d postgres qdrant redis

# 4. Run the FastAPI server
make dev
# or: uvicorn api.main:app --reload --port 8000
```

---

## API Usage

### Ingest Documents

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "X-API-Key: dev-secret-key-12345" \
  -F "files=@/path/to/document.pdf" \
  -F "files=@/path/to/data.csv"
```

### Ask a Question

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "X-API-Key: dev-secret-key-12345" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is our refund policy?", "top_k": 5}'
```

### Check Performance Stats

```bash
curl http://localhost:8000/api/v1/eval/stats \
  -H "X-API-Key: dev-secret-key-12345"
```

---

## Running Tests

```bash
# All tests
make test

# Fast unit tests only
make test-unit

# Integration tests (needs Docker running)
make test-int

# Red-team adversarial tests
make test-stress

# Tests with coverage report
make test-cov
```

---

## Key Concepts for Beginners

| Concept | What It Is | Where In This Project |
|---|---|---|
| RAG | AI that searches YOUR documents before answering | `reasoning_engine/`, `database/` |
| Vector Embedding | Text converted to numbers for similarity search | `database/vector_store.py` |
| Chunking | Splitting documents into searchable pieces | `data_preprocessing/chunking/` |
| Multi-Agent | Multiple AIs with specialised roles working together | `multi_agent_system/` |
| HyDE | Pre-generating questions per chunk for better search | `metadata_creation/question_generator.py` |
| Feedback Loop | System learns from its own mistakes automatically | `evaluation/feedback_loop.py` |
| Red Teaming | Intentionally trying to break your AI to find flaws | `stress_testing/` |
| LLM Judge | Using a powerful AI to score another AI's answers | `evaluation/llm_judges.py` |

---

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ Yes | OpenAI API key for GPT-4o and embeddings |
| `DB_PASSWORD` | ✅ Yes | PostgreSQL password |
| `API_SECRET_KEY` | ✅ Yes | Shared secret for API authentication |
| `DATABASE_URL` | ✅ Yes | Full PostgreSQL connection string |
| `QDRANT_HOST` | Default: localhost | Qdrant server host |
| `SERPER_API_KEY` | Optional | For web search fallback |
| `ANTHROPIC_API_KEY` | Optional | Alternative to OpenAI |
| `LOG_LEVEL` | Default: INFO | Logging verbosity |

---

## Makefile Commands

```bash
make help          # Show all commands
make setup         # First-time setup
make dev           # Start development server
make test          # Run all tests
make docker-up     # Start all services
make docker-down   # Stop all services
make format        # Auto-format code
make lint          # Check code style
make clean         # Remove cache files
make stress-report # Run red-team tests and save report
make generate-api-key  # Generate a secure API key
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| AI Models | OpenAI GPT-4o, text-embedding-3-small |
| Vector Database | Qdrant |
| Relational Database | PostgreSQL |
| Caching | Redis |
| Data Validation | Pydantic v2 |
| ORM | SQLAlchemy 2.0 |
| Testing | pytest |
| Containerisation | Docker + Docker Compose |
| PDF Parsing | pypdf |
| OCR | Tesseract (pytesseract) |
| Spreadsheets | pandas + openpyxl |
