# Advanced Hybrid RAG System

A production-grade Hybrid Retrieval Augmented Generation system with a multi-agent pipeline, human validation, adversarial stress testing, and a continuous feedback loop. Answers natural language questions from your own uploaded documents — PDFs, CSVs, images, code files, and more.

**Zero cost to run** — uses Mistral AI (free tier) for reasoning and HuggingFace BAAI/bge embeddings (free, local) for semantic search. No OpenAI API key required.

---

## What it does

Upload any document. Ask any question. Get a cited, accurate answer — from your documents only, not from the AI's general knowledge. If the answer is not in your documents, the system says so honestly.

```
You:    "What is the leave policy?"
RAG:    "Employees receive 18 earned leave days per year.
         - Unused leave can be carried forward up to 30 days
         - Leave is encashable
         Source: company_handbook.pdf"
```

```
You:    "What is Flipkart's share price?"
RAG:    "The uploaded documents do not contain information
         about this topic. Please upload relevant documents."
```

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
    3 LLM calls                   5 LLM calls
         │                               │
         ▼                               ▼
  Vector Search              ┌───────────────────────┐
  + Synthesis                │   Multi-Agent System   │
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
                    (5-stage: load → clean → chunk → metadata → embed)

Score Threshold (0.4):
   If best Qdrant similarity score < 0.4 → "not in documents" (no LLM call)
   Prevents hallucination when topic is absent from uploaded files.
```

---

## Routing logic

| Question type             | Route                 | LLM calls | Example                                          |
| ------------------------- | --------------------- | --------- | ------------------------------------------------ |
| Simple / single topic     | Direct                | 3         | "What is the leave policy?"                      |
| Complex / multi-part      | Multi-agent           | 5         | "Compare BERT and RoBERTa across all benchmarks" |
| Topic not in documents    | Rejected at threshold | 1         | "What is Flipkart's refund policy?"              |
| Sensitive (legal/medical) | Human review          | 3         | "Should I file a lawsuit?"                       |

---

## Project structure

```
advanced-rag/
├── config/
│   ├── settings.yaml           # Global settings (Mistral model, Qdrant, DB)
│   ├── agents_config.yaml      # Agent system prompts and behaviour
│   └── eval_config.yaml        # Evaluation thresholds (default: 7.0/10)
│
├── data_sources/               # File loaders for all document types
│   ├── document_loader.py      # PDF, DOCX, TXT, Markdown
│   ├── code_loader.py          # Python, JS, Java, SQL, YAML, JSON
│   ├── image_loader.py         # Tesseract OCR + Mistral Vision (pixtral)
│   └── spreadsheet_loader.py   # CSV, XLSX with schema detection
│
├── data_preprocessing/         # 5-stage ingestion pipeline
│   ├── pipeline.py             # Orchestrates all stages
│   ├── restructuring/          # Stage 1: Parse, clean encoding, remove noise
│   ├── chunking/               # Stage 2: Structure-aware splitting at headings/paragraphs
│   └── metadata_creation/      # Stage 3: Mistral summaries + HyDE questions + TF-IDF keywords
│
├── database/
│   ├── vector_store.py         # Qdrant — semantic search, duplicate prevention,
│   │                           # per-filename delete, source listing
│   ├── relational_db.py        # PostgreSQL — answers, scores, feedback logs
│   ├── schemas.py              # Pydantic data models
│   └── migrations/             # SQL schema scripts
│
├── reasoning_engine/
│   ├── planner.py              # Mistral-based plan generator (forces vector_search first)
│   ├── tool_executor.py        # Runs plan steps, strips file paths before LLM
│   ├── conditional_router.py   # Routes: direct / multi_agent / human_review
│   └── tools/                  # Vector search, SQL query, web search (optional)
│
├── multi_agent_system/
│   ├── agent_1.py              # Research + Synthesis + Critique (complex queries)
│   │                           # Alt queries generated locally — no extra LLM call
│   ├── agent_2.py              # Synthesis Agent — plain language, no path leakage
│   ├── agent_3.py              # Critique Agent — JSON scoring, partial JSON recovery
│   └── orchestrator.py         # Coordinates pipeline with 1 retry on low scores
│
├── human_validation/
│   ├── gatekeeper.py           # Rule-based: length, uncertainty phrases, sensitive topics
│   ├── auditor.py              # LLM fact-check: every sentence vs source chunks
│   └── strategist.py           # approve / escalate / reject (escalate_on_medium=False)
│
├── evaluation/
│   ├── llm_judges.py           # LLM-as-judge 0-10 scoring
│   ├── precision_recall.py     # Retrieval quality metrics
│   ├── latency_cost.py         # Performance and cost tracking
│   └── feedback_loop.py        # Records low-score patterns for improvement
│
├── stress_testing/
│   ├── runner.py               # Red team orchestrator (206 tests)
│   ├── biased_opinions.py      # Political bias, stereotyping resistance
│   ├── information_evaluation.py # Hallucination detection
│   └── prompt_injection.py     # Jailbreak and injection attack resistance
│
├── api/
│   ├── main.py                 # FastAPI app, startup wiring, /health endpoint
│   ├── routes/
│   │   ├── ingest.py           # POST /ingest, GET /ingest/sources,
│   │   │                       # DELETE /ingest/file, GET /ingest/status/{job_id}
│   │   ├── query.py            # POST /query — full pipeline with score threshold
│   │   └── eval.py             # GET /eval/stats, /eval/health, POST /eval/stress-test
│   └── middleware/
│       ├── auth.py             # X-API-Key validation, OPTIONS bypass for CORS
│       └── rate_limiter.py     # 60 req/min per IP with burst allowance
│
├── utils/
│   ├── llm_client.py           # MistralAdapter (OpenAI interface → Mistral API)
│   │                           # HuggingFaceEmbedder (BAAI/bge, local, free)
│   │                           # Retry logic: 35s backoff for 429/503 errors
│   └── app_factory.py          # Builds and wires all 14 components at startup
│
├── tests/                      # 206 tests, all passing
│   ├── unit/                   # Isolated tests for every module
│   ├── integration/            # End-to-end pipeline tests
│   └── stress/                 # Red-team adversarial validation
│
├── index.html                # Standalone frontend (no build step)
│                               # File upload, persistent file list, delete files,
│                               # Markdown rendering, API key saved in localStorage
│
├── Dockerfile
├── docker-compose.yml          # postgres + qdrant + redis + app containers
├── requirements.txt
├── pytest.ini
├── Makefile
└── .env.example
```

---

## Quick start

### Option A — Local development (recommended for first run)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/advanced-rag.git
cd advanced-rag

# 2. Set up environment
cp .env.example .env
# Open .env and fill in: MISTRAL_API_KEY, DB_PASSWORD, API_SECRET_KEY

# 3. Install PyTorch (CPU version is sufficient)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. Install all dependencies
pip install -r requirements.txt

# 5. Start databases only
docker compose up -d postgres qdrant redis

# 6. Start the API server
uvicorn api.main:app --reload --port 8000

# 7. Open the API docs
# http://localhost:8000/docs

# 8. Open the frontend
# Open index.html directly in your browser
```

> **First query:** the HuggingFace embedding model (~130 MB) downloads automatically on first use and is cached. All subsequent starts are instant.

### Option B — Full Docker stack

```bash
cp .env.example .env
# Fill in values in .env
docker compose up -d
# App available at http://localhost:8000
```

---

## API usage

### Ingest documents

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "X-API-Key: YOUR_API_SECRET_KEY" \
  -F "files=@document.pdf" \
  -F "files=@data.csv"
# Returns immediately with job_id — processing happens in the background
```

### Check ingestion status

```bash
curl http://localhost:8000/api/v1/ingest/status/JOB_ID \
  -H "X-API-Key: YOUR_API_SECRET_KEY"
```

### List ingested files

```bash
curl http://localhost:8000/api/v1/ingest/sources \
  -H "X-API-Key: YOUR_API_SECRET_KEY"
```

### Delete a file from the knowledge base

```bash
curl -X DELETE "http://localhost:8000/api/v1/ingest/file?filename=document.pdf" \
  -H "X-API-Key: YOUR_API_SECRET_KEY"
```

### Ask a question

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "X-API-Key: YOUR_API_SECRET_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the refund policy?", "top_k": 5}'

# The router auto-decides between direct and multi-agent based on complexity.
# Do NOT set use_agents:true — this forces multi-agent for every query
# and hits the Mistral free tier rate limit on every request.
```

### Check system health

```bash
curl http://localhost:8000/health
# Returns status of all 14 components individually
```

### Performance stats

```bash
curl http://localhost:8000/api/v1/eval/stats \
  -H "X-API-Key: YOUR_API_SECRET_KEY"
```

---

## Running tests

```bash
# All 206 tests
pytest tests/ -q

# Unit tests only (no Docker needed, runs fast)
pytest tests/unit/ -q

# Test the Mistral adapter
pytest tests/unit/test_llm_client.py -v

# Integration tests (requires Docker running)
pytest tests/integration/ -q

# Red-team adversarial tests
pytest tests/stress/ -v

# With coverage report
pytest tests/ --cov=. --cov-report=html
```

---

## Key design decisions

### Why Mistral instead of OpenAI?

Mistral offers a permanent free tier (2 req/min, 1B tokens/month) with no charges. The `MistralAdapter` in `utils/llm_client.py` wraps the Mistral SDK to expose the same interface as the OpenAI SDK — all agent code calls `client.chat.completions.create()` and works without modification. Model names remap automatically: `gpt-4o` → `mistral-large-latest`, `gpt-4o-mini` → `mistral-small-latest`.

### Why HuggingFace BAAI/bge instead of OpenAI embeddings?

BAAI/bge-small-en-v1.5 is free, local, and requires no API key. It uses your GPU automatically if available, falls back to CPU otherwise. Ranked #1 on the MTEB benchmark for its size class. Downloads once (~130 MB) then works fully offline. OpenAI embeddings charge per token on every document ingested and every query.

### Why structure-aware chunking?

Naive chunking splits every N characters regardless of content, cutting sentences and tables in half. This project detects headings, tables, and paragraph boundaries first, then splits only at natural boundaries. Tables are treated as atomic units — never split across chunks. Better chunks produce better retrieval and better answers.

### Why HyDE (Hypothetical Document Embeddings)?

User questions are short and conversational ("What is the refund policy?"). Document text is formal and dense ("14.3 Returns and Refund Procedure..."). These produce dissimilar embedding vectors even when they are semantically related. HyDE pre-generates 3 natural questions per chunk during ingestion. User questions match these stored hypothetical questions far better than they match raw document text, improving recall significantly.

### Why score threshold 0.4?

When Qdrant returns chunks with cosine similarity below 0.4, the topic is genuinely absent from the uploaded documents. Without this threshold, the LLM would answer from its training knowledge — defeating the core purpose of RAG. The threshold fires before any synthesis call, giving an honest "not in documents" response in under 3 seconds with only 1 LLM call (the planner).

### Why is the LLM judge disabled by default?

The LLM judge would be the 7th API call per query. Mistral's free tier allows 2 req/min. The judge consistently hit the rate limit and logged false scores of 5.0, polluting the feedback database. It is disabled via an early return in `_evaluate()` in `api/routes/query.py`. Remove that early return to re-enable it on a paid plan.

---

## Environment variables

| Variable                 | Required                          | Description                                      |
| ------------------------ | --------------------------------- | ------------------------------------------------ |
| `MISTRAL_API_KEY`        | ✅ Yes                            | Mistral API key — get free at console.mistral.ai |
| `MISTRAL_MODEL`          | Default: `mistral-small-latest`   | Mistral model for all reasoning                  |
| `DB_PASSWORD`            | ✅ Yes                            | PostgreSQL password                              |
| `API_SECRET_KEY`         | ✅ Yes                            | API authentication key                           |
| `HF_EMBEDDING_MODEL`     | Default: `BAAI/bge-small-en-v1.5` | Local HuggingFace embedding model                |
| `QDRANT_HOST`            | Default: `localhost`              | Qdrant server host                               |
| `QDRANT_PORT`            | Default: `6333`                   | Qdrant server port                               |
| `QDRANT_COLLECTION_NAME` | Default: `rag_chunks`             | Vector collection name                           |
| `EVAL_THRESHOLD`         | Default: `7.0`                    | Minimum score before feedback is logged          |
| `SERPER_API_KEY`         | Optional                          | Web search fallback — free tier at serper.dev    |
| `LOG_LEVEL`              | Default: `INFO`                   | Logging verbosity                                |

Generate a secure API secret key:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

> `OPENAI_API_KEY` is **not required**. The project runs entirely on Mistral (chat) and HuggingFace (embeddings) — both free.

---

## Deployment

### Backend — any Linux server

```bash
# 1. Clone and enter project
git clone https://github.com/YOUR_USERNAME/advanced-rag.git
cd advanced-rag

# 2. Install dependencies
sudo apt install -y python3-venv docker.io docker-compose-plugin

# 3. Set up Python environment
python3 -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
nano .env   # fill in your values

# 5. Start databases
docker compose up -d postgres qdrant redis

# 6. Start the API server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

For persistent background running that survives reboots:

```bash
# Create a systemd service so the server starts automatically on boot
sudo nano /etc/systemd/system/rag.service
sudo systemctl enable rag
sudo systemctl start rag
```

### HTTPS setup (required for HTTPS frontends)

```bash
# 1. Point a domain to your server IP
# 2. Install Nginx as reverse proxy to localhost:8000
# 3. Issue free SSL certificate with Certbot:
sudo certbot --nginx -d YOUR_DOMAIN
# Auto-renews every 90 days
```

### Frontend

The `index.html` file is a standalone single-file frontend — no build step needed. Update the API base URL before deploying:

```javascript
// Change this line in index.html:
const API = "https://YOUR_BACKEND_DOMAIN/api/v1";
```

It can be hosted on any static file host (GitHub Pages, Netlify, Vercel, or directly served by Nginx).

---

## Known limitations on Mistral free tier

| Limitation                       | Cause                               | Behaviour                                |
| -------------------------------- | ----------------------------------- | ---------------------------------------- |
| Simple queries: 8–16s            | Mistral API latency                 | Expected — waits for API response        |
| Complex queries: 30–60s          | Rate limit retry with 35s backoff   | System retries automatically             |
| LLM judge disabled               | Would be 7th API call per query     | Score shows null; re-enable on paid plan |
| First query slower after restart | HuggingFace model loads into memory | One-time cost per server start           |

---

## Key concepts

| Concept                  | What it is                                                       | Where in this project                       |
| ------------------------ | ---------------------------------------------------------------- | ------------------------------------------- |
| RAG                      | AI searches YOUR documents before answering                      | `reasoning_engine/`, `database/`            |
| Vector Embedding         | Text converted to 384 numbers for similarity search              | `utils/llm_client.py` — HuggingFaceEmbedder |
| Structure-aware chunking | Splits at headings and paragraphs, never mid-sentence            | `data_preprocessing/chunking/`              |
| HyDE                     | Pre-generates questions per chunk for better search recall       | `metadata_creation/question_generator.py`   |
| MistralAdapter           | Makes Mistral API look like OpenAI SDK — zero agent code changes | `utils/llm_client.py`                       |
| Multi-agent pipeline     | 3 specialised agents: Research → Synthesis → Critique            | `multi_agent_system/`                       |
| Score threshold          | Prevents LLM from answering when topic is absent                 | `api/routes/query.py` — `_generate()`       |
| Human validation         | Gatekeeper + Auditor + Strategist safety layer before delivery   | `human_validation/`                         |
| Feedback loop            | Records low-score patterns for continuous improvement            | `evaluation/feedback_loop.py`               |
| Red teaming              | 206 adversarial tests: bias, hallucination, prompt injection     | `stress_testing/`                           |

---

## Tech stack

| Layer               | Technology                                       |
| ------------------- | ------------------------------------------------ |
| API Framework       | FastAPI + Uvicorn                                |
| LLM (reasoning)     | Mistral AI — mistral-small-latest                |
| Embeddings          | HuggingFace BAAI/bge-small-en-v1.5 (local, free) |
| Vector database     | Qdrant                                           |
| Relational database | PostgreSQL                                       |
| Caching             | Redis                                            |
| Data validation     | Pydantic v2                                      |
| PDF parsing         | PyMuPDF / pdfplumber                             |
| OCR                 | Tesseract + pytesseract                          |
| Image understanding | Mistral Pixtral (pixtral-12b-2409)               |
| Spreadsheets        | pandas + openpyxl                                |
| Testing             | pytest (206 tests)                               |
| Containerisation    | Docker + Docker Compose                          |
| Reverse proxy       | Nginx                                            |
| SSL                 | Let's Encrypt (Certbot)                          |

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Make sure all 206 tests pass before submitting:

```bash
pytest tests/ -q
```
