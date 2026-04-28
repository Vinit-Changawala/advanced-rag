# ============================================================
# Makefile
#
# PURPOSE: Shortcuts for common developer commands.
#
# BEGINNER CONCEPT — What is a Makefile?
# Instead of typing long commands repeatedly, you define
# short aliases called "targets". Then run them with "make".
#
# Example:
#   Without Makefile: docker-compose up -d && uvicorn api.main:app --reload
#   With Makefile:    make dev
#
# HOW TO USE:
#   make help          → Show all available commands
#   make install       → Install all Python dependencies
#   make dev           → Start development server
#   make test          → Run all tests
#   make docker-up     → Start all Docker services
#
# REQUIRES: make (pre-installed on Mac/Linux, needs WSL on Windows)
# ============================================================

# Shell to use for all commands
SHELL := /bin/bash

# Python and pip executables
# Using ?= means "use this value if not already set"
PYTHON ?= python3
PIP    ?= pip3

# Project name (used in docker commands)
PROJECT = advanced-rag

# Colors for pretty output
RED    = \033[0;31m
GREEN  = \033[0;32m
YELLOW = \033[1;33m
BLUE   = \033[0;34m
RESET  = \033[0m

# ── DEFAULT TARGET ────────────────────────────────────────────
# Running just "make" with no arguments shows help
.DEFAULT_GOAL := help


# ── HELP ──────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "$(BLUE)Advanced RAG — Developer Commands$(RESET)"
	@echo "======================================"
	@echo ""
	@echo "$(GREEN)Setup:$(RESET)"
	@echo "  make install        Install all Python dependencies"
	@echo "  make install-dev    Install with dev tools (black, flake8)"
	@echo "  make setup          Full first-time setup (install + env + docker)"
	@echo ""
	@echo "$(GREEN)Development:$(RESET)"
	@echo "  make dev            Start FastAPI server with hot-reload"
	@echo "  make dev-docker     Start server inside Docker"
	@echo ""
	@echo "$(GREEN)Docker:$(RESET)"
	@echo "  make docker-up      Start all services (postgres, qdrant, redis, app)"
	@echo "  make docker-down    Stop all services"
	@echo "  make docker-logs    Stream logs from all containers"
	@echo "  make docker-reset   Full reset (WARNING: deletes all data!)"
	@echo ""
	@echo "$(GREEN)Testing:$(RESET)"
	@echo "  make test           Run ALL tests"
	@echo "  make test-unit      Run only unit tests (fast)"
	@echo "  make test-int       Run integration tests (needs Docker)"
	@echo "  make test-stress    Run red-team stress tests"
	@echo "  make test-cov       Run tests with coverage report"
	@echo ""
	@echo "$(GREEN)Database:$(RESET)"
	@echo "  make db-migrate     Run pending database migrations"
	@echo "  make db-seed        Seed the database with test data"
	@echo "  make db-reset       Drop and recreate all tables (loses data!)"
	@echo ""
	@echo "$(GREEN)Code Quality:$(RESET)"
	@echo "  make format         Auto-format code with black + isort"
	@echo "  make lint           Check code style with flake8"
	@echo "  make check          Run format + lint + test"
	@echo ""
	@echo "$(GREEN)Utilities:$(RESET)"
	@echo "  make clean          Remove .pyc files and caches"
	@echo "  make stress-report  Run stress tests and save report"
	@echo ""


# ── SETUP ─────────────────────────────────────────────────────
.PHONY: install
install:
	@echo "$(BLUE)Installing dependencies...$(RESET)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed (includes mistralai + openai)$(RESET)"

.PHONY: install-dev
install-dev: install
	@echo "$(BLUE)Installing dev tools...$(RESET)"
	$(PIP) install black isort flake8 mypy pytest-cov
	@echo "$(GREEN)✓ Dev tools installed$(RESET)"

.PHONY: test-adapter
test-adapter:
	@echo "$(BLUE)Testing Mistral adapter...$(RESET)"
	pytest tests/unit/test_llm_client.py -v

.PHONY: setup
setup:
	@echo "$(BLUE)First-time project setup...$(RESET)"
	@# Create .env from example if it doesn't exist
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(YELLOW)⚠ Created .env from .env.example — please fill in your API keys!$(RESET)"; \
	else \
		echo "$(GREEN)✓ .env already exists$(RESET)"; \
	fi
	$(MAKE) install
	$(MAKE) docker-up
	@echo "$(YELLOW)Waiting 10s for databases to start...$(RESET)"
	@sleep 10
	$(MAKE) db-migrate
	@echo ""
	@echo "$(GREEN)✅ Setup complete!$(RESET)"
	@echo "$(GREEN)   Run 'make dev' to start the development server$(RESET)"
	@echo "$(GREEN)   Visit http://localhost:8000/docs$(RESET)"


# ── DEVELOPMENT SERVER ────────────────────────────────────────
.PHONY: dev
dev:
	@echo "$(BLUE)Starting development server...$(RESET)"
	@echo "$(GREEN)API docs: http://localhost:8000/docs$(RESET)"
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 --log-level info

.PHONY: dev-docker
dev-docker:
	docker-compose up app


# ── DOCKER ────────────────────────────────────────────────────
.PHONY: docker-up
docker-up:
	@echo "$(BLUE)Starting all Docker services...$(RESET)"
	docker-compose up -d
	@echo "$(GREEN)✓ Services started$(RESET)"
	@echo "  PostgreSQL: localhost:5432"
	@echo "  Qdrant:     http://localhost:6333/dashboard"
	@echo "  Redis:      localhost:6379"

.PHONY: docker-down
docker-down:
	@echo "$(BLUE)Stopping all Docker services...$(RESET)"
	docker-compose down
	@echo "$(GREEN)✓ Services stopped$(RESET)"

.PHONY: docker-logs
docker-logs:
	docker-compose logs -f

.PHONY: docker-reset
docker-reset:
	@echo "$(RED)WARNING: This will delete ALL data (database, vectors)!$(RESET)"
	@read -p "Are you sure? Type 'yes' to confirm: " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		docker-compose down -v; \
		docker-compose up -d; \
		echo "$(GREEN)✓ Fresh reset complete$(RESET)"; \
	else \
		echo "Cancelled."; \
	fi

.PHONY: docker-build
docker-build:
	@echo "$(BLUE)Building Docker image...$(RESET)"
	docker build -t $(PROJECT):latest .
	@echo "$(GREEN)✓ Image built: $(PROJECT):latest$(RESET)"


# ── TESTING ───────────────────────────────────────────────────
.PHONY: test
test:
	@echo "$(BLUE)Running all tests...$(RESET)"
	pytest tests/ -v

.PHONY: test-unit
test-unit:
	@echo "$(BLUE)Running unit tests...$(RESET)"
	pytest tests/unit/ -v

.PHONY: test-int
test-int:
	@echo "$(BLUE)Running integration tests...$(RESET)"
	pytest tests/integration/ -v

.PHONY: test-stress
test-stress:
	@echo "$(BLUE)Running stress/red-team tests...$(RESET)"
	pytest tests/stress/ -v

.PHONY: test-cov
test-cov:
	@echo "$(BLUE)Running tests with coverage...$(RESET)"
	pytest tests/ --cov=. --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)Coverage report: open htmlcov/index.html$(RESET)"

.PHONY: test-fast
test-fast:
	@echo "$(BLUE)Running fast tests only (no slow/integration)...$(RESET)"
	pytest tests/unit/ -v -m "not slow"


# ── DATABASE ──────────────────────────────────────────────────
.PHONY: db-migrate
db-migrate:
	@echo "$(BLUE)Running database migrations...$(RESET)"
	$(PYTHON) database/migrations/001_initial_schema.py
	@echo "$(GREEN)✓ Migrations complete$(RESET)"

.PHONY: db-seed
db-seed:
	@echo "$(BLUE)Seeding database with test data...$(RESET)"
	$(PYTHON) -c "from database.migrations.001_initial_schema import upgrade; upgrade()"
	@echo "$(GREEN)✓ Database seeded$(RESET)"

.PHONY: db-shell
db-shell:
	@echo "$(BLUE)Opening PostgreSQL shell...$(RESET)"
	docker-compose exec postgres psql -U raguser -d ragdb


# ── CODE QUALITY ──────────────────────────────────────────────
.PHONY: format
format:
	@echo "$(BLUE)Formatting code...$(RESET)"
	black . --line-length 100
	isort . --profile black
	@echo "$(GREEN)✓ Code formatted$(RESET)"

.PHONY: lint
lint:
	@echo "$(BLUE)Linting code...$(RESET)"
	flake8 . --max-line-length 100 --exclude venv,__pycache__,.git
	@echo "$(GREEN)✓ Lint passed$(RESET)"

.PHONY: check
check: format lint test
	@echo "$(GREEN)✅ All checks passed!$(RESET)"


# ── UTILITIES ─────────────────────────────────────────────────
.PHONY: clean
clean:
	@echo "$(BLUE)Cleaning up...$(RESET)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaned$(RESET)"

.PHONY: stress-report
stress-report:
	@echo "$(BLUE)Running stress tests and saving report...$(RESET)"
	$(PYTHON) -c "\
from stress_testing.runner import RedTeamRunner; \
import json; \
runner = RedTeamRunner(query_function=lambda q: 'I cannot answer that.'); \
report = runner.run_all(); \
runner.save_report(report, 'stress_test_report.json'); \
print(report['summary'])"
	@echo "$(GREEN)✓ Report saved to stress_test_report.json$(RESET)"

.PHONY: generate-api-key
generate-api-key:
	@echo "$(BLUE)Generating a secure API key...$(RESET)"
	@$(PYTHON) -c "import secrets; print(secrets.token_urlsafe(32))"
	@echo "$(YELLOW)Add this to your .env as API_SECRET_KEY$(RESET)"
