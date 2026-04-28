# ============================================================
# Dockerfile
#
# PURPOSE: Package the entire application into a Docker image.
#
# BEGINNER CONCEPT — What is a Dockerfile?
# A Dockerfile is a recipe. It tells Docker exactly how to
# build a container image step by step — like a script that
# sets up a fresh computer with everything needed to run our app.
#
# MULTI-STAGE BUILD:
# We use two stages to keep the final image small:
#
# Stage 1 (builder): Install ALL dependencies (including build tools)
#                    Build tools are large — we don't want them in prod.
#
# Stage 2 (production): Copy only what's needed from stage 1.
#                       Result: a lean, secure production image.
#
# WHY DOES SIZE MATTER?
# Smaller images = faster deployments, less storage cost,
# smaller attack surface (fewer packages = fewer vulnerabilities).
#
# BUILD & RUN:
#   docker build -t advanced-rag .
#   docker run -p 8000:8000 advanced-rag
# ============================================================

# ── STAGE 1: Builder ─────────────────────────────────────────
# Use official Python 3.11 slim image as the build base
# "slim" = smaller than the full image but still has pip
FROM python:3.11-slim AS builder

# Set working directory inside the container
WORKDIR /app

# BEGINNER CONCEPT — Why copy requirements.txt FIRST?
# Docker builds in layers. Each instruction is a cached layer.
# If requirements.txt hasn't changed, Docker reuses the cached
# pip install layer — saving minutes of build time!
# If we copied everything first, any code change would
# re-trigger pip install unnecessarily.

# Install system dependencies needed to build Python packages
# These are build-time dependencies only (not needed at runtime)
RUN apt-get update && apt-get install -y \
    gcc \                    
    g++ \                   
    libpq-dev \             
    && rm -rf /var/lib/apt/lists/*
# rm -rf /var/lib/apt/lists/* deletes the apt cache to shrink the image

# Copy just the requirements file first (cache optimization)
COPY requirements.txt .

# Install Python dependencies into /install directory
# --no-cache-dir: don't save pip's download cache (saves space)
# --prefix /install: install to custom location for easy copying
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── STAGE 2: Production ──────────────────────────────────────
# Start fresh from the same slim base — NO build tools
FROM python:3.11-slim AS production

WORKDIR /app

# Install RUNTIME system dependencies only
# tesseract-ocr: for OCR in image_loader.py
# libpq5:        PostgreSQL client library (for psycopg2 at runtime)
# curl:          For Docker health checks
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the installed Python packages from the builder stage
# This is the magic of multi-stage builds — we get the compiled
# packages without the build tools that compiled them
COPY --from=builder /install /usr/local

# Copy our application source code
COPY . .

# Create a non-root user for security
# BEGINNER CONCEPT — Why not run as root?
# Running as root means a security breach gives attackers
# full system access. A non-root user limits the damage.
RUN useradd --create-home --shell /bin/bash raguser && \
    chown -R raguser:raguser /app
USER raguser

# Create directories for temporary files (owned by raguser)
RUN mkdir -p /tmp/rag_uploads /tmp/rag_outputs

# The port our app listens on
EXPOSE 8000

# Health check — Docker uses this to know if the container is healthy
# Runs every 30 seconds. After 3 failures → container marked unhealthy.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# The command that runs when the container starts
# uvicorn: the ASGI server
# api.main:app: Python module path → file api/main.py, variable app
# --host 0.0.0.0: listen on all network interfaces (not just localhost)
# --port 8000: listen on port 8000
# --workers 2: 2 worker processes (handle concurrent requests)
# NOTE: Remove --reload in production! It watches for file changes (dev only)
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
