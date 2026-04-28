# ============================================================
# api/middleware/auth.py
#
# PURPOSE: Protect API endpoints with API key authentication.
#
# BEGINNER CONCEPT - What is authentication?
# Authentication = proving who you are.
# Without auth, ANYONE on the internet can use your API.
# With auth, only clients with valid API keys can access it.
#
# HOW IT WORKS:
# Client includes header: "X-API-Key: your-secret-key-here"
# Our middleware checks if the key is valid before processing
# ============================================================

import os
import logging
from dotenv import load_dotenv
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Public endpoints that DON'T require authentication
# (health check and docs should always be accessible)
PUBLIC_PATHS = {"/", "/health", "/docs", "/redoc", "/openapi.json"}


class AuthMiddleware(BaseHTTPMiddleware):
    """
    API Key authentication middleware.

    BEGINNER CONCEPT - BaseHTTPMiddleware:
    Starlette's base class for middleware.
    You override dispatch() to add your custom logic.
    dispatch() is called for EVERY request.

    To add an API key to requests (using curl):
    curl -H "X-API-Key: your-key" http://localhost:8000/api/v1/query
    """

    async def dispatch(self, request: Request, call_next):
        """
        Check every request for a valid API key.

        Args:
            request: The incoming HTTP request
            call_next: Function to call the next middleware/route handler

        Returns:
            The response from the route handler (or 401 if auth fails)
        """

        # ✅ FIX 1: Allow OPTIONS (preflight)
        if request.method == "OPTIONS":
            return await call_next(request)

        # Skip authentication for public paths
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        # Get API key from header
        api_key = request.headers.get("X-API-Key")
        # print(api_key)
        # api_key = os.getenv("API_SECRET_KEY")

        if not api_key:
            return JSONResponse(
                status_code=401,
                content={"error": "Missing API key. Add header: X-API-Key: <your-key>"}
            )

        # Validate the key
        if not self._is_valid_key(api_key):
            logger.warning(f"Invalid API key attempt from {request.client.host}")
            return JSONResponse(
                status_code=403,
                content={"error": "Invalid API key"}
            )

        # Key is valid — add the key to request state for use in routes
        request.state.api_key = api_key
        return await call_next(request)

    def _is_valid_key(self, key: str) -> bool:
        """
        Check if the API key is valid.

        In production:
        - Store hashed keys in a database
        - Never store plain text API keys
        - Allow multiple keys per user

        For development: accept the key from environment variable
        """
        # Get valid keys from environment variable
        # You can set multiple keys separated by commas
        valid_keys_str = os.environ.get("API_SECRET_KEY", "dev-secret-key-12345")
        # print(f"[DEBUG] ENV key loaded: '{valid_keys_str}...'")
        # print(f"[DEBUG] Received key:   '{key}...'")
        valid_keys = {k.strip() for k in valid_keys_str.split(",")}
        # print(f"[DEBUG] Valid keys: {valid_keys}")

        return key in valid_keys


# ============================================================
# api/middleware/rate_limiter.py
#
# PURPOSE: Prevent abuse by limiting requests per time window.
#
# BEGINNER CONCEPT - What is rate limiting?
# Without rate limiting, one user can send 10,000 requests per second
# and crash your server (or cost you thousands in API fees).
# Rate limiting: "You get 100 requests per minute. After that → 429 error."
#
# We use a "token bucket" algorithm:
# - Each user starts with N tokens
# - Each request uses 1 token
# - Tokens refill slowly over time
# - No tokens left? → Rejected
# ============================================================

import time
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiter.

    Default: 60 requests per minute per IP address.

    NOTE: For production with multiple server instances,
    use Redis-based rate limiting (e.g., slowapi library).
    In-memory only works for single-instance deployments.
    """

    def __init__(self, app,
                 max_requests: int = 60,      # Max requests per window
                 window_seconds: int = 60):   # Time window in seconds
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds

        # In-memory store: {ip_address: [(timestamp, count)]}
        # defaultdict automatically creates empty list for new keys
        self._requests: dict = defaultdict(list)

    async def dispatch(self, request, call_next):
        # Get client IP address
        client_ip = request.client.host if request.client else "unknown"

        # Check rate limit
        if not self._is_allowed(client_ip):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Max {self.max_requests} requests per {self.window_seconds} seconds"
                },
                headers={"Retry-After": str(self.window_seconds)}
            )

        return await call_next(request)

    def _is_allowed(self, client_ip: str) -> bool:
        """
        Check if the client is within their rate limit.

        Algorithm:
        1. Remove old timestamps (outside the time window)
        2. Count remaining requests in the window
        3. If count < max → allow and record the timestamp
        4. If count >= max → deny
        """
        now = time.time()
        window_start = now - self.window_seconds

        # Remove timestamps older than the window
        self._requests[client_ip] = [
            ts for ts in self._requests[client_ip]
            if ts > window_start
        ]

        # Count requests in current window
        current_count = len(self._requests[client_ip])

        if current_count >= self.max_requests:
            return False   # Rate limit exceeded

        # Record this request
        self._requests[client_ip].append(now)
        return True
