# ============================================================
# api/middleware/rate_limiter.py
#
# PURPOSE: Prevent API abuse by limiting requests per time window.
#
# BEGINNER CONCEPT — Why Rate Limiting?
# Without rate limiting:
# - A single script could send 10,000 requests per second
# - This could crash your server (DDoS)
# - It could rack up huge OpenAI API bills ($$$)
# - It's unfair to other users
#
# With rate limiting:
# "You get 60 requests per minute. After that, you wait."
#
# WE USE TOKEN BUCKET ALGORITHM:
# Imagine each user has a bucket with 60 tokens.
# Each request uses 1 token.
# Tokens refill at 1 per second (60 per minute).
# If bucket is empty → request is rejected with HTTP 429.
#
# WHY HTTP 429?
# 429 is the official HTTP status code for "Too Many Requests".
# Good clients see this and know to slow down.
# ============================================================

import time
import logging
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Endpoints to EXEMPT from rate limiting (monitoring tools need unrestricted access)
EXEMPT_PATHS = {"/health", "/", "/docs", "/redoc", "/openapi.json"}


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Token-bucket rate limiter middleware.

    Default behavior: 60 requests per minute per IP address.

    For production with multiple server instances, use Redis-based
    rate limiting (e.g., `slowapi` with Redis backend).
    This in-memory version only works for single-instance deployments.

    Usage (added in api/main.py):
        app.add_middleware(
            RateLimiterMiddleware,
            max_requests=60,
            window_seconds=60
        )
    """

    def __init__(self,
                 app,
                 max_requests: int = 60,
                 window_seconds: int = 60,
                 burst_limit: int = 10):
        """
        Args:
            app: The ASGI application (FastAPI app)
            max_requests: Maximum requests allowed per window
            window_seconds: Duration of the sliding window in seconds
            burst_limit: Extra requests allowed in a short burst
                        (handles legitimate traffic spikes)
        """
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.burst_limit = burst_limit

        # Storage: {ip_address: [timestamp1, timestamp2, ...]}
        # Each timestamp represents one request made by that IP
        # defaultdict(list) creates empty list for new IPs automatically
        self._request_log: dict = defaultdict(list)

        logger.info(
            f"RateLimiter: {max_requests} req/{window_seconds}s per IP "
            f"(burst: +{burst_limit})"
        )

    async def dispatch(self, request, call_next):
        """
        Check rate limit for every incoming request.

        This method runs BEFORE the route handler.
        If over limit → return 429 immediately (route handler never runs).
        If under limit → let the request through.
        """
        # Skip rate limiting for exempt paths
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        # Get the client's IP address
        client_ip = self._get_client_ip(request)

        # Check if this IP is within their rate limit
        allowed, current_count, reset_after = self._check_rate_limit(client_ip)

        if not allowed:
            logger.warning(
                f"Rate limit exceeded: IP={client_ip}, "
                f"count={current_count}/{self.max_requests}"
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"You have made too many requests. "
                              f"Limit: {self.max_requests} per {self.window_seconds} seconds.",
                    "retry_after_seconds": reset_after,
                },
                headers={
                    "Retry-After": str(reset_after),
                    "X-RateLimit-Limit": str(self.max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + reset_after),
                }
            )

        # Request is allowed — add rate limit info to response headers
        response = await call_next(request)
        remaining = max(0, self.max_requests - current_count)
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response

    def _check_rate_limit(self, client_ip: str):
        """
        Check if the IP is within rate limits and record this request.

        Returns:
            Tuple of (allowed: bool, current_count: int, reset_after: int)

        HOW THE SLIDING WINDOW WORKS:
        We keep timestamps of all requests within the last N seconds.
        Old timestamps (outside the window) are removed each time.
        Current count = remaining timestamps.
        """
        now = time.time()
        window_start = now - self.window_seconds

        # Remove timestamps outside the current window (sliding window cleanup)
        self._request_log[client_ip] = [
            ts for ts in self._request_log[client_ip]
            if ts > window_start
        ]

        current_count = len(self._request_log[client_ip])
        effective_limit = self.max_requests + self.burst_limit

        if current_count >= effective_limit:
            # Calculate how long until the oldest request falls out of window
            oldest = min(self._request_log[client_ip]) if self._request_log[client_ip] else now
            reset_after = max(1, int(oldest + self.window_seconds - now))
            return False, current_count, reset_after

        # Record this request
        self._request_log[client_ip].append(now)
        return True, current_count + 1, 0

    def _get_client_ip(self, request) -> str:
        """
        Extract the real client IP address from the request.

        When behind a proxy/load balancer, the actual client IP
        is in the X-Forwarded-For header, not request.client.host.
        """
        # Check X-Forwarded-For header first (proxy/load balancer passes this)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # X-Forwarded-For can contain multiple IPs: "client, proxy1, proxy2"
            # The leftmost (first) IP is the original client
            return forwarded_for.split(",")[0].strip()

        # Fall back to direct connection IP
        if request.client:
            return request.client.host

        return "unknown"

    def get_stats(self) -> dict:
        """Get current rate limiter statistics (useful for monitoring)."""
        now = time.time()
        active_ips = 0
        total_requests_in_window = 0

        for ip, timestamps in self._request_log.items():
            recent = [ts for ts in timestamps if ts > now - self.window_seconds]
            if recent:
                active_ips += 1
                total_requests_in_window += len(recent)

        return {
            "tracked_ips": len(self._request_log),
            "active_ips_in_window": active_ips,
            "total_requests_in_window": total_requests_in_window,
            "window_seconds": self.window_seconds,
            "max_requests_per_window": self.max_requests,
        }
