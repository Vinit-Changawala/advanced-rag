# ============================================================
# api/main.py  — FastAPI application entry point
# ============================================================

import logging
import time

from fastapi import FastAPI, Request
from fastapi.openapi.models import APIKey, APIKeyIn
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes.ingest import router as ingest_router
from api.routes.query  import router as query_router
from api.routes.eval   import router as eval_router
from api.middleware.auth         import AuthMiddleware
from api.middleware.rate_limiter import RateLimiterMiddleware

from dotenv import load_dotenv
import os
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# app = FastAPI(
#     title="Advanced RAG API",
#     description="Production-grade RAG with multi-agent pipeline, human validation, and feedback loop.",
#     version="1.0.0",
#     docs_url="/docs",
#     redoc_url="/redoc",
# )
# This tells Swagger UI to show the Authorize button and
# ask for a value to send as the X-API-Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

app = FastAPI(
    title="Advanced RAG API",
    description="Production-grade RAG with multi-agent pipeline, human validation, and feedback loop.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    # Declare the security scheme so Swagger UI shows the Authorize button
    swagger_ui_parameters={"persistAuthorization": True},
)

# Register the security scheme on the OpenAPI spec
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    # Add the X-API-Key security scheme
    schema["components"]["securitySchemes"] = {
        "APIKeyHeader": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
        }
    }
    # Apply it globally to every endpoint
    for path in schema.get("paths", {}).values():
        for method in path.values():
            method["security"] = [{"APIKeyHeader": []}]
    app.openapi_schema = schema
    return schema

app.openapi = custom_openapi


app.add_middleware(CORSMiddleware, allow_origins=["*", "http://localhost:5500"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(AuthMiddleware)
app.add_middleware(RateLimiterMiddleware)


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Response-Time"] = f"{int((time.time()-start)*1000)}ms"
    return response


@app.on_event("startup")
async def startup():
    """
    Build and wire every component at boot. Stores on app.state.
    Routes access shared components via request.app.state.XXX
    """
    logger.info("Starting Advanced RAG API...")
    try:
        from utils.app_factory import build_app_state
        state = build_app_state()
        for key, value in vars(state).items():
            setattr(app.state, key, value)
    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)
        app.state.llm_client   = None
        app.state.vector_store = None


app.include_router(ingest_router, prefix="/api/v1", tags=["Ingestion"])
app.include_router(query_router,  prefix="/api/v1", tags=["Query"])
app.include_router(eval_router,   prefix="/api/v1", tags=["Evaluation"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(status_code=500,
                        content={"error": "Internal server error", "detail": str(exc)})


@app.get("/health", tags=["System"])
async def health_check(request: Request):
    """Reports which components are ready. Used by Docker health checks."""
    components = {
        "llm_client":       getattr(request.app.state, "llm_client",      None),
        "embedding_client": getattr(request.app.state, "embedding_client", None),
        "vector_store":     getattr(request.app.state, "vector_store",     None),
        "relational_db":    getattr(request.app.state, "relational_db",    None),
        "orchestrator":     getattr(request.app.state, "orchestrator",     None),
    }
    status  = {k: ("ok" if v is not None else "unavailable") for k, v in components.items()}
    all_ok  = all(v == "ok" for v in status.values())
    return JSONResponse(
        status_code=200 if all_ok else 206,
        content={"status": "healthy" if all_ok else "degraded", "components": status},
    )


@app.get("/", tags=["System"])
async def root():
    return {"service": "Advanced RAG API", "docs": "/docs",
            "health": "/health", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
