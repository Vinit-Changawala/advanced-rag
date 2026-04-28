# api/routes/ingest.py  — Document ingestion endpoint

import os, uuid, logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

UPLOAD_DIR = Path("/tmp/rag_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job status store (use Redis in production)
_JOB_STATUS: dict = {}


class IngestResponse(BaseModel):
    status:         str
    job_id:         str
    files_received: int
    message:        str


class IngestStatusResponse(BaseModel):
    job_id:        str
    status:        str   # "pending" | "processing" | "done" | "failed"
    files_total:   int
    files_done:    int
    chunks_created: int
    error:         Optional[str] = None


@router.post("/ingest", response_model=IngestResponse)
async def ingest_files(
    request:          Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Documents to ingest (PDF, DOCX, TXT, CSV, PY, PNG…)"),
):
    """
    Upload documents into the RAG knowledge base.

    Processing happens in the background so you get a response
    immediately. Poll GET /api/v1/ingest/status/{job_id} to track progress.

    Supported file types: PDF, DOCX, TXT, MD, PY, JS, CSV, XLSX, PNG, JPG
    """
    if not files:
        raise HTTPException(400, "No files provided")

    job_id     = str(uuid.uuid4())
    saved_paths: List[str] = []

    for file in files:
        dest = UPLOAD_DIR / f"{job_id}_{file.filename}"
        try:
            content = await file.read()
            dest.write_bytes(content)
            saved_paths.append(str(dest))
            logger.info(f"Saved upload: {file.filename} ({len(content):,} bytes)")
        except Exception as e:
            logger.error(f"Failed to save {file.filename}: {e}")

    if not saved_paths:
        raise HTTPException(500, "All file saves failed")

    _JOB_STATUS[job_id] = {
        "status":        "pending",
        "files_total":   len(saved_paths),
        "files_done":    0,
        "chunks_created": 0,
        "error":         None,
    }

    # Kick off background processing
    background_tasks.add_task(
        _process_files,
        job_id=job_id,
        file_paths=saved_paths,
        app_state=request.app.state,
    )

    return IngestResponse(
        status="processing",
        job_id=job_id,
        files_received=len(saved_paths),
        message=f"Processing {len(saved_paths)} file(s) in background. "
                f"Check status at /api/v1/ingest/status/{job_id}",
    )


@router.get("/ingest/status/{job_id}", response_model=IngestStatusResponse)
async def ingest_status(job_id: str):
    """Check the processing status of a previously submitted ingestion job."""
    job = _JOB_STATUS.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")
    return IngestStatusResponse(job_id=job_id, **job)


@router.delete("/ingest/source")
async def delete_source(source_path: str, request: Request):
    """
    Remove all chunks from a specific source file.
    Useful when you want to re-ingest an updated version of a document.

    Example: DELETE /api/v1/ingest/source?source_path=docs/report.pdf
    """
    s = request.app.state
    if not getattr(s, "vector_store", None):
        raise HTTPException(503, "Vector store unavailable")

    try:
        s.vector_store.delete_by_source(source_path)
        if getattr(s, "relational_db", None):
            # Also remove from relational DB if that source exists there
            pass  # optional — chunks table doesn't have a delete-by-source yet
        return {"deleted": True, "source": source_path}
    except Exception as e:
        raise HTTPException(500, f"Delete failed: {e}")
    
@router.get("/ingest/sources")
async def list_sources(request: Request):
    """
    Returns all filenames currently ingested in the knowledge base.
    Used by the frontend to show the persistent file list.
    """
    s = request.app.state
    if getattr(s, "vector_store", None) is None:
        return {"sources": []}
    try:
        sources = s.vector_store.list_sources()
        return {"sources": sources, "count": len(sources)}
    except Exception as e:
        return {"sources": [], "error": str(e)}


@router.delete("/ingest/file")
async def delete_file(filename: str, request: Request):
    """
    Delete all chunks for a given filename from the knowledge base.
    Example: DELETE /api/v1/ingest/file?filename=RoBERTa.pdf
    """
    s = request.app.state
    if getattr(s, "vector_store", None) is None:
        raise HTTPException(503, "Vector store unavailable")
    try:
        s.vector_store.delete_by_source(filename)
        return {"deleted": True, "filename": filename}
    except Exception as e:
        raise HTTPException(500, f"Delete failed: {e}")


# ── BACKGROUND TASK ───────────────────────────────────────────

async def _process_files(job_id: str, file_paths: List[str], app_state):
    """
    Full ingestion pipeline running in background.

    For each file:
      1. Choose the right loader based on extension
      2. Run preprocessing pipeline (restructure → chunk → metadata)
      3. Upsert chunks into Qdrant and PostgreSQL
    """
    _JOB_STATUS[job_id]["status"] = "processing"
    total_chunks = 0

    pipeline = getattr(app_state, "preprocessing_pipeline", None)
    vs       = getattr(app_state, "vector_store",           None)
    db       = getattr(app_state, "relational_db",          None)

    if pipeline is None or vs is None:
        _JOB_STATUS[job_id]["status"] = "failed"
        _JOB_STATUS[job_id]["error"]  = (
            "Pipeline or vector store not available. "
            "Check startup logs."
        )
        return

    for i, file_path in enumerate(file_paths):
        try:
            logger.info(f"[job:{job_id[:8]}] Processing file {i+1}/{len(file_paths)}: "
                        f"{Path(file_path).name}")

            # 1. Load the file into a raw document dict
            raw_doc = _load_file(file_path, app_state)
            if raw_doc is None:
                logger.warning(f"Skipped unsupported file: {file_path}")
                continue

            # ── DUPLICATE CHECK ─────────────────────────────────
            # If this file was already ingested, delete old chunks first
            # then re-ingest. This prevents duplicate answers.
            original_name = Path(file_path).name
            if len(original_name) > 37 and original_name[36] == "_":
                original_name = original_name[37:]

            if vs.source_exists(original_name):
                logger.info(f"[job:{job_id[:8]}] File already exists — removing old chunks: {original_name}")
                vs.delete_by_source(original_name)

            # 2. Run preprocessing: restructure → chunk → metadata
            chunks = pipeline.process(raw_doc)
            logger.info(f"[job:{job_id[:8]}] Created {len(chunks)} chunks from "
                        f"{Path(file_path).name}")

            # 3. Store in vector database (Qdrant)
            vs.upsert_batch(chunks, batch_size=50)

            # 4. Store metadata in relational database (PostgreSQL)
            if db:
                for chunk in chunks:
                    try:
                        db.save_chunk_metadata(chunk)
                    except Exception:
                        pass   # Non-fatal: vector search still works without metadata

            total_chunks += len(chunks)
            _JOB_STATUS[job_id]["files_done"]    += 1
            _JOB_STATUS[job_id]["chunks_created"]  = total_chunks

        except Exception as e:
            # logger.error(f"[job:{job_id[:8]}] Failed on {file_path}: {e}")
            # _JOB_STATUS[job_id]["error"] = f"Error on {Path(file_path).name}: {str(e)}"
            import traceback
            logger.error(f"[job:{job_id[:8]}] FAILED on {Path(file_path).name}: {e}")
            logger.error(traceback.format_exc())   # ← shows the full error stacktrace
            _JOB_STATUS[job_id]["error"] = f"Error on {Path(file_path).name}: {str(e)}"

        finally:
            # Clean up the temp file regardless of success/failure
            try:
                os.remove(file_path)
            except Exception:
                pass

    _JOB_STATUS[job_id]["status"] = "done"
    logger.info(f"[job:{job_id[:8]}] Ingestion complete: "
                f"{_JOB_STATUS[job_id]['files_done']}/{len(file_paths)} files, "
                f"{total_chunks} chunks")


def _load_file(file_path: str, app_state=None):
    """
    Load a file using the correct loader based on its extension.
    Returns a raw document dict, or None if the type is unsupported.
    """
    path = Path(file_path)
    ext  = path.suffix.lower()

    try:
        # Documents: PDF, DOCX, TXT, Markdown
        if ext in {".pdf", ".docx", ".txt", ".md", ".markdown"}:
            from data_sources.document_loader import DocumentLoader
            return DocumentLoader().load(file_path)

        # Code files
        elif ext in {".py", ".js", ".ts", ".java", ".cpp", ".c", ".go",
                     ".rb", ".php", ".cs", ".rs", ".sql", ".sh",
                     ".yaml", ".yml", ".json", ".html", ".css"}:
            from data_sources.code_loader import CodeLoader
            return CodeLoader().load(file_path)

        # Spreadsheets
        elif ext in {".csv", ".xlsx", ".xls", ".tsv"}:
            from data_sources.spreadsheet_loader import SpreadsheetLoader
            return SpreadsheetLoader().load(file_path)

        # Images
        # elif ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}:
        #     from data_sources.image_loader import ImageLoader
        #     return ImageLoader(use_ocr=True).load(file_path)
        elif ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}:
            from data_sources.image_loader import ImageLoader
            # Pass the Mistral LLM client so vision AI can describe the image
            # app_state is passed down from _process_files
            loader = ImageLoader(
                mistral_client=getattr(app_state, "llm_client", None),
                use_ocr=True
            )
            return loader.load(file_path)


        else:
            logger.warning(f"Unsupported file type: {ext} ({path.name})")
            return None

    except Exception as e:
        logger.error(f"Loader failed for {path.name}: {e}")
        return None
