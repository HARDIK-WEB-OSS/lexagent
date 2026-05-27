"""
FastAPI Backend for LexAgent
Handles file upload, async analysis jobs, and serves the React frontend.
"""
import asyncio
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("lexagent.api")

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOADS_DIR.mkdir(exist_ok=True)

MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}
MAX_HISTORY = 20

# ── In-memory job store ───────────────────────────────────────────────
# {job_id: {"status": str, "result": dict|None, "error": str|None, "created_at": str}}
jobs: Dict[str, dict] = {}

# Thread pool for running sync pipeline in async context
_executor = ThreadPoolExecutor(max_workers=2)

app = FastAPI(
    title="LexAgent",
    description="Local Agentic Contract Intelligence System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Lazy-load pipeline (heavy models)
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from pipeline import Pipeline
        _pipeline = Pipeline()
    return _pipeline


# ── Routes ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the React frontend."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(str(index_path))


@app.post("/analyze")
async def analyze_contract(file: UploadFile = File(...)):
    """
    Accept PDF or DOCX file upload, start async analysis, return job_id.
    """
    # Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Accepted: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Read and validate size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content) // 1024 // 1024}MB). Max: 20MB.",
        )

    # Save with UUID prefix
    job_id = str(uuid.uuid4())
    safe_name = f"{job_id}{ext}"
    save_path = UPLOADS_DIR / safe_name
    save_path.write_bytes(content)

    # Register job
    jobs[job_id] = {
        "status": "processing",
        "result": None,
        "error": None,
        "file_name": file.filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Run pipeline in thread pool (blocking ops: model inference, file I/O)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        _executor,
        _run_pipeline_sync,
        job_id,
        str(save_path),
    )

    logger.info(f"Job {job_id} started for file: {file.filename}")
    return {"job_id": job_id, "status": "processing"}


def _run_pipeline_sync(job_id: str, file_path: str):
    """Synchronous wrapper for pipeline execution."""
    try:
        pipeline = get_pipeline()
        result = pipeline.run(file_path)
        jobs[job_id]["status"] = "complete"
        jobs[job_id]["result"] = result
        logger.info(f"Job {job_id} completed successfully.")
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
    finally:
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except OSError:
            pass


@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Poll for job completion."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "result": job.get("result"),
        "error": job.get("error"),
    }


@app.get("/history")
async def get_history():
    """Return last 20 analyzed contracts (metadata only)."""
    completed_jobs = [
        {
            "job_id": jid,
            "file_name": job.get("file_name"),
            "created_at": job.get("created_at"),
            "overall_risk_score": job["result"].get("overall_risk_score") if job.get("result") else None,
            "document_type": job["result"].get("document_type") if job.get("result") else None,
            "total_clauses": job["result"].get("total_clauses") if job.get("result") else None,
            "status": job["status"],
        }
        for jid, job in jobs.items()
        if job["status"] == "complete"
    ]
    # Sort by creation time, newest first
    completed_jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    return completed_jobs[:MAX_HISTORY]


@app.get("/health")
async def health_check():
    """Health check for all system components."""
    import importlib

    # Check Ollama
    ollama_ok = False
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        ollama_ok = resp.status_code == 200
    except Exception:
        pass

    # Check ChromaDB
    chromadb_ok = False
    try:
        import chromadb
        client = chromadb.Client()
        chromadb_ok = True
    except Exception:
        pass

    # Check GPU
    gpu_ok = False
    try:
        import torch
        gpu_ok = torch.cuda.is_available()
    except Exception:
        pass

    return {
        "status": "ok",
        "ollama": ollama_ok,
        "chromadb": chromadb_ok,
        "gpu": gpu_ok,
        "active_jobs": sum(1 for j in jobs.values() if j["status"] == "processing"),
        "completed_jobs": sum(1 for j in jobs.values() if j["status"] == "complete"),
    }
