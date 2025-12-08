from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Literal
from pathlib import Path
import os
from utils.embeddings import embed_text, to_pgvector
from utils.db import get_conn, search_videos, close_pool

app = FastAPI(title="EgoDex Search API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://tryasimov.ai",
        "http://localhost:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
    )

# Directory where .rrd files are stored on the EC2
RRD_DIR = Path(os.environ.get("RRD_DIR", "/data/rrd"))

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/search")
def search(
    q: str,
    k: int = Query(default=5, ge=1, le=100),
    mode: Literal["semantic", "keyword", "hybrid"] = "semantic",
):
    """
    Search EgoDex videos.
    
    Args:
        q: Search query text
        k: Number of results to return (1-100)
        mode: Search mode - "semantic" (embedding), "keyword" (text), or "hybrid" (both)
    """
    text = q.strip()
    if not text:
        raise HTTPException(status_code=400, detail="q must be non-empty")

    # Generate embedding for semantic or hybrid search
    emb_text = None
    if mode in ("semantic", "hybrid"):
        emb = embed_text(text)
        emb_text = to_pgvector(emb)

    with get_conn() as conn:
        results = search_videos(conn, query=text, k=k, mode=mode, embedding=emb_text)
    
    return results

@app.get("/rrd/{filename}")
def stream_rrd(filename: str):
    """
    Stream an .rrd file for Rerun viewer.

    Args:
        filename: Name of the .rrd file (e.g., "test_output_minimal.rrd")
    """
    # Security: only allow .rrd files, no path traversal
    if not filename.endswith(".rrd") or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    filepath = RRD_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Recording not found")

    def iter_file():
        # Stream in 64KB chunks
        chunk_size = 64 * 1024
        with open(filepath, "rb") as f:
            while chunk := f.read(chunk_size):
                yield chunk

    return StreamingResponse(
        iter_file(),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"inline; filename={filename}",
            "Content-Length": str(filepath.stat().st_size),
        }
    )


@app.on_event("shutdown")
def _shutdown():
    close_pool()