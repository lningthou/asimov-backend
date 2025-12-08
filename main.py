from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Literal
from pathlib import Path
import os
import httpx
from dotenv import load_dotenv

# Load environment variables from .env file (for DATABASE_URL)
load_dotenv()

# Try to import DB utils (optional - only needed for search)
try:
    from utils.embeddings import embed_text, to_pgvector
    from utils.db import get_conn, search_videos, close_pool
    DB_AVAILABLE = True
except Exception as e:
    print(f"Warning: Database not configured: {e}")
    DB_AVAILABLE = False

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

# S3 public URL configuration
S3_BUCKET = "rrd-files-skild"
S3_REGION = "us-east-1"
S3_BASE_URL = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com"

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
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not configured")

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
async def get_rrd(filename: str):
    """
    Stream an .rrd file from S3.

    Args:
        filename: Name of the .rrd file (e.g., "episode_001.rrd")
    """
    # Security: only allow .rrd files, no path traversal
    if not filename.endswith(".rrd") or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    s3_url = f"{S3_BASE_URL}/{filename}"

    async def stream_from_s3():
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", s3_url) as response:
                if response.status_code == 404:
                    raise HTTPException(status_code=404, detail=f"Recording not found: {filename}")
                async for chunk in response.aiter_bytes(chunk_size=65536):
                    yield chunk

    return StreamingResponse(
        stream_from_s3(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"inline; filename={filename}"}
    )


@app.on_event("shutdown")
def _shutdown():
    if DB_AVAILABLE:
        close_pool()