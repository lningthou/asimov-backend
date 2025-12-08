from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Literal
from pathlib import Path
import os
import boto3
from botocore.exceptions import ClientError
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

# S3 configuration
S3_BUCKET = os.environ.get("S3_BUCKET", "rrd-files-skild")
S3_PREFIX = os.environ.get("S3_PREFIX", "")  # no prefix - files stored at root

# Initialize S3 client
s3_client = boto3.client("s3")

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
    Stream an .rrd file from S3 for Rerun viewer.

    Args:
        filename: Name of the .rrd file (e.g., "episode_001.rrd")
    """
    # Security: only allow .rrd files, no path traversal
    if not filename.endswith(".rrd") or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    s3_key = f"{S3_PREFIX}{filename}"

    try:
        # Get object metadata for Content-Length
        head = s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
        content_length = head["ContentLength"]
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            raise HTTPException(status_code=404, detail=f"Recording not found: {filename}")
        raise HTTPException(status_code=500, detail=f"S3 error: {str(e)}")

    def iter_s3_object():
        # Stream from S3 in 64KB chunks
        chunk_size = 64 * 1024
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        for chunk in response["Body"].iter_chunks(chunk_size=chunk_size):
            yield chunk

    return StreamingResponse(
        iter_s3_object(),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"inline; filename={filename}",
            "Content-Length": str(content_length),
        }
    )


@app.on_event("shutdown")
def _shutdown():
    close_pool()