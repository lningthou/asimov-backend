from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal
import os
from dotenv import load_dotenv

# Load environment variables from .env file (for DATABASE_URL)
load_dotenv()

# Auth configuration
EXPLORE_PASSWORDS = ["skild", "pi-samples", "persona", "figure", "robot-samples", "openai-samples", "figure-samples", "maven-samples"]
AUTH_TOKEN = "xK9mP2vL8nQ4wR7jT1yB5cF3hD6gA0sE"

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
    allow_credentials=True,  # Required for cookies
)


# Auth models
class AuthRequest(BaseModel):
    password: str


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/auth")
def authenticate(auth: AuthRequest):
    """Verify password and return auth token."""
    if auth.password in EXPLORE_PASSWORDS:
        return {"authenticated": True, "token": AUTH_TOKEN}
    else:
        raise HTTPException(status_code=401, detail="Invalid password")


@app.post("/api/auth/verify")
def verify_token(request: Request):
    """Verify if token is valid."""
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:]
        if token == AUTH_TOKEN:
            return {"authenticated": True}
    return {"authenticated": False}


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


@app.on_event("shutdown")
def _shutdown():
    if DB_AVAILABLE:
        close_pool()
