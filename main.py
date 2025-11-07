from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from utils.embeddings import embed_text, to_pgvector
from utils.db import get_conn, search_by_vector, close_pool

app = FastAPI(title="EgoDex Search API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://tryasimov.ai",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],   
    allow_headers=["*"],
    )
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/search")
def search(q: str, k: int = 5):
    text = q.strip()
    if not text:
        raise HTTPException(status_code=400, detail="q must be non-empty")

    emb = embed_text(text)
    vtxt = to_pgvector(emb)

    with get_conn() as conn:
        results = search_by_vector(conn, vtxt, k)

    return results

@app.on_event("shutdown")
def _shutdown():
    close_pool()
