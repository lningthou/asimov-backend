import os
from contextlib import contextmanager
from typing import List, Dict, Optional
import psycopg2
from psycopg2.pool import SimpleConnectionPool

DBURL = os.getenv("DATABASE_URL")
if not DBURL:
    raise RuntimeError("Set DATABASE_URL")

# Basic connection pool for FastAPI concurrency
_pool = SimpleConnectionPool(minconn=1, maxconn=8, dsn=DBURL)

@contextmanager
def get_conn():
    conn = _pool.getconn()
    try:
        yield conn
    finally:
        _pool.putconn(conn)

def search_videos(
    conn,
    query: str,
    k: int = 5,
    mode: str = "semantic",
    embedding: Optional[str] = None,
) -> List[Dict]:
    """
    Search videos using semantic, keyword, or hybrid mode.
    
    Args:
        conn: Database connection
        query: Search query text
        k: Number of results to return
        mode: "semantic", "keyword", or "hybrid"
        embedding: Pgvector-formatted embedding string (required for semantic/hybrid)
    
    Returns:
        List of result dictionaries with task, description, score, mp4, hdf5
    """
    if mode == "semantic":
        return _search_semantic(conn, embedding, k)
    elif mode == "keyword":
        return _search_keyword(conn, query, k)
    elif mode == "hybrid":
        return _search_hybrid(conn, query, embedding, k)
    else:
        raise ValueError(f"Invalid search mode: {mode}")


def _search_semantic(conn, embedding: str, k: int) -> List[Dict]:
    """Pure vector similarity search."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 
                task,
                s3_uri_mp4,
                s3_uri_h5,
                description,
                embedding <=> %s::vector AS score
            FROM egodex_videos
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (embedding, embedding, k),
        )
        rows = cur.fetchall()
    
    return [
        {
            "task": t,
            "description": desc,
            "score": float(score),
            "mp4": mp4,
            "hdf5": h5,
            "search_type": "semantic"
        }
        for (t, mp4, h5, desc, score) in rows
    ]


def _search_keyword(conn, query: str, k: int) -> List[Dict]:
    """
    Full-text search using PostgreSQL tsvector.
    Uses the description_tsv column and ts_rank for scoring.
    """
    with conn.cursor() as cur:
        # Using websearch_to_tsquery for natural query parsing
        cur.execute(
            """
            SELECT 
                task,
                s3_uri_mp4,
                s3_uri_h5,
                description,
                ts_rank(description_tsv, websearch_to_tsquery('english', %s)) AS score
            FROM egodex_videos
            WHERE description_tsv @@ websearch_to_tsquery('english', %s)
            ORDER BY score DESC
            LIMIT %s
            """,
            (query, query, k),
        )
        rows = cur.fetchall()
    
    return [
        {
            "task": t,
            "description": desc,
            "score": float(score),
            "mp4": mp4,
            "hdf5": h5,
            "search_type": "keyword"
        }
        for (t, mp4, h5, desc, score) in rows
    ]


def _search_hybrid(conn, query: str, embedding: str, k: int) -> List[Dict]:
    """
    Hybrid search using Reciprocal Rank Fusion (RRF).
    Combines semantic (vector) and keyword (full-text) search results.
    """
    # Get more candidates from each search (2x k) for better fusion
    candidate_k = k * 2
    
    with conn.cursor() as cur:
        # RRF query: combines vector similarity and text search
        # Lower RRF score is better (rank-based)
        cur.execute(
            """
            WITH semantic AS (
                SELECT 
                    s3_uri_h5,
                    ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS rank
                FROM egodex_videos
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            ),
            keyword AS (
                SELECT 
                    s3_uri_h5,
                    ROW_NUMBER() OVER (ORDER BY ts_rank(description_tsv, websearch_to_tsquery('english', %s)) DESC) AS rank
                FROM egodex_videos
                WHERE description_tsv @@ websearch_to_tsquery('english', %s)
                LIMIT %s
            ),
            rrf AS (
                SELECT 
                    COALESCE(s.s3_uri_h5, k.s3_uri_h5) AS s3_uri_h5,
                    COALESCE(1.0 / (60 + s.rank), 0.0) + 
                    COALESCE(1.0 / (60 + k.rank), 0.0) AS rrf_score
                FROM semantic s
                FULL OUTER JOIN keyword k ON s.s3_uri_h5 = k.s3_uri_h5
            )
            SELECT 
                v.task,
                v.s3_uri_mp4,
                v.s3_uri_h5,
                v.description,
                rrf.rrf_score
            FROM rrf
            JOIN egodex_videos v ON rrf.s3_uri_h5 = v.s3_uri_h5
            ORDER BY rrf.rrf_score DESC
            LIMIT %s
            """,
            (embedding, embedding, candidate_k, query, query, candidate_k, k),
        )
        rows = cur.fetchall()
        
    return [
        {
            "task": t,
            "description": desc,
            "score": float(score),
            "mp4": mp4,
            "hdf5": h5,
            "search_type": "hybrid"
        }
        for (t, mp4, h5, desc, score) in rows
    ]

def close_pool():
    """Close all pooled connections (called on app shutdown)."""
    try:
        _pool.closeall()
    except Exception:
        pass