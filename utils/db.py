import os
from contextlib import contextmanager
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

def search_by_vector(conn, vec_text: str, k: int = 5):
    """Query pgvector index and return rows as dicts."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT task, s3_uri_mp4, s3_uri_h5, caption, emb <=> %s::vector AS score
            FROM egodex_index
            ORDER BY score
            LIMIT %s
            """,
            (vec_text, k),
        )
        rows = cur.fetchall()
    return [
        {"task": t, "caption": cap, "score": float(score)}, "mp4": mp4, "hdf5": h5}
        for (t, mp4, h5, cap, score) in rows
    ]
