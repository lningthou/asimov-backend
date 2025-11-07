from functools import lru_cache
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Loaded embedding model once and reuse across requests"""
    return SentenceTransformer(MODEL_NAME)

def embed_text(text: str) -> list[float]:
    """Return embedding as a Python list[float]."""
    model = _load_model()
    return model.encode(text).tolist()

def to_pgvector(vec: list[float]) -> str:
    """Format for pgvector text input: [v1,v2,...]."""
    return "[" + ",".join(f"{x:.7f}" for x in vec) + "]"

