import os
from sentence_transformers import SentenceTransformer

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
DEVICE = os.getenv("DEVICE", "cpu")

embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)

def embed_text(text: str):
    return embedder.encode([text], normalize_embeddings=True)[0].tolist()