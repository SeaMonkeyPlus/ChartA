from .supabase_client import supa
from .embeddings import embed_text

def search_similar_chunks(query: str, top_k: int = 3):
    """Supabase RPC 'match_chunks' 호출 (pgvector 검색)"""
    qvec = embed_text(query)
    res = supa.rpc("match_chunks", {"query_vec": qvec, "match_count": top_k}).execute()
    return [r["text"] for r in res.data] if res.data else []
