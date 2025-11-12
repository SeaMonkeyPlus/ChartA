CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_chunks (
  id BIGSERIAL PRIMARY KEY,
  text TEXT,
  embedding vector(1024)
);

CREATE INDEX IF NOT EXISTS idx_chunks_vec_hnsw
  ON rag_chunks USING hnsw (embedding vector_cosine_ops);

CREATE OR REPLACE FUNCTION match_chunks(query_vec vector(1024), match_count int)
RETURNS TABLE (id bigint, text text, similarity float4)
LANGUAGE sql STABLE AS $$
  SELECT id, text, 1 - (embedding <-> query_vec) AS similarity
  FROM rag_chunks
  ORDER BY embedding <-> query_vec
  LIMIT match_count;
$$;
