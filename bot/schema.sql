CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    chat_id     BIGINT      NOT NULL,
    title       TEXT        NOT NULL,
    source_type TEXT        NOT NULL,
    chunk_count INT         NOT NULL DEFAULT 0,
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_chat_id_idx ON documents (chat_id);

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    document_id BIGINT      NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    chat_id     BIGINT      NOT NULL,
    ordinal     INT         NOT NULL,
    page        INT,
    content     TEXT        NOT NULL,
    embedding   vector(384) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_chat_id_idx ON chunks (chat_id);

-- HNSW index for cosine distance. Built once the table has rows; on an empty
-- table it is created immediately and populated as rows arrive.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
