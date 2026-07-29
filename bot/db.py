"""PostgreSQL + pgvector storage for the knowledge base.

Every row carries a `chat_id`, so two Telegram chats using the same deployment
never see each other's documents.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import asyncpg
from pgvector.asyncpg import register_vector

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


@dataclass(frozen=True)
class Retrieved:
    """One chunk pulled back by similarity search."""

    content: str
    title: str
    page: int | None
    distance: float


@dataclass(frozen=True)
class DocumentInfo:
    id: int
    title: str
    chunk_count: int


class Database:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Database.connect() has not been awaited yet")
        return self._pool

    async def connect(self) -> None:
        # Order matters. register_vector introspects the `vector` type, which
        # only exists after CREATE EXTENSION has run, so the schema is applied
        # on a plain connection first. Doing it as the pool's init callback
        # fails on a fresh database with "unknown type: public.vector".
        await self._apply_schema()
        self._pool = await asyncpg.create_pool(
            self._dsn, min_size=1, max_size=10, init=register_vector
        )
        logger.info("Database ready")

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _apply_schema(self) -> None:
        conn = await asyncpg.connect(self._dsn)
        try:
            await conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
        finally:
            await conn.close()

    async def add_document(
        self,
        *,
        chat_id: int,
        title: str,
        source_type: str,
        chunks: list[tuple[str, int | None]],
        embeddings: list[list[float]],
    ) -> int:
        """Store a document and its embedded chunks in a single transaction."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")

        async with self.pool.acquire() as conn, conn.transaction():
            document_id: int = await conn.fetchval(
                """
                INSERT INTO documents (chat_id, title, source_type, chunk_count)
                VALUES ($1, $2, $3, $4)
                RETURNING id
                """,
                chat_id,
                title,
                source_type,
                len(chunks),
            )
            await conn.executemany(
                """
                INSERT INTO chunks
                    (document_id, chat_id, ordinal, page, content, embedding)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                [
                    (document_id, chat_id, ordinal, page, content, embedding)
                    for ordinal, ((content, page), embedding) in enumerate(
                        zip(chunks, embeddings)
                    )
                ],
            )
        return document_id

    async def search(
        self, *, chat_id: int, embedding: list[float], top_k: int, max_distance: float
    ) -> list[Retrieved]:
        """Return the closest chunks, dropping anything beyond `max_distance`.

        The distance filter is what lets the bot answer "I don't know": if a
        question has no nearby chunk, retrieval comes back empty and we never
        reach the model.
        """
        rows = await self.pool.fetch(
            """
            SELECT c.content,
                   d.title,
                   c.page,
                   c.embedding <=> $2 AS distance
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.chat_id = $1
              AND c.embedding <=> $2 < $4
            ORDER BY distance
            LIMIT $3
            """,
            chat_id,
            embedding,
            top_k,
            max_distance,
        )
        return [
            Retrieved(
                content=row["content"],
                title=row["title"],
                page=row["page"],
                distance=row["distance"],
            )
            for row in rows
        ]

    async def list_documents(self, chat_id: int) -> list[DocumentInfo]:
        rows = await self.pool.fetch(
            """
            SELECT id, title, chunk_count
            FROM documents
            WHERE chat_id = $1
            ORDER BY uploaded_at
            """,
            chat_id,
        )
        return [
            DocumentInfo(id=row["id"], title=row["title"], chunk_count=row["chunk_count"])
            for row in rows
        ]

    async def clear(self, chat_id: int) -> int:
        """Drop every document for a chat. Chunks go with them via ON DELETE CASCADE."""
        result = await self.pool.execute(
            "DELETE FROM documents WHERE chat_id = $1", chat_id
        )
        # asyncpg returns a command tag like "DELETE 3".
        return int(result.rsplit(" ", 1)[-1])
