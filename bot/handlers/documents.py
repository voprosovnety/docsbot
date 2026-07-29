"""Document upload: download, extract, chunk, embed, store."""

from __future__ import annotations

import logging
from html import escape

from aiogram import Bot, F, Router
from aiogram.types import Message

from ..config import Config
from ..db import Database
from ..embeddings import Embedder
from ..ingest import (
    SUPPORTED_EXTENSIONS,
    EmptyDocument,
    UnreadableDocument,
    UnsupportedFormat,
    build_chunks,
    extract_text,
)

logger = logging.getLogger(__name__)
router = Router(name="documents")


@router.message(F.document)
async def handle_document(
    message: Message, bot: Bot, config: Config, db: Database, embedder: Embedder
) -> None:
    document = message.document
    filename = document.file_name or "document"

    if document.file_size and document.file_size > config.max_file_bytes:
        limit_mb = config.max_file_bytes // (1024 * 1024)
        await message.answer(f"That file is over the {limit_mb} MB limit.")
        return

    status = await message.answer(f"Reading <b>{escape(filename)}</b>…")

    try:
        buffer = await bot.download(document)
        if buffer is None:
            raise RuntimeError("Telegram returned no file content")
        data = buffer.read()

        pages = extract_text(filename, data)
        chunks = build_chunks(
            pages, chunk_size=config.chunk_size, overlap=config.chunk_overlap
        )
        if not chunks:
            raise EmptyDocument(filename)

        await status.edit_text(
            f"Indexing <b>{escape(filename)}</b> — {len(chunks)} chunks…"
        )
        embeddings = await embedder.embed([content for content, _ in chunks])
        await db.add_document(
            chat_id=message.chat.id,
            title=filename,
            source_type=filename.rsplit(".", 1)[-1].lower(),
            chunks=chunks,
            embeddings=embeddings,
        )
    except UnsupportedFormat:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        await status.edit_text(f"I can't read that format. Supported: {supported}")
        return
    except EmptyDocument:
        await status.edit_text(
            "I couldn't extract any text from that file. If it's a scanned PDF, "
            "it needs OCR before I can read it."
        )
        return
    except UnreadableDocument:
        await status.edit_text(
            "I couldn't open that file — it may be corrupted, password-protected, "
            "or not really the format its extension claims."
        )
        return
    except Exception:
        logger.exception("chat=%s failed to index %s", message.chat.id, filename)
        await status.edit_text("Something went wrong while indexing that file.")
        return

    await status.edit_text(
        f"<b>{escape(filename)}</b> is indexed ({len(chunks)} chunks). Ask me anything about it."
    )
