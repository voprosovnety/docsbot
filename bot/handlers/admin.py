"""Knowledge-base management: list what's indexed, wipe it and start over."""

from __future__ import annotations

from html import escape

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from ..config import Config
from ..db import Database

router = Router(name="admin")


@router.message(Command("docs"))
async def handle_docs(message: Message, db: Database) -> None:
    documents = await db.list_documents(message.chat.id)
    if not documents:
        await message.answer("Nothing indexed yet. Send me a file to get started.")
        return

    lines = [f"<b>Indexed documents ({len(documents)})</b>"]
    total_chunks = 0
    for document in documents:
        total_chunks += document.chunk_count
        lines.append(f"• {escape(document.title)} — {document.chunk_count} chunks")
    lines.append(f"\n{total_chunks} chunks searchable in this chat.")
    await message.answer("\n".join(lines))


@router.message(Command("reset"))
async def handle_reset(message: Message, db: Database, config: Config) -> None:
    """Wipe this chat's knowledge base.

    When ADMIN_IDS is set, only those users may reset; leaving it empty makes
    the bot self-serve, which is what the public demo wants.
    """
    if config.admin_ids and message.from_user.id not in config.admin_ids:
        await message.answer("Only an admin can reset the knowledge base here.")
        return

    deleted = await db.clear(message.chat.id)
    if deleted:
        await message.answer(
            f"Deleted {deleted} document(s). Send me a new file whenever you're ready."
        )
    else:
        await message.answer("There was nothing to delete.")
