"""Free-text messages are treated as questions about the indexed documents."""

from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.enums import ChatAction
from aiogram.types import Message

from ..db import Database
from ..rag import Answerer

logger = logging.getLogger(__name__)
router = Router(name="questions")

TELEGRAM_MESSAGE_LIMIT = 4096


@router.message(F.text & ~F.text.startswith("/"))
async def handle_question(message: Message, db: Database, answerer: Answerer) -> None:
    question = (message.text or "").strip()
    if not question:
        return

    if not await db.list_documents(message.chat.id):
        await message.answer(
            "I don't have any documents yet. Send me a PDF, DOCX, Markdown or TXT "
            "file and I'll answer questions about it."
        )
        return

    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    try:
        answer = await answerer.answer(chat_id=message.chat.id, question=question)
    except Exception:
        logger.exception("chat=%s failed to answer", message.chat.id)
        await message.answer("Something went wrong on my side. Try again in a moment.")
        return

    # Long answers are rare because the system prompt asks for brevity, but a
    # single over-limit reply would otherwise fail silently.
    #
    # parse_mode=None because the answer quotes the user's documents verbatim,
    # and a stray "<" would make Telegram reject the whole message.
    for start in range(0, len(answer), TELEGRAM_MESSAGE_LIMIT):
        await message.answer(
            answer[start : start + TELEGRAM_MESSAGE_LIMIT], parse_mode=None
        )
