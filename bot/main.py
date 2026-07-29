"""Entry point: wire dependencies together and start long polling."""

from __future__ import annotations

import asyncio
import logging
import os

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from .config import Config
from .db import Database
from .embeddings import Embedder
from .handlers import build_router
from .rag import Answerer

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("docsbot")


async def run() -> None:
    config = Config.from_env()

    db = Database(config.database_url)
    await db.connect()

    embedder = Embedder(config.embedding_model)
    # Warm up before polling so the first upload isn't stuck behind a download.
    await embedder.warm_up()

    answerer = Answerer(config, db, embedder)

    bot = Bot(
        token=config.bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dispatcher = Dispatcher()
    # Injected into every handler that declares them as arguments.
    dispatcher["config"] = config
    dispatcher["db"] = db
    dispatcher["embedder"] = embedder
    dispatcher["answerer"] = answerer
    dispatcher.include_router(build_router())

    me = await bot.get_me()
    logger.info("Starting @%s", me.username)

    try:
        await bot.delete_webhook(drop_pending_updates=True)
        await dispatcher.start_polling(bot)
    finally:
        await db.close()
        await bot.session.close()


def main() -> None:
    try:
        asyncio.run(run())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down")


if __name__ == "__main__":
    main()
