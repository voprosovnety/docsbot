"""Retrieval-augmented answering.

The contract with the user is that the bot answers from their documents or not
at all. Two things enforce it:

1. If similarity search returns nothing within `max_distance`, we never call the
   model — there is nothing to ground an answer in.
2. If it returns something, the system prompt restricts the model to that
   context and requires a source line.
"""

from __future__ import annotations

import logging

from anthropic import AsyncAnthropic

from .config import Config
from .db import Database, Retrieved
from .embeddings import Embedder

logger = logging.getLogger(__name__)

NO_ANSWER = (
    "I don't know — I couldn't find anything about that in your documents.\n\n"
    "Try rephrasing the question, or upload a document that covers it."
)

SYSTEM_PROMPT = """You answer questions strictly from the excerpts supplied in the user message.

Rules:
- Use only the excerpts. Never use outside knowledge, and never guess.
- If the excerpts do not contain the answer, reply with exactly: I don't know
- Do not hedge or pad. Answer the question directly, then stop.
- Keep answers under 120 words unless the question genuinely needs more.
- End every answer with a source line naming the excerpts you used, like:
  Source: handbook.pdf, p. 4
- If several excerpts contradict each other, say so and cite both.

You are talking to someone in a Telegram chat. Write plain sentences: no
markdown headings, no bullet lists unless you are genuinely enumerating items."""


def format_context(chunks: list[Retrieved]) -> str:
    """Render retrieved chunks into a numbered block the model can cite."""
    blocks = []
    for index, chunk in enumerate(chunks, 1):
        location = f"{chunk.title}, p. {chunk.page}" if chunk.page else chunk.title
        blocks.append(f"[Excerpt {index} — {location}]\n{chunk.content}")
    return "\n\n".join(blocks)


def build_user_message(question: str, chunks: list[Retrieved]) -> str:
    return f"{format_context(chunks)}\n\n---\n\nQuestion: {question}"


class Answerer:
    def __init__(self, config: Config, db: Database, embedder: Embedder) -> None:
        self._config = config
        self._db = db
        self._embedder = embedder
        self._client = AsyncAnthropic(api_key=config.anthropic_api_key)

    async def answer(self, *, chat_id: int, question: str) -> str:
        query_vector = await self._embedder.embed_one(question)
        chunks = await self._db.search(
            chat_id=chat_id,
            embedding=query_vector,
            top_k=self._config.top_k,
            max_distance=self._config.max_distance,
        )
        if not chunks:
            logger.info("chat=%s no chunk within distance threshold", chat_id)
            return NO_ANSWER

        response = await self._client.messages.create(
            model=self._config.model,
            # Bounded because a Telegram message caps at 4096 characters and the
            # system prompt already asks for short answers.
            max_tokens=1500,
            output_config={"effort": self._config.effort},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_message(question, chunks)}],
        )

        if response.stop_reason == "refusal":
            logger.warning("chat=%s model refused the request", chat_id)
            return "I can't answer that one. Try a different question about your documents."

        # Thinking blocks come before text blocks, so filter rather than index.
        text = "\n".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        return text or NO_ANSWER
