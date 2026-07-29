"""Local text embeddings.

Embeddings run on-device via fastembed (ONNX runtime), so the only API key the
bot needs is the Anthropic one. The model is downloaded on first use and cached
in the `models/` volume.
"""

from __future__ import annotations

import asyncio
import logging

from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

# BAAI/bge-small-en-v1.5 produces 384-dimensional vectors. Changing the model
# means changing the `vector(384)` column in schema.sql to match.
EMBEDDING_DIM = 384


class Embedder:
    def __init__(self, model_name: str, cache_dir: str = "models") -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model: TextEmbedding | None = None

    async def warm_up(self) -> None:
        """Download and load the model before the bot starts serving."""
        await asyncio.to_thread(self._load)

    def _load(self) -> TextEmbedding:
        if self._model is None:
            logger.info("Loading embedding model %s", self._model_name)
            self._model = TextEmbedding(
                model_name=self._model_name, cache_dir=self._cache_dir
            )
            logger.info("Embedding model ready")
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        return [vector.tolist() for vector in model.embed(texts)]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Runs in a thread so the event loop keeps moving."""
        if not texts:
            return []
        return await asyncio.to_thread(self._encode, texts)

    async def embed_one(self, text: str) -> list[float]:
        vectors = await self.embed([text])
        return vectors[0]
