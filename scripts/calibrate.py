"""Measure the retrieval threshold for your own documents.

MAX_DISTANCE decides when the bot answers and when it says "I don't know", and
the right value depends on your embedding model and your documents. This script
indexes a folder, scores questions you know are answerable against questions you
know are not, and reports whether a clean cutoff exists between them.

Usage:
    python scripts/calibrate.py sample_docs/

    # with your own questions, one per line
    python scripts/calibrate.py my_docs/ --on-topic on.txt --off-topic off.txt

Requires DATABASE_URL to point at a PostgreSQL with pgvector
(`docker compose up -d postgres` is enough).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from bot.db import Database  # noqa: E402
from bot.embeddings import Embedder  # noqa: E402
from bot.ingest import SUPPORTED_EXTENSIONS, build_chunks, extract_text  # noqa: E402

# A scratch chat id so calibration never touches a real chat's documents.
CALIBRATION_CHAT = -1


DEFAULT_ON_TOPIC = [
    "How many days of annual leave do I get?",
    "What is the equipment budget for new starters?",
    "How long do I have to return an item?",
    "Who pays return shipping if I changed my mind?",
    "What are the core working hours?",
    "How long does a refund take?",
    "What is the notice period for managers?",
    "Can I exchange an item instead of returning it?",
]

DEFAULT_OFF_TOPIC = [
    "What is the company's stock ticker symbol?",
    "How do I bake sourdough bread?",
    "What is the capital of Peru?",
    "Who won the 2022 world cup?",
    "How do I configure nginx as a reverse proxy?",
]


def read_questions(path: str | None, fallback: list[str]) -> list[str]:
    if path is None:
        return fallback
    lines = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents", help="folder of documents to index")
    parser.add_argument("--on-topic", help="file of questions the documents answer")
    parser.add_argument("--off-topic", help="file of questions they do not answer")
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    args = parser.parse_args()

    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        print("Set DATABASE_URL first, e.g.")
        print("  export DATABASE_URL=postgresql://docsbot:docsbot@localhost:5432/docsbot")
        return 2

    on_topic = read_questions(args.on_topic, DEFAULT_ON_TOPIC)
    off_topic = read_questions(args.off_topic, DEFAULT_OFF_TOPIC)

    db = Database(dsn)
    await db.connect()
    embedder = Embedder(os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    await embedder.warm_up()

    await db.clear(CALIBRATION_CHAT)
    folder = pathlib.Path(args.documents)
    indexed = 0
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        pages = extract_text(path.name, path.read_bytes())
        chunks = build_chunks(
            pages, chunk_size=args.chunk_size, overlap=args.chunk_overlap
        )
        embeddings = await embedder.embed([content for content, _ in chunks])
        await db.add_document(
            chat_id=CALIBRATION_CHAT,
            title=path.name,
            source_type=path.suffix.lstrip("."),
            chunks=chunks,
            embeddings=embeddings,
        )
        indexed += len(chunks)

    if not indexed:
        print(f"No supported documents in {folder}")
        await db.close()
        return 1
    print(f"Indexed {indexed} chunks from {folder}\n")

    async def nearest(question: str) -> float | None:
        vector = await embedder.embed_one(question)
        # max_distance=2.0 disables the cutoff so we can see the raw score.
        hits = await db.search(
            chat_id=CALIBRATION_CHAT, embedding=vector, top_k=1, max_distance=2.0
        )
        return hits[0].distance if hits else None

    print("ON-TOPIC — want small distances")
    on_scores = []
    for question in on_topic:
        distance = await nearest(question)
        on_scores.append(distance)
        print(f"  {distance:.3f}  {question}")

    print("\nOFF-TOPIC — want large distances")
    off_scores = []
    for question in off_topic:
        distance = await nearest(question)
        off_scores.append(distance)
        print(f"  {distance:.3f}  {question}")

    worst_on, best_off = max(on_scores), min(off_scores)
    print(f"\non-topic worst:  {worst_on:.3f}")
    print(f"off-topic best:  {best_off:.3f}")

    if best_off > worst_on:
        suggestion = (worst_on + best_off) / 2
        print(f"gap:             {best_off - worst_on:.3f}")
        print(f"\nAny cutoff in ({worst_on:.3f}, {best_off:.3f}) separates the two sets.")
        print(f"Suggested: MAX_DISTANCE={suggestion:.2f}")
    else:
        print("\nThe two sets overlap — no cutoff separates them perfectly.")
        print("Try smaller chunks, a stronger embedding model, or accept that")
        print("some off-topic questions will reach the model (it can still")
        print("answer 'I don't know' from the excerpts it is given).")

    await db.clear(CALIBRATION_CHAT)
    await db.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
