# docsbot

A Telegram bot that answers questions about **your** documents — and only your
documents. Upload a PDF, DOCX, Markdown or TXT file, ask a question in plain
language, and get an answer with the source it came from.

If the answer isn't in your documents, the bot says **"I don't know"** instead
of inventing one.

```
You  ▸ [uploads handbook.pdf]
Bot  ▸ handbook.pdf is indexed (34 chunks). Ask me anything about it.

You  ▸ How many days of annual leave do I get?
Bot  ▸ Full-time staff get 28 days of paid leave per year, plus public
       holidays. Up to 5 unused days can be carried into the next year;
       anything above that is forfeited on 31 December.

       Source: handbook.pdf, p. 2

You  ▸ What's the company's stock ticker?
Bot  ▸ I don't know — I couldn't find anything about that in your documents.
```

## Why "I don't know" is the feature

Most demo chatbots will confidently answer anything, which makes them unusable
for support, HR, or policy questions where a wrong answer costs real money.

Two mechanisms keep this one honest:

1. **A distance cutoff on retrieval.** Every question is embedded and compared
   against the stored chunks. If nothing scores within `MAX_DISTANCE`, the model
   is never called — there is no context to ground an answer in, so the bot
   says so.
2. **A grounded system prompt.** When context *is* found, the model is
   restricted to those excerpts and required to cite them.

The default cutoff of `0.40` is measured rather than guessed. Against the
included sample documents, questions the documents answer score 0.19–0.33,
and questions they don't score 0.47–0.61:

```
ON-TOPIC — want small distances        OFF-TOPIC — want large distances
  0.190  annual leave?                   0.473  stock ticker symbol?
  0.211  how long to return an item?     0.515  configure nginx?
  0.334  equipment budget?               0.610  capital of Peru?

on-topic worst: 0.334   off-topic best: 0.473   →  MAX_DISTANCE=0.40
```

That gap is specific to this embedding model and these documents. Re-measure it
for yours with [`scripts/calibrate.py`](scripts/calibrate.py):

```bash
export DATABASE_URL=postgresql://docsbot:docsbot@localhost:5432/docsbot
python scripts/calibrate.py my_docs/ --on-topic on.txt --off-topic off.txt
```

## Features

- **Upload documents straight to the bot** — PDF, DOCX, Markdown, TXT.
- **Answers cite their source**, including the page number for PDFs.
- **Honest "I don't know"** when the documents don't cover the question.
- **Per-chat knowledge bases** — two chats using the same deployment never see
  each other's documents.
- **`/docs`** to list what's indexed, **`/reset`** to wipe it and start over.
- **One API key.** Embeddings run locally on ONNX, so Anthropic is the only
  external service.
- **`docker compose up`** and it works.

## Stack

| Layer | Choice |
|---|---|
| Bot framework | aiogram 3 (async) |
| Answers | Claude API (`claude-opus-5`) |
| Embeddings | fastembed / `BAAI/bge-small-en-v1.5`, running locally |
| Vector search | PostgreSQL 16 + pgvector, HNSW index, cosine distance |
| Document parsing | pypdf, python-docx |
| Deployment | Docker Compose |

## Quick start

You need Docker, a bot token from [@BotFather](https://t.me/BotFather), and an
[Anthropic API key](https://console.anthropic.com/settings/keys).

```bash
git clone https://github.com/voprosovnety/docsbot.git
cd docsbot
cp .env.example .env      # fill in BOT_TOKEN and ANTHROPIC_API_KEY
docker compose up --build -d
docker compose logs -f bot
```

Then open your bot in Telegram and send it a file. Three sample documents are
included in [`sample_docs/`](sample_docs/) if you want something to try
immediately:

| Upload this | Then ask |
|---|---|
| `employee_handbook.md` | *How much is the equipment budget?* |
| `refund_policy.md` | *Who pays return shipping if I changed my mind?* |
| `support_playbook.pdf` | *When does the on-call rota run?* — the answer cites page 2 |

Then ask something the documents don't cover, like *"What is the capital of
Peru?"*, to see the "I don't know" path.

The first start downloads the embedding model (~130 MB) into a Docker volume.
Later restarts skip that.

## Running locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Postgres with the pgvector extension
docker compose up -d postgres

cp .env.example .env      # add BOT_TOKEN, ANTHROPIC_API_KEY, and:
                          # DATABASE_URL=postgresql://docsbot:docsbot@localhost:5432/docsbot

python -m bot.main
```

## Configuration

Everything is set through the environment; see [`.env.example`](.env.example)
for the annotated list. The settings worth knowing about:

| Variable | Default | What it does |
|---|---|---|
| `MAX_DISTANCE` | `0.40` | Cosine-distance cutoff. Lower is stricter and produces more "I don't know". Measured — see above. |
| `TOP_K` | `5` | How many excerpts are sent to the model. |
| `CHUNK_SIZE` | `700` | Chunk length in characters. Smaller chunks give more precise citations and a wider on/off-topic gap. |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks, so a sentence on a boundary isn't lost. |
| `CLAUDE_EFFORT` | `low` | Reasoning effort. Raise it if your documents need deeper reasoning. |
| `ADMIN_IDS` | *(empty)* | Restrict `/reset` to specific Telegram user IDs. Empty means anyone can reset their own chat. |

## How it works

```
Upload                                    Question
  │                                          │
  ├─ extract text (pypdf / python-docx)      ├─ embed the question
  ├─ chunk with overlap, keep page numbers   ├─ cosine search in pgvector
  ├─ embed each chunk locally                ├─ drop anything past MAX_DISTANCE
  └─ store in PostgreSQL                     │
                                             ├─ nothing left? → "I don't know"
                                             └─ else → Claude, restricted to
                                                       those excerpts, must cite
```

Chunking splits on paragraph and sentence boundaries where it can, rather than
mid-word, because a chunk cut through the middle of a sentence retrieves badly.
Page numbers ride along with each chunk so the citation can point at a page.

## Project structure

```
bot/
  main.py              # entry point, dependency wiring, long polling
  config.py            # environment configuration
  db.py                # asyncpg pool, pgvector similarity search
  embeddings.py        # local ONNX embeddings
  ingest.py            # text extraction + chunking
  rag.py               # retrieval, prompt assembly, Claude call
  schema.sql           # tables and the HNSW index
  handlers/
    start.py           # /start, /help
    documents.py       # file upload -> indexed
    questions.py       # free text -> answer
    admin.py           # /docs, /reset
scripts/
  calibrate.py         # measure MAX_DISTANCE for your own documents
sample_docs/           # three documents to try the bot with
tests/                 # pytest: chunking, extraction, prompt assembly
```

## Tests

```bash
pytest -q
```

Covered: chunking (boundaries, overlap, termination on pathological input),
text extraction and normalisation including per-page PDF attribution, the
unreadable-file paths (empty, corrupt, wrong format behind a valid extension),
and prompt assembly for the RAG call.

## Limitations

- **Scanned PDFs need OCR first.** The bot extracts embedded text; it does not
  read images. A scanned document produces an "I couldn't extract any text"
  message rather than silently indexing nothing.
- **The embedding model is English-first.** `BAAI/bge-small-en-v1.5` works best
  on English documents. For other languages, set `EMBEDDING_MODEL` to a
  multilingual model — remember to update the `vector(384)` column in
  `schema.sql` if its dimension differs.
- **Telegram caps uploads at 20 MB** for bots using the standard Bot API.

## License

MIT
