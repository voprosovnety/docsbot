from aiogram import Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

router = Router(name="start")

WELCOME = """Hi! I answer questions about your documents — and only your documents.

<b>How to use me</b>
1. Send me a file: PDF, DOCX, Markdown or TXT.
2. Ask a question in plain language.
3. I answer from the file and tell you where the answer came from.

If the answer isn't in your documents, I say "I don't know" instead of making
something up.

<b>Commands</b>
/docs — list what I've indexed
/reset — delete every document in this chat
/help — show this message again"""


@router.message(CommandStart())
@router.message(Command("help"))
async def handle_start(message: Message) -> None:
    await message.answer(WELCOME)
