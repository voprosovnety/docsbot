from bot.db import Retrieved
from bot.rag import build_user_message, format_context


def chunk(content: str, title: str = "handbook.pdf", page: int | None = None):
    return Retrieved(content=content, title=title, page=page, distance=0.1)


class TestFormatContext:
    def test_excerpts_are_numbered(self):
        rendered = format_context([chunk("first"), chunk("second")])
        assert "[Excerpt 1" in rendered
        assert "[Excerpt 2" in rendered

    def test_page_number_is_included_when_known(self):
        rendered = format_context([chunk("text", page=4)])
        assert "handbook.pdf, p. 4" in rendered

    def test_page_is_omitted_for_flat_documents(self):
        rendered = format_context([chunk("text", title="notes.md", page=None)])
        assert "notes.md" in rendered
        assert "p. None" not in rendered

    def test_content_is_preserved_verbatim(self):
        rendered = format_context([chunk("Annual leave is 28 days.")])
        assert "Annual leave is 28 days." in rendered


class TestBuildUserMessage:
    def test_question_follows_the_context(self):
        message = build_user_message("How many days?", [chunk("28 days")])
        assert message.index("28 days") < message.index("How many days?")

    def test_question_is_labelled(self):
        message = build_user_message("How many days?", [chunk("28 days")])
        assert "Question: How many days?" in message
