import pathlib

import pytest

from bot.ingest import (
    EmptyDocument,
    UnreadableDocument,
    UnsupportedFormat,
    build_chunks,
    chunk_text,
    extract_text,
)

SAMPLE_PDF = pathlib.Path(__file__).parent.parent / "sample_docs" / "support_playbook.pdf"


class TestChunkText:
    def test_short_text_stays_one_chunk(self):
        chunks = chunk_text("Short and sweet.", chunk_size=100, overlap=20)
        assert chunks == [("Short and sweet.", None)]

    def test_empty_text_produces_nothing(self):
        assert chunk_text("   ", chunk_size=100, overlap=20) == []

    def test_long_text_is_split(self):
        text = "word " * 500
        chunks = chunk_text(text, chunk_size=200, overlap=50)
        assert len(chunks) > 1
        assert all(len(content) <= 200 for content, _ in chunks)

    def test_chunks_overlap_so_boundary_sentences_survive(self):
        text = "".join(f"Sentence number {i}. " for i in range(100))
        chunks = chunk_text(text, chunk_size=300, overlap=100)
        # Consecutive chunks should share text, otherwise a sentence split
        # across the boundary would be unretrievable from either side.
        first_tail = chunks[0][0][-50:]
        assert any(part in chunks[1][0] for part in first_tail.split() if len(part) > 3)

    def test_prefers_paragraph_boundary(self):
        body = "a" * 240
        # The tail must push the total past chunk_size, otherwise the whole
        # text fits in one chunk and no boundary decision is made.
        text = f"{body}\n\n{'Second paragraph. ' * 12}"
        chunks = chunk_text(text, chunk_size=300, overlap=50)
        assert len(chunks) > 1
        assert chunks[0][0] == body

    def test_page_number_is_attached(self):
        chunks = chunk_text("Some content.", chunk_size=100, overlap=10, page=7)
        assert chunks == [("Some content.", 7)]

    def test_overlap_must_be_smaller_than_chunk(self):
        with pytest.raises(ValueError):
            chunk_text("text", chunk_size=100, overlap=100)

    def test_always_terminates_on_pathological_overlap(self):
        # overlap == chunk_size - 1 is the worst legal case; it must still finish.
        chunks = chunk_text("x" * 1000, chunk_size=100, overlap=99)
        assert chunks


class TestExtractText:
    def test_markdown_is_read_as_plain_text(self):
        pages = extract_text("notes.md", b"# Title\n\nBody text.")
        assert pages == [("# Title\n\nBody text.", None)]

    def test_crlf_is_normalised(self):
        pages = extract_text("notes.txt", b"line one\r\nline two")
        assert pages == [("line one\nline two", None)]

    def test_runs_of_blank_lines_collapse(self):
        pages = extract_text("notes.txt", b"a\n\n\n\n\nb")
        assert pages == [("a\n\nb", None)]

    def test_unsupported_extension_is_rejected(self):
        with pytest.raises(UnsupportedFormat):
            extract_text("archive.zip", b"whatever")

    def test_whitespace_only_file_is_rejected(self):
        with pytest.raises(EmptyDocument):
            extract_text("blank.txt", b"   \n\n  ")

    def test_invalid_utf8_does_not_crash(self):
        pages = extract_text("weird.txt", b"caf\xe9 open")
        assert "caf" in pages[0][0]


class TestPdfExtraction:
    def test_each_page_is_returned_with_its_number(self):
        pages = extract_text(SAMPLE_PDF.name, SAMPLE_PDF.read_bytes())
        assert [page for _, page in pages] == [1, 2]

    def test_page_two_content_is_not_attributed_to_page_one(self):
        pages = extract_text(SAMPLE_PDF.name, SAMPLE_PDF.read_bytes())
        by_page = {page: text for text, page in pages}
        assert "Escalation levels" in by_page[1]
        assert "Out-of-hours cover" in by_page[2]
        assert "Out-of-hours cover" not in by_page[1]

    def test_empty_pdf_is_reported_as_unreadable(self):
        with pytest.raises(UnreadableDocument):
            extract_text("broken.pdf", b"")

    def test_garbage_pdf_is_reported_as_unreadable(self):
        with pytest.raises(UnreadableDocument):
            extract_text("broken.pdf", b"this is definitely not a pdf")

    def test_garbage_docx_is_reported_as_unreadable(self):
        with pytest.raises(UnreadableDocument):
            extract_text("broken.docx", b"not a zip archive either")


class TestBuildChunks:
    def test_page_numbers_survive_chunking(self):
        pages = [("a" * 500, 1), ("b" * 500, 2)]
        chunks = build_chunks(pages, chunk_size=200, overlap=50)
        assert {page for _, page in chunks} == {1, 2}

    def test_empty_page_contributes_nothing(self):
        chunks = build_chunks([("", 1), ("real text", 2)], chunk_size=200, overlap=50)
        assert chunks == [("real text", 2)]
