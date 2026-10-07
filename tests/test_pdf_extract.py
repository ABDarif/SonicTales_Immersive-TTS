from pathlib import Path

import pymupdf
import pytest

from app.pdf_extract import PdfError, book_from_text, extract_book, load_book

from .conftest import STORY, make_pdf


def test_extract_splits_chapters_and_cleans_text(story_pdf: Path):
    book = extract_book(story_pdf)
    assert [c.title for c in book.chapters] == ["Chapter 1", "Chapter 2", "Chapter 3", "Chapter 4"]
    text = " ".join(book.chapters[0].paragraphs)
    assert "exhaustion showed" in text, "hyphenation across a line break should be repaired"
    assert "\n" not in text


def test_running_headers_and_page_numbers_removed(story_pdf: Path):
    book = extract_book(story_pdf)
    everything = [p for c in book.chapters for p in c.paragraphs]
    assert not any("The Test Book" in p for p in everything)
    assert not any(p.strip().isdigit() for p in everything)


def test_title_defaults_to_filename(story_pdf: Path):
    assert extract_book(story_pdf).title == "story"
    assert extract_book(story_pdf, title="My Book").title == "My Book"


def test_pdf_without_headings_is_one_chapter(tmp_path: Path):
    pdf = make_pdf(tmp_path / "plain.pdf", [("", [f"Prose number {i} is here."]) for i in range(4)])
    book = extract_book(pdf)
    assert len(book.chapters) == 1
    assert len(book.chapters[0].paragraphs) == 4  # headers/page numbers removed, prose kept


def test_scanned_pdf_gives_clear_error(tmp_path: Path):
    doc = pymupdf.open()
    doc.new_page()
    path = tmp_path / "blank.pdf"
    doc.save(path)
    with pytest.raises(PdfError, match="scanned"):
        extract_book(path)


def test_garbage_file_is_rejected(tmp_path: Path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"this is not a pdf")
    with pytest.raises(PdfError):
        extract_book(bad)


def test_text_input_uses_same_chapter_logic():
    book = book_from_text(STORY, "Story")
    assert [c.title for c in book.chapters] == ["Chapter 1", "Chapter 2"]
    assert "<<SCENE_BREAK>>" in book.chapters[0].paragraphs


def test_text_without_headings():
    book = book_from_text("One paragraph.\n\nTwo paragraph.")
    assert len(book.chapters) == 1 and len(book.chapters[0].paragraphs) == 2


def test_load_book_rejects_unknown_type(tmp_path: Path):
    f = tmp_path / "x.docx"
    f.write_text("hi")
    with pytest.raises(PdfError, match="unsupported"):
        load_book(f)
