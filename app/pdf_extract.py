"""Stage 1: PDF (or plain text) -> clean paragraphs grouped into chapters."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pymupdf

from app.models import Book, RawChapter

CHAPTER_RE = re.compile(
    r"^\s*(?:chapter|part|book)\s+(?:[0-9]+|[ivxlcdm]+|[a-z\-]+)\b[\s.:\-—]*.{0,80}$"
    r"|^\s*(?:prologue|epilogue|preface|introduction|foreword|afterword)\s*$",
    re.IGNORECASE,
)
SCENE_BREAK_RE = re.compile(r"^\s*(?:\*\s*){3,}$|^\s*[-–—]{3,}\s*$|^\s*#\s*$")
SCENE_MARK = "<<SCENE_BREAK>>"
PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s+)?[0-9ivxlc]{1,5}\s*$", re.IGNORECASE)


class PdfError(ValueError):
    pass


def _norm_for_repeat(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())


def _tidy(text: str) -> str:
    """Join hard-wrapped lines, repair hyphenation and normalise whitespace."""
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # dehyphenate across line breaks
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"[ \t ]+", " ", text).strip()


def _page_blocks(page: pymupdf.Page) -> list[str]:
    blocks = page.get_text("blocks", sort=True)
    return [b[4] for b in blocks if b[6] == 0 and b[4].strip()]  # type 0 = text


def extract_pages(path: Path) -> tuple[list[list[str]], pymupdf.Document]:
    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        raise PdfError(f"cannot open PDF: {exc}") from exc
    if doc.needs_pass:
        raise PdfError("PDF is password protected")
    pages = [_page_blocks(p) for p in doc]
    if not any(pages):
        raise PdfError("no extractable text found; this looks like a scanned PDF (OCR is not supported yet)")
    return pages, doc


def _strip_running_headers(pages: list[list[str]]) -> list[list[str]]:
    """Remove page numbers and headers/footers that repeat across many pages."""
    if len(pages) >= 4:
        counts: Counter[str] = Counter()
        for blocks in pages:
            edge = {_norm_for_repeat(b) for b in (blocks[:1] + blocks[-1:])}
            counts.update(edge)
        threshold = max(3, int(len(pages) * 0.4))
        repeated = {k for k, v in counts.items() if v >= threshold and len(k) < 100}
    else:
        repeated = set()
    cleaned = []
    for blocks in pages:
        keep = []
        for i, b in enumerate(blocks):
            on_edge = i == 0 or i == len(blocks) - 1
            flat = b.strip()
            if on_edge and (PAGE_NUMBER_RE.match(flat) or _norm_for_repeat(flat) in repeated):
                continue
            keep.append(b)
        cleaned.append(keep)
    return cleaned


def _paragraphs_from_pages(pages: list[list[str]]) -> list[tuple[int, str]]:
    """Flatten to (page_index, paragraph) and merge paragraphs split across pages."""
    out: list[tuple[int, str]] = []
    carry_open = False
    for page_idx, blocks in enumerate(pages):
        for b in blocks:
            text = _tidy(b)
            if not text:
                continue
            if SCENE_BREAK_RE.match(text):
                out.append((page_idx, SCENE_MARK))
                carry_open = False
                continue
            if carry_open and out and out[-1][1] != SCENE_MARK and text[:1].islower():
                out[-1] = (out[-1][0], out[-1][1] + " " + text)
            else:
                out.append((page_idx, text))
            is_heading = len(text) <= 90 and bool(CHAPTER_RE.match(text))
            carry_open = not is_heading and not re.search(r"[.!?\"”’'»)\]:]$", text)
    return out


def split_chapters(paragraphs: list[tuple[int, str]], toc: list[tuple[int, str]] | None = None) -> list[RawChapter]:
    """Chapters from heading regexes; falls back to the PDF outline, then a single chapter."""
    chapters: list[RawChapter] = []
    current: RawChapter | None = None
    for _, para in paragraphs:
        if para != SCENE_MARK and len(para) <= 90 and CHAPTER_RE.match(para):
            current = RawChapter(title=para.strip(), paragraphs=[])
            chapters.append(current)
        elif current is not None:
            current.paragraphs.append(para)
        else:
            current = RawChapter(title="Beginning", paragraphs=[para])
            chapters.append(current)
    heading_chapters = [c for c in chapters if c.paragraphs]
    if len(heading_chapters) >= 2 or not toc:
        return heading_chapters or [RawChapter(title="Full text", paragraphs=[])]

    # Outline fallback: chapter boundaries at the first paragraph of each outlined page.
    starts: list[tuple[int, str]] = []
    seen_pages: set[int] = set()
    for page_idx, title in toc:
        if page_idx not in seen_pages:
            seen_pages.add(page_idx)
            starts.append((page_idx, title))
    starts.sort()
    result: list[RawChapter] = []
    for i, (page_idx, title) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else 10**9
        paras = [p for pg, p in paragraphs if page_idx <= pg < end]
        if paras:
            result.append(RawChapter(title=title, paragraphs=paras))
    first_start = starts[0][0]
    front = [p for pg, p in paragraphs if pg < first_start]
    if front:
        result.insert(0, RawChapter(title="Beginning", paragraphs=front))
    return result or heading_chapters


def extract_book(path: Path, title: str | None = None) -> Book:
    pages, doc = extract_pages(path)
    meta_title = (doc.metadata or {}).get("title") or ""
    toc = [(max(0, pg - 1), t.strip()) for lvl, t, pg in doc.get_toc() if lvl == 1 and t.strip()]
    doc.close()
    paragraphs = _paragraphs_from_pages(_strip_running_headers(pages))
    chapters = split_chapters(paragraphs, toc)
    return Book(title=title or meta_title.strip() or path.stem.replace("_", " "), chapters=chapters)


def book_from_text(text: str, title: str = "Pasted text") -> Book:
    """Plain-text input (the web textarea / .txt files) shares the same chapter logic."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs: list[tuple[int, str]] = []
    for block in re.split(r"\n\s*\n", text):
        tidy = _tidy(block)
        if not tidy:
            continue
        paragraphs.append((0, SCENE_MARK if SCENE_BREAK_RE.match(tidy) else tidy))
    return Book(title=title, chapters=split_chapters(paragraphs))


def load_book(path: Path, title: str | None = None) -> Book:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_book(path, title)
    if suffix in {".txt", ".md"}:
        return book_from_text(path.read_text(encoding="utf-8", errors="replace"), title or path.stem)
    raise PdfError(f"unsupported input type {suffix!r}; use .pdf or .txt")
