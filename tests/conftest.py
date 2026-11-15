from __future__ import annotations

import shutil
from pathlib import Path

import pymupdf
import pytest

from app.config import Settings

STORY = """Chapter 1

It was a dark night and the rain hammered the windows of the old inn. Tom poured a drink by the fire. He looked tired.

"You're late," said Mary, folding her arms.

"The storm," Tom replied. "The road was flooded."

"Of course it was."

She whispered, "I was worried."

* * *

Morning came. Birds sang in the trees and the sun was bright.

"Let's go!" Tom shouted happily.

Chapter 2

The wind picked up on the shore, and the waves rolled in.
"""

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        max_chars=100_000,
        secure_cookies=False,
        retention_days=7,
        llm_model="test-model",
        anthropic_api_key=None,
        openai_api_key=None,
    )


def make_pdf(path: Path, chapters: list[tuple[str, list[str]]], header: str = "The Test Book") -> Path:
    """Build a small PDF: running header, page number footer, one chapter per page."""
    doc = pymupdf.open()
    for i, (title, paragraphs) in enumerate(chapters, start=1):
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(72, 30, 540, 50), header, fontsize=9)
        page.insert_textbox(pymupdf.Rect(72, 70, 540, 100), title, fontsize=16)
        y = 110
        for para in paragraphs:
            page.insert_textbox(pymupdf.Rect(72, y, 540, y + 90), para, fontsize=11)
            y += 100
        page.insert_textbox(pymupdf.Rect(280, 800, 320, 820), str(i), fontsize=9)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def story_pdf(tmp_path: Path) -> Path:
    return make_pdf(
        tmp_path / "story.pdf",
        [
            (
                "Chapter 1",
                [
                    "It was a dark night and the rain hammered the windows of the old inn. Tom poured a drink "
                    "by the fire. He looked tired and the exhaust-\nion showed.",
                    "“You're late,” said Mary, folding her arms.",
                    "“The storm,” Tom replied. “The road was flooded.”",
                ],
            ),
            ("Chapter 2", ["Morning came. Birds sang in the trees.", "“Let's go!” Tom shouted happily."]),
            ("Chapter 3", ["The wind picked up on the shore and the waves rolled in."]),
            ("Chapter 4", ["Silence fell over the empty house."]),
        ],
    )
