"""Data model shared by every pipeline stage: book -> script -> audio."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Emotion = Literal["neutral", "happy", "sad", "angry", "afraid", "whisper", "excited"]
Gender = Literal["male", "female", "unknown"]

NARRATOR = "Narrator"


class RawChapter(BaseModel):
    title: str
    paragraphs: list[str]


class Book(BaseModel):
    title: str
    chapters: list[RawChapter]


class Segment(BaseModel):
    """One contiguous utterance spoken by a single voice."""

    speaker: str = NARRATOR
    text: str
    emotion: Emotion = "neutral"
    is_dialogue: bool = False
    # Silence after this segment in seconds; None lets the mixer pick a default.
    pause_after: float | None = Field(default=None, ge=0, le=5)


class Scene(BaseModel):
    setting: str = ""
    ambience: list[str] = Field(default_factory=list)
    segments: list[Segment]


class Chapter(BaseModel):
    index: int
    title: str
    scenes: list[Scene]


class Character(BaseModel):
    name: str
    gender: Gender = "unknown"
    voice: str | None = None


class Script(BaseModel):
    title: str
    characters: dict[str, Character] = Field(default_factory=dict)
    chapters: list[Chapter]

    def segment_count(self) -> int:
        return sum(len(s.segments) for c in self.chapters for s in c.scenes)
