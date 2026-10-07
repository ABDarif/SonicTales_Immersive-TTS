"""Orchestrates the whole thing: PDF/text -> script -> voices -> ambience -> audiobook files."""

from __future__ import annotations

import logging
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.audio_io import SAMPLE_RATE, concat_export, export_audio, resample, write_wav
from app.config import Settings, get_settings
from app.mixer import AMBIENCE_DB, Clip, master, mix_scene, stitch
from app.models import Book, Script
from app.pdf_extract import load_book
from app.script_builder import CompleteFn, build_script
from app.tts_engine import EMOTION_PROFILE, TTSEngine, TTSError, cast_voices, get_engine

log = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]
MAX_TTS_CHARS = 900  # split longer utterances at sentence boundaries (cloud APIs cap input length)


@dataclass
class PipelineOptions:
    tts: str = "auto"
    builder: str = "auto"  # auto | heuristic | llm
    ambience: bool = True
    ambience_db: float = AMBIENCE_DB
    fmt: str = "mp3"  # mp3 | m4b | wav
    chapters: list[int] | None = None  # 1-based chapter numbers to render; None = all
    max_chars: int | None = None


@dataclass
class PipelineResult:
    script: Script
    script_path: Path
    audio_path: Path
    chapter_paths: list[Path] = field(default_factory=list)
    duration_s: float = 0.0
    engine: str = ""
    elapsed_s: float = 0.0


def split_for_tts(text: str, limit: int = MAX_TTS_CHARS) -> list[str]:
    if len(text) <= limit:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts: list[str] = []
    cur = ""
    for s in sentences:
        while len(s) > limit:  # a single monster sentence: cut at the last space
            cut = s.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(s[:cut])
            s = s[cut:].lstrip()
        if cur and len(cur) + 1 + len(s) > limit:
            parts.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        parts.append(cur)
    return parts


def _synthesize(engine: TTSEngine, text: str, voice: str, emotion: str, retries: int = 2) -> np.ndarray:
    pieces = []
    for part in split_for_tts(text):
        for attempt in range(retries + 1):
            try:
                audio = engine.synthesize(part, voice, emotion)
                break
            except TTSError:
                if attempt == retries:
                    raise
                time.sleep(1.5 * (attempt + 1))
        pieces.append(resample(audio, engine.sample_rate, SAMPLE_RATE))
        pieces.append(np.zeros(int(0.12 * SAMPLE_RATE), dtype=np.float32))
    return np.concatenate(pieces[:-1])


def select_chapters(book: Book, chapters: list[int] | None, max_chars: int | None) -> Book:
    picked = book.chapters
    if chapters:
        bad = [c for c in chapters if not 1 <= c <= len(book.chapters)]
        if bad:
            raise ValueError(f"chapter(s) {bad} out of range; the book has {len(book.chapters)} chapters")
        picked = [book.chapters[c - 1] for c in sorted(set(chapters))]
    picked = [c for c in picked if c.paragraphs]
    if not picked:
        raise ValueError("no readable text found in the selected chapters")
    total = sum(len(p) for c in picked for p in c.paragraphs)
    if max_chars and total > max_chars:
        raise ValueError(
            f"selected text has {total:,} characters, above the limit of {max_chars:,}; select fewer chapters"
        )
    return Book(title=book.title, chapters=picked)


def render_script(
    script: Script,
    engine: TTSEngine,
    out_dir: Path,
    options: PipelineOptions,
    progress: ProgressFn | None = None,
) -> tuple[Path, list[Path], float]:
    voices = cast_voices(script, engine)
    total = max(1, script.segment_count())
    done = 0
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    wavs: list[Path] = []
    chapter_files: list[Path] = []
    markers: list[tuple[str, float, float]] = []
    cursor = 0.0
    gap_s = 1.5
    try:
        for chapter in script.chapters:
            scenes = []
            for scene in chapter.scenes:
                clips = []
                for seg in scene.segments:
                    audio = _synthesize(engine, seg.text, voices[seg.speaker], seg.emotion)
                    gain = EMOTION_PROFILE.get(seg.emotion, EMOTION_PROFILE["neutral"])["gain"]
                    clips.append(Clip(audio * gain, seg.pause_after if seg.pause_after is not None else 0.35))
                    done += 1
                    if progress:
                        progress(0.1 + 0.8 * done / total, f"Voicing chapter {chapter.index}: {done}/{total} lines")
                scenes.append(
                    mix_scene(clips, scene.ambience, ambience_db=options.ambience_db, with_ambience=options.ambience)
                )
            audio = master(stitch(scenes))
            wav = work / f"chapter_{chapter.index:03d}.wav"
            write_wav(wav, audio)
            wavs.append(wav)
            dur = audio.size / SAMPLE_RATE
            markers.append((chapter.title, cursor, cursor + dur))
            cursor += dur + gap_s
            if options.fmt != "wav" and len(script.chapters) > 1:
                ch_path = out_dir / f"chapter_{chapter.index:03d}.{options.fmt}"
                export_audio(audio, ch_path, title=f"{script.title}: {chapter.title}")
                chapter_files.append(ch_path)

        if progress:
            progress(0.93, "Encoding audiobook")
        full = out_dir / f"audiobook.{options.fmt}"
        concat_export(wavs, full, title=script.title, chapters=markers, gap_s=gap_s)
        return full, chapter_files, max(0.0, cursor - gap_s)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_pipeline(
    source: Path | Book,
    out_dir: Path,
    options: PipelineOptions | None = None,
    settings: Settings | None = None,
    progress: ProgressFn | None = None,
    engine: TTSEngine | None = None,
    complete: CompleteFn | None = None,
) -> PipelineResult:
    options = options or PipelineOptions()
    settings = settings or get_settings()
    if options.fmt not in {"mp3", "m4b", "wav"}:
        raise ValueError("fmt must be mp3, m4b or wav")
    started = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)
    emit = progress or (lambda *_: None)

    emit(0.01, "Reading book")
    book = source if isinstance(source, Book) else load_book(source)
    book = select_chapters(book, options.chapters, options.max_chars)

    emit(0.03, "Analysing speakers and scenes")
    script = build_script(
        book, options.builder, settings, complete=complete, progress=lambda f, m: emit(0.03 + 0.07 * f, m)
    )
    engine = engine or get_engine(options.tts, settings)

    audio_path, chapter_paths, duration = render_script(script, engine, out_dir, options, emit)
    script_path = out_dir / "script.json"
    script_path.write_text(script.model_dump_json(indent=2), encoding="utf-8")
    emit(1.0, "Done")
    return PipelineResult(
        script=script,
        script_path=script_path,
        audio_path=audio_path,
        chapter_paths=chapter_paths,
        duration_s=duration,
        engine=engine.name,
        elapsed_s=time.monotonic() - started,
    )
