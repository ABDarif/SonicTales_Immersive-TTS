import subprocess
from pathlib import Path

import pytest

from app.audio_io import SAMPLE_RATE, read_audio_file
from app.pdf_extract import book_from_text
from app.pipeline import PipelineOptions, run_pipeline, select_chapters, split_for_tts
from app.tts_engine import SyntheticEngine, TTSError

from .conftest import STORY, needs_ffmpeg


def probe(path: Path, *args: str) -> str:
    return subprocess.run(
        ["ffprobe", "-v", "error", *args, str(path)], capture_output=True, text=True, check=True
    ).stdout


def test_end_to_end_wav(tmp_path, settings):
    progress = []
    res = run_pipeline(
        book_from_text(STORY, "Story"),
        tmp_path,
        PipelineOptions(tts="synthetic", fmt="wav"),
        settings,
        progress=lambda f, m: progress.append(f),
    )
    audio, sr = read_audio_file(res.audio_path)
    assert sr == SAMPLE_RATE and audio.size / sr > 10
    assert res.duration_s == pytest.approx(audio.size / sr, abs=2.0)
    assert res.script_path.is_file() and res.script.characters["Mary"].voice
    assert progress == sorted(progress) and progress[-1] == 1.0, "progress is monotonic and ends at 100%"
    assert not (tmp_path / "work").exists(), "temp files are cleaned up"


def test_ambience_changes_the_audio_and_can_be_disabled(tmp_path, settings):
    book = book_from_text("It was a stormy night. The rain fell hard on the roof of the old house.")
    opts = dict(tts="synthetic", fmt="wav")
    on = run_pipeline(book, tmp_path / "on", PipelineOptions(**opts), settings)
    off = run_pipeline(book, tmp_path / "off", PipelineOptions(ambience=False, **opts), settings)
    a_on, _ = read_audio_file(on.audio_path)
    a_off, _ = read_audio_file(off.audio_path)
    assert a_on.size > a_off.size, "ambience adds a pre-roll and a tail"
    assert on.script.chapters[0].scenes[0].ambience[:2] == ["rain", "night"]


@needs_ffmpeg
def test_mp3_per_chapter_files_and_m4b_chapter_markers(tmp_path, settings):
    mp3 = run_pipeline(
        book_from_text(STORY, "S"), tmp_path / "mp3", PipelineOptions(tts="synthetic", fmt="mp3"), settings
    )
    assert mp3.audio_path.suffix == ".mp3" and mp3.audio_path.stat().st_size > 5000
    assert [p.name for p in mp3.chapter_paths] == ["chapter_001.mp3", "chapter_002.mp3"]

    m4b = run_pipeline(
        book_from_text(STORY, "S"), tmp_path / "m4b", PipelineOptions(tts="synthetic", fmt="m4b"), settings
    )
    out = probe(
        m4b.audio_path, "-show_chapters", "-show_entries", "chapter=start_time:chapter_tags=title", "-of", "compact"
    )
    assert "title=Chapter 1" in out and "title=Chapter 2" in out
    dur = float(probe(m4b.audio_path, "-show_entries", "format=duration", "-of", "csv=p=0"))
    assert dur == pytest.approx(m4b.duration_s, abs=1.5)


@needs_ffmpeg
def test_loudness_is_normalised(tmp_path, settings):
    res = run_pipeline(book_from_text(STORY), tmp_path, PipelineOptions(tts="synthetic", fmt="mp3"), settings)
    out = subprocess.run(
        ["ffmpeg", "-nostats", "-i", str(res.audio_path), "-af", "ebur128", "-f", "null", "-"],
        capture_output=True,
        text=True,
    ).stderr
    integrated = float(out.rsplit("I:", 1)[1].split("LUFS")[0])
    assert -19 < integrated < -13


def test_chapter_selection_and_limits():
    book = book_from_text(STORY)
    assert [c.title for c in select_chapters(book, [2], None).chapters] == ["Chapter 2"]
    with pytest.raises(ValueError, match="out of range"):
        select_chapters(book, [9], None)
    with pytest.raises(ValueError, match="above the limit"):
        select_chapters(book, None, 50)
    with pytest.raises(ValueError, match="no readable text"):
        select_chapters(book_from_text("Chapter 1\n\nChapter 2"), None, None)


def test_chapters_option_renders_only_those(tmp_path, settings):
    res = run_pipeline(
        book_from_text(STORY), tmp_path, PipelineOptions(tts="synthetic", fmt="wav", chapters=[2]), settings
    )
    assert [c.title for c in res.script.chapters] == ["Chapter 2"]


def test_split_for_tts_keeps_everything_under_limit():
    text = " ".join(f"This is sentence number {i}." for i in range(200))
    parts = split_for_tts(text, 300)
    assert all(len(p) <= 300 for p in parts) and " ".join(parts) == text
    assert split_for_tts("short") == ["short"]
    long_word_sentence = "word " * 500
    assert all(len(p) <= 100 for p in split_for_tts(long_word_sentence.strip(), 100))


class FlakyEngine(SyntheticEngine):
    def __init__(self, failures: int):
        self.failures = failures

    def synthesize(self, text, voice, emotion="neutral"):
        if self.failures > 0:
            self.failures -= 1
            raise TTSError("transient")
        return super().synthesize(text, voice, emotion)


def test_transient_tts_errors_are_retried(tmp_path, settings, monkeypatch):
    monkeypatch.setattr("app.pipeline.time.sleep", lambda s: None)
    book = book_from_text("A short calm line of narration for the test.")
    run_pipeline(book, tmp_path / "ok", PipelineOptions(fmt="wav"), settings, engine=FlakyEngine(2))
    with pytest.raises(TTSError):
        run_pipeline(book, tmp_path / "bad", PipelineOptions(fmt="wav"), settings, engine=FlakyEngine(99))


def test_invalid_format_rejected(tmp_path, settings):
    with pytest.raises(ValueError, match="fmt"):
        run_pipeline(book_from_text("Hello there."), tmp_path, PipelineOptions(fmt="ogg"), settings)
