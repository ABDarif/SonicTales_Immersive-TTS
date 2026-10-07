"""Audio helpers: float32 mono numpy arrays in, WAV/MP3/M4B files out.

We deliberately avoid pydub (it depends on the removed `audioop` module on Python 3.13)
and talk to ffmpeg directly for compressed formats.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24000  # internal mixing rate


class AudioError(RuntimeError):
    pass


def resample(audio: np.ndarray, sr_from: int, sr_to: int = SAMPLE_RATE) -> np.ndarray:
    if sr_from == sr_to or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    n_out = max(1, int(round(audio.size * sr_to / sr_from)))
    x_old = np.linspace(0.0, 1.0, audio.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def read_wav_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Decode a PCM WAV (8/16/32-bit, any channel count) to mono float32."""
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            frames = w.readframes(w.getnframes())
    except (wave.Error, EOFError) as exc:
        raise AudioError(f"not a readable PCM WAV: {exc}") from exc
    if width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise AudioError(f"unsupported sample width {width}")
    if ch > 1:
        audio = audio[: audio.size // ch * ch].reshape(-1, ch).mean(axis=1)
    return audio, sr


def wav_bytes(audio: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def write_wav(path: Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(wav_bytes(audio, sr))


def read_audio_file(path: Path) -> tuple[np.ndarray, int]:
    """Read WAV directly, anything else through ffmpeg."""
    if path.suffix.lower() == ".wav":
        try:
            return read_wav_bytes(path.read_bytes())
        except AudioError:
            pass  # e.g. float WAV; let ffmpeg handle it
    ffmpeg = _ffmpeg()
    proc = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg could not decode {path.name}: {proc.stderr.decode(errors='replace')[:200]}")
    return np.frombuffer(proc.stdout, dtype="<f4").copy(), SAMPLE_RATE


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise AudioError("ffmpeg is required for mp3/m4b export and non-WAV input; install it first")
    return exe


def _escape_meta(value: str) -> str:
    for ch in ("\\", "=", ";", "#", "\n"):
        value = value.replace(ch, "\\" + ch)
    return value


def export_audio(
    audio: np.ndarray,
    path: Path,
    sr: int = SAMPLE_RATE,
    title: str = "",
    chapters: list[tuple[str, float, float]] | None = None,
    loudnorm: bool = True,
) -> Path:
    """Write `audio` to `path`; the extension picks the format (.wav, .mp3, .m4b/.m4a).

    `chapters` is a list of (title, start_s, end_s) embedded as chapter markers for .m4b.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".wav":
        write_wav(path, audio, sr)
        return path
    if suffix not in {".mp3", ".m4b", ".m4a"}:
        raise AudioError(f"unsupported output format {suffix!r} (use .wav, .mp3 or .m4b)")

    ffmpeg = _ffmpeg()
    cmd = [ffmpeg, "-v", "error", "-y", "-f", "wav", "-i", "pipe:0"]
    meta_path: Path | None = None
    if suffix in {".m4b", ".m4a"} and chapters:
        lines = [";FFMETADATA1", f"title={_escape_meta(title)}"]
        for ch_title, start, end in chapters:
            lines += [
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={int(start * 1000)}",
                f"END={int(end * 1000)}",
                f"title={_escape_meta(ch_title)}",
            ]
        meta_path = path.with_suffix(path.suffix + ".meta")
        meta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        cmd += ["-i", str(meta_path), "-map_metadata", "1", "-map_chapters", "1"]
    if loudnorm:
        cmd += ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"]
    cmd += ["-metadata", f"title={title}"]
    cmd += ["-c:a", "libmp3lame", "-b:a", "128k"] if suffix == ".mp3" else ["-c:a", "aac", "-b:a", "96k"]
    if suffix == ".m4b":
        cmd += ["-f", "ipod"]
    cmd.append(str(path))
    try:
        proc = subprocess.run(cmd, input=wav_bytes(audio, sr), capture_output=True, check=False)
    finally:
        if meta_path:
            meta_path.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg export failed: {proc.stderr.decode(errors='replace')[:300]}")
    return path


def concat_export(
    wav_paths: list[Path],
    path: Path,
    title: str = "",
    chapters: list[tuple[str, float, float]] | None = None,
    gap_s: float = 1.5,
    loudnorm: bool = True,
) -> Path:
    """Join chapter WAVs (with `gap_s` of silence between them) into one file without loading them all into memory."""
    if not wav_paths:
        raise AudioError("nothing to concatenate")
    ffmpeg = _ffmpeg()
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    work = path.parent / f".concat_{path.stem}"
    work.mkdir(exist_ok=True)
    try:
        gap = work / "gap.wav"
        write_wav(gap, np.zeros(int(gap_s * SAMPLE_RATE), dtype=np.float32))
        listing = work / "list.txt"
        lines: list[str] = []
        for i, p in enumerate(wav_paths):
            if i:
                lines.append(f"file '{gap.resolve().as_posix()}'")
            lines.append(f"file '{p.resolve().as_posix()}'")
        listing.write_text("\n".join(lines) + "\n", encoding="utf-8")

        cmd = [ffmpeg, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing)]
        if suffix in {".m4b", ".m4a"} and chapters:
            meta = work / "chapters.ffmeta"
            body = [";FFMETADATA1", f"title={_escape_meta(title)}"]
            for ch_title, start, end in chapters:
                body += [
                    "[CHAPTER]",
                    "TIMEBASE=1/1000",
                    f"START={int(start * 1000)}",
                    f"END={int(end * 1000)}",
                    f"title={_escape_meta(ch_title)}",
                ]
            meta.write_text("\n".join(body) + "\n", encoding="utf-8")
            cmd += ["-i", str(meta), "-map", "0:a", "-map_metadata", "1", "-map_chapters", "1"]
        if suffix == ".wav":
            cmd += ["-c", "copy"]
        else:
            if loudnorm:
                cmd += ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"]
            cmd += ["-metadata", f"title={title}"]
            cmd += ["-c:a", "libmp3lame", "-b:a", "128k"] if suffix == ".mp3" else ["-c:a", "aac", "-b:a", "96k"]
            if suffix == ".m4b":
                cmd += ["-f", "ipod"]
        cmd.append(str(path))
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0:
            raise AudioError(f"ffmpeg concat failed: {proc.stderr.decode(errors='replace')[:300]}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return path
