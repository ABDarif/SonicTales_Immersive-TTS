"""Stage 5: speech clips + ambience beds -> finished scene / chapter audio.

Per scene:   [pre-roll bed] narration (+ ducked bed underneath) [tail bed]
Between scenes the tail of one overlaps the pre-roll of the next, so ambience
crossfades during the quiet gap instead of cutting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.ambience import render_bed
from app.audio_io import SAMPLE_RATE

AMBIENCE_DB = -12.0  # gain on ambience loops (normalised to ~-20 dBFS RMS) => bed ~-32 dBFS, ~15 dB under speech
DUCK_DB = -7.0  # extra attenuation of the bed while someone is speaking
PRE_ROLL_S = 1.2
TAIL_S = 1.8
DEFAULT_PAUSE_S = 0.35
MIN_SCENE_GAP_S = 0.4


@dataclass
class Clip:
    audio: np.ndarray
    pause_after_s: float = DEFAULT_PAUSE_S


def db_to_gain(db: float) -> float:
    return 10 ** (db / 20)


def assemble_speech(clips: list[Clip], sr: int = SAMPLE_RATE) -> np.ndarray:
    parts: list[np.ndarray] = []
    for c in clips:
        parts.append(c.audio.astype(np.float32))
        parts.append(np.zeros(int(c.pause_after_s * sr), dtype=np.float32))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
    """Centered box filter in O(n) (np.convolve with long kernels is O(n*k) and very slow)."""
    k = max(1, k | 1)  # odd width keeps it centered
    if k == 1 or x.size == 0:
        return x
    pad = k // 2
    padded = np.pad(x, (pad + 1, pad), mode="edge")
    csum = np.cumsum(padded, dtype=np.float64)
    return ((csum[k:] - csum[:-k]) / k)[: x.size]


def speech_envelope(speech: np.ndarray, sr: int = SAMPLE_RATE, window_s: float = 0.4) -> np.ndarray:
    """0..1 'someone is talking' curve: smoothed energy, thresholded relative to its own peak."""
    if speech.size == 0:
        return np.zeros(0, dtype=np.float32)
    energy = np.sqrt(np.maximum(_moving_average(speech.astype(np.float64) ** 2, int(window_s * sr)), 0.0))
    peak = energy.max()
    if peak < 1e-6:
        return np.zeros(speech.size, dtype=np.float32)
    env = np.clip(energy / (0.08 * peak), 0.0, 1.0)
    # two box passes ~ a triangular window: ducking glides instead of pumping
    glide = int(0.15 * sr)
    env = _moving_average(_moving_average(env, glide), glide)
    return np.clip(env, 0, 1).astype(np.float32)


def _fade(n_in: int, n_out: int, total: int) -> np.ndarray:
    f = np.ones(total, dtype=np.float32)
    n_in, n_out = min(n_in, total), min(n_out, total)
    if n_in:
        f[:n_in] = np.linspace(0, 1, n_in, dtype=np.float32) ** 1.5
    if n_out:
        f[total - n_out :] *= np.linspace(1, 0, n_out, dtype=np.float32) ** 1.5
    return f


def mix_scene(
    clips: list[Clip],
    ambience: list[str],
    sr: int = SAMPLE_RATE,
    ambience_db: float = AMBIENCE_DB,
    duck_db: float = DUCK_DB,
    with_ambience: bool = True,
) -> tuple[np.ndarray, float, float]:
    """Return (audio, pre_roll_s, tail_s). Pre-roll/tail are only added when there is a bed."""
    speech = assemble_speech(clips, sr)
    has_bed = bool(ambience) and with_ambience
    pre = int(PRE_ROLL_S * sr) if has_bed else 0
    tail = int(TAIL_S * sr) if has_bed else int(MIN_SCENE_GAP_S * sr)
    total = pre + speech.size + tail
    out = np.zeros(total, dtype=np.float32)
    out[pre : pre + speech.size] += speech
    if has_bed:
        bed = render_bed(ambience, total / sr, sr)[:total]
        if bed.size < total:
            bed = np.pad(bed, (0, total - bed.size))
        env = np.zeros(total, dtype=np.float32)
        env[pre : pre + speech.size] = speech_envelope(speech, sr)
        # while speech is present the bed drops by `duck_db`
        gain = 1.0 - env * (1.0 - db_to_gain(duck_db))
        # fade in/out only at the scene boundaries; `mix_chapter` overlaps neighbours to crossfade
        out += bed * gain * _fade(pre, tail, total) * db_to_gain(ambience_db)
    return out, pre / sr, (tail / sr if has_bed else 0.0)


def stitch(scenes: list[tuple[np.ndarray, float, float]], sr: int = SAMPLE_RATE) -> np.ndarray:
    """Concatenate scenes, overlapping one scene's tail with the next scene's pre-roll."""
    if not scenes:
        return np.zeros(0, dtype=np.float32)
    out = scenes[0][0].copy()
    prev_tail = scenes[0][2]
    for audio, pre_s, tail_s in scenes[1:]:
        overlap = int(min(prev_tail, pre_s) * sr)
        overlap = min(overlap, out.size, audio.size)
        start = out.size - overlap
        out = np.concatenate([out, np.zeros(audio.size - overlap, dtype=np.float32)])
        out[start : start + audio.size] += audio
        prev_tail = tail_s
    return out


def master(audio: np.ndarray, peak_db: float = -1.0) -> np.ndarray:
    """Remove DC and scale so the loudest sample sits at `peak_db` (final loudness is set by ffmpeg loudnorm)."""
    if audio.size == 0:
        return audio
    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio * (db_to_gain(peak_db) / peak)
    return audio.astype(np.float32)
