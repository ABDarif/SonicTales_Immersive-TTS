"""Stage 4: scene ambience tags -> background audio beds.

Each tag has a procedural generator (numpy only, no downloads, no licensing issues).
Noise is shaped in the frequency domain and every modulation uses a whole number of
cycles per loop, so each generated loop is *periodic* and tiles without a seam.

Drop your own recordings in `assets/ambience/<tag>.(wav|mp3|ogg|flac)` to override a
generator; those files are looped with a short crossfade.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.audio_io import SAMPLE_RATE, read_audio_file, resample
from app.config import ROOT

LOOP_SECONDS = 24.0
ASSET_DIR = ROOT / "assets" / "ambience"
ASSET_EXTS = (".wav", ".mp3", ".ogg", ".flac")

# Relative level of each tag's bed, in dB, before the global ambience level is applied.
TAG_LEVEL_DB = {
    "rain": 0.0,
    "thunder": -2.0,
    "wind": -3.0,
    "night": -6.0,
    "fire": -3.0,
    "ocean": -1.0,
    "forest": -5.0,
    "river": -2.0,
    "city": -4.0,
    "crowd": -5.0,
    "birds": -7.0,
}


def _rng(tag: str) -> np.random.Generator:
    return np.random.default_rng(int.from_bytes(hashlib.sha256(tag.encode()).digest()[:4], "big"))


def _bandpass_noise(rng: np.random.Generator, n: int, sr: int, lo: float, hi: float, slope: float = 0.0) -> np.ndarray:
    """White noise filtered in the FFT domain to [lo, hi] Hz with a 1/f^slope tilt (0 white, 1 pink, 2 brown)."""
    spec = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    # soft-edged band: ramp up around `lo`, ramp down around `hi`, so the filter doesn't ring
    w_lo, w_hi = max(lo * 0.25, 20.0), max(hi * 0.25, 20.0)
    mask = np.clip(0.5 + (freqs - lo) / (2 * w_lo), 0, 1) * np.clip(0.5 + (hi - freqs) / (2 * w_hi), 0, 1)
    tilt = 1.0 / np.maximum(freqs, 20.0) ** (slope / 2.0)
    out = np.fft.irfft(spec * mask * tilt, n)
    return out / (np.max(np.abs(out)) + 1e-9)


def _periodic_lfo(n: int, sr: int, period_s: float, phase: float = 0.0) -> np.ndarray:
    cycles = max(1, round(LOOP_SECONDS / period_s))
    return np.sin(2 * np.pi * (cycles * np.arange(n) / n) + phase)


def _smooth_random(rng: np.random.Generator, n: int, sr: int, rate_hz: float) -> np.ndarray:
    """Slowly varying random curve in [0,1] that is periodic over the loop."""
    pts = max(4, int(LOOP_SECONDS * rate_hz))
    knots = rng.random(pts)
    x = np.linspace(0, pts, n, endpoint=False)
    i0 = np.floor(x).astype(int) % pts
    i1 = (i0 + 1) % pts
    f = x - np.floor(x)
    f = f * f * (3 - 2 * f)
    return knots[i0] * (1 - f) + knots[i1] * f


def _add_wrapped(buf: np.ndarray, start: int, sig: np.ndarray) -> None:
    """Add `sig` to circular buffer `buf` at `start`, wrapping around the end."""
    idx = (start + np.arange(sig.size)) % buf.size
    np.add.at(buf, idx, sig)


def _impulses(
    rng: np.random.Generator, n: int, sr: int, rate: float, decay_ms: float, amp: tuple[float, float], tone: float = 0.0
) -> np.ndarray:
    """Random clicks / droplets / crackles with exponential decay."""
    out = np.zeros(n)
    length = int(decay_ms / 1000 * sr * 6)
    t = np.arange(length) / sr
    env = np.exp(-t / (decay_ms / 1000))
    for pos in rng.integers(0, n, size=int(rate * LOOP_SECONDS)):
        a = rng.uniform(*amp)
        if tone:
            f = tone * rng.uniform(0.7, 1.4)
            sig = np.sin(2 * np.pi * f * t) * env
        else:
            sig = rng.standard_normal(length) * env
        _add_wrapped(out, int(pos), a * sig)
    return out


def _gen_rain(rng, n, sr):
    body = _bandpass_noise(rng, n, sr, 800, 9000, slope=0.6)
    hiss = _bandpass_noise(rng, n, sr, 3000, 11000, slope=0.0) * 0.4
    drops = _impulses(rng, n, sr, rate=140, decay_ms=3, amp=(0.1, 0.5))
    swell = 0.85 + 0.15 * _periodic_lfo(n, sr, 12.0)
    return (body * 0.7 + hiss + drops * 0.5) * swell


def _gen_thunder(rng, n, sr):
    out = np.zeros(n)
    rumble = _bandpass_noise(rng, n, sr, 25, 160, slope=2.0)
    for start in (int(n * 0.18), int(n * 0.62)):
        length = int(sr * rng.uniform(4.0, 6.0))
        t = np.arange(length) / sr
        env = (1 - np.exp(-t / 0.15)) * np.exp(-t / 1.6)
        env *= 0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(0.6, 1.3) * t) ** 2
        _add_wrapped(out, start, rumble[:length] * env * rng.uniform(0.7, 1.0))
    return out


def _gen_wind(rng, n, sr):
    low = _bandpass_noise(rng, n, sr, 80, 700, slope=1.5)
    whistle = _bandpass_noise(rng, n, sr, 600, 2200, slope=1.0) * 0.35
    gust = 0.35 + 0.65 * _smooth_random(rng, n, sr, 0.25)
    return (low + whistle * gust) * gust


def _gen_night(rng, n, sr):
    t = np.arange(n) / sr
    out = np.zeros(n)
    for _ in range(5):  # several crickets, each with its own pitch and chirp rhythm
        f = rng.uniform(3800, 5200)
        group_s = LOOP_SECONDS / max(1, round(LOOP_SECONDS / rng.uniform(1.1, 2.3)))
        pulse_hz = round(rng.uniform(26, 34) * LOOP_SECONDS) / LOOP_SECONDS
        phase = rng.uniform(0, 1)
        grp = (((t / group_s) + phase) % 1.0) < 0.35
        pulses = 0.5 + 0.5 * np.sin(2 * np.pi * pulse_hz * t)
        carrier = np.sin(2 * np.pi * round(f * LOOP_SECONDS) / LOOP_SECONDS * t)
        smooth = np.convolve(
            grp.astype(float), np.hanning(int(sr * 0.02)) / np.hanning(int(sr * 0.02)).sum(), mode="same"
        )
        out += carrier * pulses * smooth * rng.uniform(0.4, 0.8)
    hush = _bandpass_noise(rng, n, sr, 60, 500, slope=2.0) * 0.5
    return out * 0.45 + hush


def _gen_fire(rng, n, sr):
    base = _bandpass_noise(rng, n, sr, 40, 900, slope=2.0) * (0.6 + 0.4 * _smooth_random(rng, n, sr, 3.0))
    crackle = _impulses(rng, n, sr, rate=22, decay_ms=7, amp=(0.2, 1.0))
    pops = _impulses(rng, n, sr, rate=4, decay_ms=18, amp=(0.4, 1.0), tone=900)
    return base * 0.8 + crackle * 0.8 + pops * 0.5


def _gen_ocean(rng, n, sr):
    noise = _bandpass_noise(rng, n, sr, 120, 3500, slope=1.2)
    wave = 0.5 + 0.5 * _periodic_lfo(n, sr, 8.0)
    swell = wave**2 * 0.9 + 0.1
    foam = _bandpass_noise(rng, n, sr, 2500, 8000, slope=0.5) * wave**4 * 0.5
    return noise * swell + foam


def _gen_river(rng, n, sr):
    flow = _bandpass_noise(rng, n, sr, 300, 5000, slope=0.8)
    babble = _bandpass_noise(rng, n, sr, 1200, 4500, slope=0.0) * _smooth_random(rng, n, sr, 12.0)
    return flow * 0.7 + babble * 0.7


def _chirp(rng, sr) -> np.ndarray:
    dur = rng.uniform(0.08, 0.22)
    t = np.arange(int(dur * sr)) / sr
    f0, f1 = rng.uniform(2200, 4200), rng.uniform(3000, 6000)
    freq = f0 + (f1 - f0) * (t / dur) ** rng.uniform(0.6, 1.6)
    wobble = 1 + 0.03 * np.sin(2 * np.pi * rng.uniform(20, 50) * t)
    return np.sin(2 * np.pi * np.cumsum(freq * wobble) / sr) * np.sin(np.pi * t / dur) ** 2


def _gen_birds(rng, n, sr):
    out = np.zeros(n)
    t = 0.0
    while t < LOOP_SECONDS - 1.5:
        t += rng.uniform(0.6, 3.5)
        for k in range(int(rng.integers(1, 5))):  # a short phrase of 1-4 chirps
            _add_wrapped(out, int((t + k * rng.uniform(0.12, 0.28)) * sr), _chirp(rng, sr) * rng.uniform(0.3, 0.9))
    return out + _bandpass_noise(rng, n, sr, 200, 1500, slope=1.5) * 0.08


def _gen_forest(rng, n, sr):
    leaves = _bandpass_noise(rng, n, sr, 200, 4000, slope=1.4) * (0.35 + 0.65 * _smooth_random(rng, n, sr, 0.4))
    return leaves * 0.8 + _gen_birds(rng, n, sr) * 0.45


def _gen_city(rng, n, sr):
    rumble = _bandpass_noise(rng, n, sr, 40, 400, slope=2.0)
    hum = _bandpass_noise(rng, n, sr, 400, 2500, slope=1.2) * 0.3 * (0.5 + 0.5 * _smooth_random(rng, n, sr, 0.5))
    out = rumble * 0.8 + hum
    for _ in range(3):  # passing cars: swells of mid-low noise
        length = int(sr * rng.uniform(3, 5))
        t = np.linspace(0, 1, length)
        env = np.sin(np.pi * t) ** 2
        _add_wrapped(out, int(rng.integers(0, n)), _bandpass_noise(rng, length, sr, 100, 1500, slope=1.0) * env * 0.6)
    return out


def _gen_crowd(rng, n, sr):
    out = np.zeros(n)
    for _ in range(14):  # many overlapping, syllable-rate-modulated voices
        voice = _bandpass_noise(rng, n, sr, rng.uniform(150, 400), rng.uniform(1800, 3200), slope=1.0)
        syll = np.clip(_smooth_random(rng, n, sr, rng.uniform(2.5, 5.0)) * 1.6 - 0.2, 0, 1)
        out += voice * syll * rng.uniform(0.3, 1.0)
    return out


GENERATORS = {
    "rain": _gen_rain,
    "thunder": _gen_thunder,
    "wind": _gen_wind,
    "night": _gen_night,
    "fire": _gen_fire,
    "ocean": _gen_ocean,
    "river": _gen_river,
    "birds": _gen_birds,
    "forest": _gen_forest,
    "city": _gen_city,
    "crowd": _gen_crowd,
}


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def _loop_with_crossfade(audio: np.ndarray, sr: int, fade_s: float = 1.0) -> np.ndarray:
    """Make an arbitrary recording tile seamlessly by crossfading its tail into its head."""
    fade = min(int(fade_s * sr), audio.size // 3)
    if fade < 2:
        return audio
    head, tail = audio[:fade], audio[-fade:]
    ramp = np.linspace(0, 1, fade)
    blended = tail * (1 - ramp) + head * ramp
    return np.concatenate([audio[fade:-fade], blended]).astype(np.float32)


def _find_asset(tag: str, asset_dir: Path) -> Path | None:
    for ext in ASSET_EXTS:
        p = asset_dir / f"{tag}{ext}"
        if p.is_file():
            return p
    return None


@lru_cache(maxsize=64)
def _loop(tag: str, sr: int, asset_dir: str) -> np.ndarray:
    """One loopable segment for `tag`, normalised to a fixed RMS so levels are predictable."""
    asset = _find_asset(tag, Path(asset_dir))
    if asset is not None:
        audio, src_sr = read_audio_file(asset)
        audio = _loop_with_crossfade(resample(audio, src_sr, sr), sr)
    else:
        if tag not in GENERATORS:
            raise KeyError(f"unknown ambience tag {tag!r}")
        n = int(LOOP_SECONDS * sr)
        audio = GENERATORS[tag](_rng(tag), n, sr)
    audio = audio - np.mean(audio)
    audio = audio / _rms(audio) * 0.1  # ~ -20 dBFS RMS
    return audio.astype(np.float32)


def _tile(loop: np.ndarray, n: int) -> np.ndarray:
    reps = -(-n // loop.size)
    return np.tile(loop, reps)[:n]


def render_bed(tags: list[str], duration_s: float, sr: int = SAMPLE_RATE, asset_dir: Path | None = None) -> np.ndarray:
    """Mix the beds for `tags` into a single track of `duration_s` seconds (silence if no tags)."""
    n = int(duration_s * sr)
    bed = np.zeros(n, dtype=np.float32)
    for tag in tags:
        if tag not in GENERATORS and not _find_asset(tag, asset_dir or ASSET_DIR):
            continue
        gain = 10 ** (TAG_LEVEL_DB.get(tag, -4.0) / 20)
        # Offset each tag differently into its loop so layers don't phase together.
        loop = np.roll(_loop(tag, sr, str(asset_dir or ASSET_DIR)), int(_rng(tag + "off").integers(0, sr * 5)))
        bed += _tile(loop, n) * gain
    peak = float(np.max(np.abs(bed))) if n else 0.0
    if peak > 0.9:
        bed *= 0.9 / peak
    return bed


def known_tags() -> list[str]:
    return sorted(GENERATORS)
