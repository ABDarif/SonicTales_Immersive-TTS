"""Stage 3: text + voice + emotion -> audio.

Engines implement one tiny interface so the rest of the pipeline never cares which
backend produced the speech:

    kokoro    local neural TTS (pip install kokoro soundfile; best free quality, CPU ok)
    openai    cloud TTS (needs OPENAI_API_KEY)
    espeak    espeak-ng command line (robotic, but works anywhere it is installed)
    synthetic offline stand-in that is NOT speech; used for tests, CI and dry runs
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from app.audio_io import SAMPLE_RATE, AudioError, read_wav_bytes, resample, wav_bytes
from app.config import Settings
from app.models import NARRATOR, Character, Script

log = logging.getLogger(__name__)

# How each emotion changes delivery. `speed` is honoured by engines that support it,
# `gain` is applied by the mixer so even simple engines get louder shouts / softer whispers.
EMOTION_PROFILE: dict[str, dict[str, float]] = {
    "neutral": {"speed": 1.0, "gain": 1.0},
    "happy": {"speed": 1.05, "gain": 1.05},
    "excited": {"speed": 1.1, "gain": 1.15},
    "sad": {"speed": 0.9, "gain": 0.9},
    "angry": {"speed": 1.05, "gain": 1.2},
    "afraid": {"speed": 1.08, "gain": 0.95},
    "whisper": {"speed": 0.92, "gain": 0.55},
}
OPENAI_EMOTION_HINT = {
    "happy": "cheerful and warm",
    "excited": "excited and energetic",
    "sad": "sad and subdued",
    "angry": "angry and forceful",
    "afraid": "frightened and trembling",
    "whisper": "whispering softly",
}


class TTSError(RuntimeError):
    pass


class TTSEngine(ABC):
    name: str
    sample_rate: int = SAMPLE_RATE
    #: voice pools keyed by role: "narrator", "male", "female"
    voices: dict[str, list[str]]

    @abstractmethod
    def synthesize(self, text: str, voice: str, emotion: str = "neutral") -> np.ndarray:
        """Return mono float32 audio at `self.sample_rate`."""

    def cache_key(self) -> str:
        return self.name


# --------------------------------------------------------------------------- synthetic


class SyntheticEngine(TTSEngine):
    """Deterministic, dependency-free placeholder: pitched babble shaped like speech timing.

    It exists so the whole pipeline (mixing, export, UI) can run and be tested without
    downloading models. It is intentionally not intelligible.
    """

    name = "synthetic"
    voices = {
        "narrator": ["syn_narrator"],
        "male": ["syn_m1", "syn_m2", "syn_m3"],
        "female": ["syn_f1", "syn_f2", "syn_f3"],
    }
    _PITCH = {
        "syn_narrator": 125.0,
        "syn_m1": 105.0,
        "syn_m2": 140.0,
        "syn_m3": 90.0,
        "syn_f1": 200.0,
        "syn_f2": 235.0,
        "syn_f3": 180.0,
    }

    def synthesize(self, text: str, voice: str, emotion: str = "neutral") -> np.ndarray:
        sr = self.sample_rate
        base = self._PITCH.get(voice, 130.0)
        speed = EMOTION_PROFILE.get(emotion, EMOTION_PROFILE["neutral"])["speed"]
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big"))
        out: list[np.ndarray] = []
        for word in text.split():
            letters = max(1, len(word.strip(".,;:!?\"'“”()")))
            n_syll = max(1, round(letters / 3))
            for _ in range(n_syll):
                dur = float(rng.uniform(0.11, 0.17)) / speed
                t = np.arange(int(dur * sr)) / sr
                f0 = base * float(rng.uniform(0.9, 1.15))
                phase = 2 * np.pi * np.cumsum(np.full(t.size, f0)) / sr
                sig = sum(np.sin(h * phase) / h for h in range(1, 7))
                env = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 0.7
                out.append((sig * env * 0.12).astype(np.float32))
            gap = 0.05 + (0.18 if word[-1:] in ",;:" else 0.0) + (0.3 if word[-1:] in ".!?" else 0.0)
            out.append(np.zeros(int(gap / speed * sr), dtype=np.float32))
        return np.concatenate(out) if out else np.zeros(int(0.2 * sr), dtype=np.float32)


# --------------------------------------------------------------------------- espeak


class EspeakEngine(TTSEngine):
    name = "espeak"
    sample_rate = 22050
    voices = {
        "narrator": ["en-us+m3"],
        "male": ["en-us+m1", "en-us+m2", "en-gb+m4", "en-us+m5"],
        "female": ["en-us+f1", "en-us+f2", "en-us+f3", "en-gb+f4"],
    }

    def __init__(self) -> None:
        self.exe = shutil.which("espeak-ng") or shutil.which("espeak")
        if not self.exe:
            raise TTSError("espeak-ng is not installed (apt install espeak-ng)")

    def synthesize(self, text: str, voice: str, emotion: str = "neutral") -> np.ndarray:
        speed = EMOTION_PROFILE.get(emotion, EMOTION_PROFILE["neutral"])["speed"]
        proc = subprocess.run(
            [self.exe, "--stdout", "-v", voice, "-s", str(int(165 * speed)), "--", text],
            capture_output=True,
            check=False,
            timeout=120,
        )
        if proc.returncode != 0:
            raise TTSError(f"espeak failed: {proc.stderr.decode(errors='replace')[:200]}")
        audio, sr = read_wav_bytes(proc.stdout)
        return resample(audio, sr, self.sample_rate)


# --------------------------------------------------------------------------- kokoro


class KokoroEngine(TTSEngine):
    name = "kokoro"
    sample_rate = 24000
    voices = {
        "narrator": ["bm_george", "bf_emma"],
        "male": ["am_adam", "am_michael", "bm_lewis", "am_fenrir", "am_puck", "bm_daniel", "am_eric"],
        "female": ["af_bella", "af_nicole", "af_sarah", "af_sky", "bf_isabella", "af_nova", "bf_alice"],
    }

    def __init__(self) -> None:
        try:
            from kokoro import KPipeline  # type: ignore[import-not-found]
        except ImportError as exc:
            raise TTSError("kokoro is not installed: pip install -r requirements-tts.txt") from exc
        self._pipelines: dict[str, object] = {}
        self._KPipeline = KPipeline

    def _pipeline(self, voice: str):
        lang = voice[0]  # 'a' American English, 'b' British English
        if lang not in self._pipelines:
            self._pipelines[lang] = self._KPipeline(lang_code=lang)
        return self._pipelines[lang]

    def synthesize(self, text: str, voice: str, emotion: str = "neutral") -> np.ndarray:
        speed = EMOTION_PROFILE.get(emotion, EMOTION_PROFILE["neutral"])["speed"]
        chunks = []
        for _, _, audio in self._pipeline(voice)(text, voice=voice, speed=speed):
            chunks.append(np.asarray(audio, dtype=np.float32))
        if not chunks:
            raise TTSError(f"kokoro produced no audio for: {text[:40]!r}")
        return np.concatenate(chunks)


# --------------------------------------------------------------------------- openai


class OpenAIEngine(TTSEngine):
    name = "openai"
    sample_rate = 24000
    voices = {
        "narrator": ["onyx", "fable"],
        "male": ["echo", "ash", "onyx", "verse"],
        "female": ["nova", "shimmer", "coral", "sage"],
    }
    url = "https://api.openai.com/v1/audio/speech"

    def __init__(self, api_key: str | None, model: str = "gpt-4o-mini-tts", client=None) -> None:
        if not api_key:
            raise TTSError("OPENAI_API_KEY is not set")
        import httpx

        self.model = model
        self.client = client or httpx.Client(timeout=120)
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def synthesize(self, text: str, voice: str, emotion: str = "neutral") -> np.ndarray:
        payload: dict[str, object] = {
            "model": self.model,
            "voice": voice,
            "input": text,
            "response_format": "wav",
        }
        if emotion in OPENAI_EMOTION_HINT:
            payload["instructions"] = f"Speak in a {OPENAI_EMOTION_HINT[emotion]} tone, like an audiobook performer."
        resp = self.client.post(self.url, headers=self.headers, json=payload)
        if resp.status_code != 200:
            raise TTSError(f"OpenAI TTS error {resp.status_code}: {resp.text[:200]}")
        audio, sr = read_wav_bytes(resp.content)
        return resample(audio, sr, self.sample_rate)

    def cache_key(self) -> str:
        return f"{self.name}:{self.model}"


# --------------------------------------------------------------------------- cache + casting


class CachedEngine(TTSEngine):
    """Wraps an engine with an on-disk cache so reruns and retries cost nothing."""

    def __init__(self, inner: TTSEngine, cache_dir: Path) -> None:
        self.inner = inner
        self.cache_dir = cache_dir
        self.name = inner.name
        self.sample_rate = inner.sample_rate
        self.voices = inner.voices
        self.hits = 0
        self.misses = 0

    def synthesize(self, text: str, voice: str, emotion: str = "neutral") -> np.ndarray:
        key = hashlib.sha256(f"{self.inner.cache_key()}|{voice}|{emotion}|{text}".encode()).hexdigest()
        path = self.cache_dir / key[:2] / f"{key}.wav"
        if path.is_file():
            try:
                audio, sr = read_wav_bytes(path.read_bytes())
                self.hits += 1
                return resample(audio, sr, self.sample_rate)
            except AudioError:
                path.unlink(missing_ok=True)  # corrupt entry: regenerate
        audio = self.inner.synthesize(text, voice, emotion)
        self.misses += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(wav_bytes(audio, self.sample_rate))
        tmp.replace(path)  # atomic: a crash never leaves a half-written cache file
        return audio

    def cache_key(self) -> str:
        return self.inner.cache_key()


def cast_voices(script: Script, engine: TTSEngine) -> dict[str, str]:
    """Give every character a stable voice from the engine's pools (by gender, round-robin)."""
    narrator = engine.voices["narrator"][0]
    assignments: dict[str, str] = {NARRATOR: narrator}
    used = {"male": 0, "female": 0}
    unknown_toggle = 0

    def pick(gender: str) -> str:
        pool = engine.voices[gender]
        voice = pool[used[gender] % len(pool)]
        used[gender] += 1
        return voice

    for name, ch in script.characters.items():
        if ch.voice and ch.voice in {v for pool in engine.voices.values() for v in pool}:
            assignments[name] = ch.voice
            continue
        gender = ch.gender
        if gender == "unknown":
            gender = "male" if unknown_toggle % 2 == 0 else "female"
            unknown_toggle += 1
        assignments[name] = pick(gender)
    for name, voice in assignments.items():
        if name != NARRATOR:
            script.characters.setdefault(name, Character(name=name)).voice = voice
    return assignments


def get_engine(name: str, settings: Settings | None = None, cache: bool = True) -> TTSEngine:
    """name: auto | kokoro | openai | espeak | synthetic."""
    name = name.lower()
    engine: TTSEngine
    if name == "auto":
        for candidate in ("kokoro", "espeak"):
            try:
                return get_engine(candidate, settings, cache)
            except TTSError:
                continue
        log.warning("No real TTS engine available; falling back to the non-speech synthetic engine")
        return get_engine("synthetic", settings, cache)
    if name == "kokoro":
        engine = KokoroEngine()
    elif name == "openai":
        engine = OpenAIEngine(settings.openai_api_key if settings else None)
    elif name == "espeak":
        engine = EspeakEngine()
    elif name == "synthetic":
        engine = SyntheticEngine()
    else:
        raise TTSError(f"unknown TTS engine {name!r}")
    if cache and settings:
        return CachedEngine(engine, settings.cache_dir / "tts")
    return engine


def available_engines(settings: Settings | None = None) -> dict[str, bool]:
    """Which engines can be constructed right now (for the UI dropdown)."""
    result = {}
    for n in ("kokoro", "openai", "espeak", "synthetic"):
        try:
            get_engine(n, settings, cache=False)
            result[n] = True
        except TTSError:
            result[n] = False
    return result
