import httpx
import numpy as np
import pytest

from app.audio_io import AudioError, read_wav_bytes, resample, wav_bytes
from app.models import Character, Script
from app.tts_engine import (
    CachedEngine,
    OpenAIEngine,
    SyntheticEngine,
    TTSError,
    available_engines,
    cast_voices,
    get_engine,
)


def test_synthetic_engine_is_deterministic_and_scales_with_text():
    eng = SyntheticEngine()
    a = eng.synthesize("Hello there, my friend.", "syn_m1")
    assert np.array_equal(a, eng.synthesize("Hello there, my friend.", "syn_m1"))
    assert eng.synthesize("A much longer sentence with many more words in it.", "syn_m1").size > a.size
    assert np.isfinite(a).all() and np.abs(a).max() <= 1.0


def test_synthetic_voices_differ_in_pitch():
    eng = SyntheticEngine()

    def peak_hz(voice):
        x = eng.synthesize("aaaaaaaa", voice)
        spec = np.abs(np.fft.rfft(x))
        return np.fft.rfftfreq(x.size, 1 / eng.sample_rate)[spec[:3000].argmax()]

    assert peak_hz("syn_f2") > peak_hz("syn_m3") * 1.5


def test_cache_hits_and_survives_corruption(tmp_path):
    eng = CachedEngine(SyntheticEngine(), tmp_path)
    first = eng.synthesize("cached line", "syn_m1")
    second = eng.synthesize("cached line", "syn_m1")
    assert (eng.misses, eng.hits) == (1, 1)
    assert np.allclose(first, second, atol=1e-3)
    for f in tmp_path.rglob("*.wav"):
        f.write_bytes(b"corrupt")
    eng.synthesize("cached line", "syn_m1")
    assert eng.misses == 2, "a corrupt cache entry is regenerated, not fatal"


def test_cache_key_depends_on_voice_and_emotion(tmp_path):
    eng = CachedEngine(SyntheticEngine(), tmp_path)
    eng.synthesize("same text", "syn_m1")
    eng.synthesize("same text", "syn_f1")
    eng.synthesize("same text", "syn_m1", "sad")
    assert eng.misses == 3


def test_cast_voices_is_stable_gendered_and_unique():
    script = Script(
        title="t",
        chapters=[],
        characters={
            "Tom": Character(name="Tom", gender="male"),
            "Bob": Character(name="Bob", gender="male"),
            "Mary": Character(name="Mary", gender="female"),
            "Zed": Character(name="Zed"),
        },
    )
    eng = SyntheticEngine()
    voices = cast_voices(script, eng)
    assert voices["Narrator"] == "syn_narrator"
    assert voices["Tom"] in eng.voices["male"] and voices["Mary"] in eng.voices["female"]
    assert voices["Tom"] != voices["Bob"]
    assert script.characters["Tom"].voice == voices["Tom"], "cast is recorded on the script for the UI"
    assert cast_voices(script, eng) == voices, "re-casting an already cast script changes nothing"


def test_openai_engine_parses_wav_and_sends_emotion():
    sent = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent["auth"] = request.headers["authorization"]
        sent["body"] = request.read().decode()
        return httpx.Response(200, content=wav_bytes(np.full(2400, 0.1, dtype=np.float32), 24000))

    eng = OpenAIEngine("sk-test", client=httpx.Client(transport=httpx.MockTransport(handler)))
    audio = eng.synthesize("hi", "nova", "whisper")
    assert audio.size == 2400 and sent["auth"] == "Bearer sk-test"
    assert "whispering" in sent["body"] and '"voice":"nova"' in sent["body"].replace(" ", "")


def test_openai_engine_errors_are_ttserrors():
    eng = OpenAIEngine(
        "k", client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(429, text="slow down")))
    )
    with pytest.raises(TTSError, match="429"):
        eng.synthesize("hi", "nova")
    with pytest.raises(TTSError, match="OPENAI_API_KEY"):
        OpenAIEngine(None)


def test_get_engine_auto_always_returns_something(settings):
    assert get_engine("auto", settings).name in {"kokoro", "espeak", "synthetic"}
    with pytest.raises(TTSError):
        get_engine("nope", settings)
    assert available_engines(settings)["synthetic"] is True


def test_audio_io_roundtrip_and_resample():
    x = (0.5 * np.sin(np.linspace(0, 100, 4800))).astype(np.float32)
    y, sr = read_wav_bytes(wav_bytes(x, 24000))
    assert sr == 24000 and np.allclose(x, y, atol=1e-3)
    assert resample(y, 24000, 12000).size == 2400
    with pytest.raises(AudioError):
        read_wav_bytes(b"nope")
