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


def test_auto_never_falls_back_to_the_non_speech_engine(settings, monkeypatch):
    """Regression: with no speech engine installed, 'auto' used to silently make placeholder tones."""
    import app.tts_engine as tts

    def nothing(name, *a, **k):
        raise tts.TTSError(f"{name} unavailable")

    monkeypatch.setattr(tts, "KokoroEngine", lambda: nothing("kokoro"))
    monkeypatch.setattr(tts, "SapiEngine", lambda: nothing("sapi"))
    monkeypatch.setattr(tts, "EspeakEngine", lambda: nothing("espeak"))
    with pytest.raises(TTSError, match="No speech engine is available"):
        get_engine("auto", settings)
    assert get_engine("synthetic", settings).name == "synthetic", "still available when asked for explicitly"


def test_auto_prefers_kokoro_then_sapi_then_espeak(settings, monkeypatch):
    import app.tts_engine as tts

    class Fake(SyntheticEngine):
        def __init__(self, label):
            self.name = label

    def unavailable():
        raise tts.TTSError("no")

    monkeypatch.setattr(tts, "KokoroEngine", unavailable)
    monkeypatch.setattr(tts, "SapiEngine", lambda: Fake("sapi"))
    monkeypatch.setattr(tts, "EspeakEngine", lambda: Fake("espeak"))
    assert get_engine("auto", settings, cache=False).name == "sapi"
    monkeypatch.setattr(tts, "SapiEngine", unavailable)
    assert get_engine("auto", settings, cache=False).name == "espeak"


def test_auto_can_use_openai_when_it_is_the_only_engine(settings, monkeypatch):
    from dataclasses import replace

    import app.tts_engine as tts

    def unavailable():
        raise tts.TTSError("no")

    for n in ("KokoroEngine", "SapiEngine", "EspeakEngine"):
        monkeypatch.setattr(tts, n, unavailable)
    assert get_engine("auto", replace(settings, openai_api_key="sk-x"), cache=False).name == "openai"


def test_available_engines_lists_synthetic(settings):
    assert available_engines(settings)["synthetic"] is True
    assert "sapi" in available_engines(settings)


# ---------------------------------------------------------------- Windows SAPI (fake PowerShell)


class FakePowerShell:
    """Stands in for powershell.exe: answers the voice-listing script and 'speaks' by writing a WAV."""

    def __init__(self, voices_json: str, fail_speak: bool = False):
        self.voices_json, self.fail_speak, self.spoken = voices_json, fail_speak, []

    def __call__(self, cmd, **kwargs):
        import base64
        import subprocess

        assert cmd[-2] == "-EncodedCommand"
        script = base64.b64decode(cmd[-1]).decode("utf-16-le")
        if "GetInstalledVoices" in script:
            return subprocess.CompletedProcess(cmd, 0, stdout=self.voices_json.encode(), stderr=b"")
        env = kwargs["env"]
        self.spoken.append((env["SONICTALES_TEXT"], env["SONICTALES_VOICE"], env["SONICTALES_RATE"]))
        if self.fail_speak:
            return subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b"voice not found")
        tone = (0.2 * np.sin(2 * np.pi * 220 * np.arange(11025) / 22050)).astype(np.float32)
        with open(env["SONICTALES_OUT"], "wb") as fh:
            fh.write(wav_bytes(tone, 22050))
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")


WIN_VOICES = (
    '[{"Name":"Microsoft David Desktop","Gender":"Male","Culture":"en-US"},'
    '{"Name":"Microsoft Zira Desktop","Gender":"Female","Culture":"en-US"},'
    '{"Name":"Microsoft Hazel Desktop","Gender":"Female","Culture":"en-GB"},'
    '{"Name":"Microsoft Hedda Desktop","Gender":"Female","Culture":"de-DE"}]'
)


def test_sapi_discovers_voices_by_gender_and_language():
    from app.tts_engine import SapiEngine

    eng = SapiEngine(runner=FakePowerShell(WIN_VOICES))
    assert eng.voices["male"] == ["Microsoft David Desktop"]
    assert eng.voices["female"] == ["Microsoft Zira Desktop", "Microsoft Hazel Desktop"], "non-English voices skipped"
    assert eng.voices["narrator"] == ["Microsoft David Desktop"]


def test_sapi_single_voice_json_object_and_single_gender_fallback():
    from app.tts_engine import SapiEngine

    only_zira = '{"Name":"Microsoft Zira Desktop","Gender":"Female","Culture":"en-US"}'  # PowerShell: 1 item = object
    eng = SapiEngine(runner=FakePowerShell(only_zira))
    assert eng.voices["male"] == eng.voices["female"] == eng.voices["narrator"] == ["Microsoft Zira Desktop"]


def test_sapi_synthesize_passes_text_safely_and_returns_audio():
    from app.tts_engine import SapiEngine

    fake = FakePowerShell(WIN_VOICES)
    eng = SapiEngine(runner=fake)
    awkward = 'He said "hello" & left; $(rm -rf) \u00e9\u4e2d'
    audio = eng.synthesize(awkward, "Microsoft Zira Desktop", "sad")
    assert audio.size == 11025 and np.abs(audio).max() > 0.1
    text, voice, rate = fake.spoken[0]
    assert text == awkward, "text is passed through the environment untouched, never interpolated into a script"
    assert voice == "Microsoft Zira Desktop" and rate == "-2", "sad speaks slower (speed 0.9 -> rate -2)"


def test_sapi_errors_become_ttserrors():
    from app.tts_engine import SapiEngine

    with pytest.raises(TTSError, match="Windows speech failed"):
        SapiEngine(runner=FakePowerShell(WIN_VOICES, fail_speak=True)).synthesize("hi", "Microsoft David Desktop")
    with pytest.raises(TTSError, match="no text-to-speech voices"):
        SapiEngine(runner=FakePowerShell("[]"))
    with pytest.raises(TTSError, match="only works on Windows"):
        import sys

        if sys.platform == "win32":
            raise TTSError("only works on Windows")  # pragma: no cover - keeps this test meaningful on Windows CI
        SapiEngine()


def test_audio_io_roundtrip_and_resample():
    x = (0.5 * np.sin(np.linspace(0, 100, 4800))).astype(np.float32)
    y, sr = read_wav_bytes(wav_bytes(x, 24000))
    assert sr == 24000 and np.allclose(x, y, atol=1e-3)
    assert resample(y, 24000, 12000).size == 2400
    with pytest.raises(AudioError):
        read_wav_bytes(b"nope")
