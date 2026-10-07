import numpy as np

from app import mixer
from app.audio_io import SAMPLE_RATE as SR
from app.mixer import Clip


def tone(seconds: float, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def test_assemble_speech_inserts_pauses():
    out = mixer.assemble_speech([Clip(tone(1.0), 0.5), Clip(tone(1.0), 0.25)])
    assert abs(out.size / SR - 2.75) < 0.01


def test_scene_without_ambience_has_no_preroll():
    audio, pre, tail = mixer.mix_scene([Clip(tone(1.0), 0.0)], [])
    assert pre == 0.0 and tail == 0.0
    assert np.allclose(audio[:SR], tone(1.0))


def test_ambience_can_be_disabled():
    audio, pre, _ = mixer.mix_scene([Clip(tone(1.0), 0.0)], ["rain"], with_ambience=False)
    assert pre == 0.0
    assert np.allclose(audio[:SR], tone(1.0))


def test_bed_is_quieter_than_speech_and_ducks_while_talking():
    # 2s of silence-pause then speech: compare bed level before and during speech
    clips = [Clip(np.zeros(SR * 3, dtype=np.float32), 0.0), Clip(tone(4.0, 0.0001), 0.0)]
    quiet_audio, pre, _ = mixer.mix_scene(clips[:1], ["rain"])
    bed_open = rms(quiet_audio[int(pre * SR) + SR : int(pre * SR) + 2 * SR])

    speech = Clip(tone(6.0, 0.3), 0.0)
    audio, pre, _ = mixer.mix_scene([speech], ["rain"])
    mid = audio[int(pre * SR) + 2 * SR : int(pre * SR) + 4 * SR]
    only_speech = tone(6.0, 0.3)[2 * SR : 4 * SR]
    bed_ducked = rms(mid - only_speech)

    assert bed_open > 0
    assert bed_ducked < bed_open * 0.6, "bed should drop (>4 dB) under speech"
    assert rms(only_speech) > 8 * bed_ducked, "speech stays clearly on top"


def test_preroll_and_tail_fade_the_bed():
    audio, pre, tail = mixer.mix_scene([Clip(tone(2.0), 0.0)], ["wind"])
    assert abs(audio[0]) < 1e-3, "starts from silence"
    assert abs(audio[-1]) < 1e-3, "ends in silence"
    assert pre > 0 and tail > pre


def test_stitch_overlaps_tail_with_next_preroll():
    a = (np.ones(SR * 4, dtype=np.float32) * 0.1, 1.0, 2.0)
    b = (np.ones(SR * 4, dtype=np.float32) * 0.1, 1.0, 2.0)
    out = mixer.stitch([a, b])
    assert out.size == 8 * SR - SR  # 1s overlap = min(prev tail, next preroll)


def test_stitch_handles_empty_and_single():
    assert mixer.stitch([]).size == 0
    a = (tone(1.0), 0.0, 0.0)
    assert mixer.stitch([a]).size == SR


def test_master_normalises_peak_and_removes_dc():
    out = mixer.master(tone(1.0, 0.05) + 0.2)
    assert abs(np.max(np.abs(out)) - 10 ** (-1 / 20)) < 1e-3
    assert abs(np.mean(out)) < 1e-3


def test_envelope_tracks_speech():
    sig = np.concatenate([np.zeros(SR * 2, dtype=np.float32), tone(2.0), np.zeros(SR * 2, dtype=np.float32)])
    env = mixer.speech_envelope(sig)
    assert env[SR // 2] < 0.05 and env[int(3 * SR)] > 0.95 and env[int(5.5 * SR)] < 0.05
