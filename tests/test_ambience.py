from pathlib import Path

import numpy as np
import pytest

from app import ambience
from app.audio_io import SAMPLE_RATE, write_wav

SR = SAMPLE_RATE


@pytest.mark.parametrize("tag", ambience.known_tags())
def test_every_generator_makes_clean_audio(tag):
    seconds = ambience.LOOP_SECONDS  # a full loop: sparse sounds like thunder are silent between events
    bed = ambience.render_bed([tag], seconds)
    assert bed.shape == (int(seconds * SR),)
    assert np.isfinite(bed).all()
    assert 0.003 < np.sqrt(np.mean(bed**2)) < 0.3, "audible but not blasting"
    assert np.max(np.abs(bed)) <= 0.91


@pytest.mark.parametrize("tag", ambience.known_tags())
def test_loops_are_seamless(tag):
    loop = ambience._loop(tag, SR, str(ambience.ASSET_DIR))
    seam = abs(float(loop[0] - loop[-1]))
    assert seam <= np.abs(np.diff(loop)).max(), "the wrap-around step must look like any other step"


def test_beds_are_deterministic():
    assert np.array_equal(ambience.render_bed(["rain", "night"], 2.0), ambience.render_bed(["rain", "night"], 2.0))


def test_different_tags_sound_different():
    rain, fire = ambience.render_bed(["rain"], 4.0), ambience.render_bed(["fire"], 4.0)
    assert abs(np.corrcoef(rain, fire)[0, 1]) < 0.2

    # rain is brighter than fire: more energy in the high band
    def hi_ratio(x):
        spec = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(x.size, 1 / SR)
        return spec[freqs > 3000].sum() / spec.sum()

    assert hi_ratio(rain) > hi_ratio(fire)


def test_empty_and_unknown_tags_are_silent():
    assert not ambience.render_bed([], 1.0).any()
    assert not ambience.render_bed(["definitely-not-a-sound"], 1.0).any()


def test_layers_mix_without_clipping():
    bed = ambience.render_bed(ambience.known_tags(), 3.0)
    assert np.max(np.abs(bed)) <= 0.91


def test_custom_asset_overrides_generator(tmp_path: Path):
    t = np.arange(SR * 2) / SR
    write_wav(tmp_path / "rain.wav", (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32))
    ambience._loop.cache_clear()
    bed = ambience.render_bed(["rain"], 3.0, asset_dir=tmp_path)
    spectrum = np.abs(np.fft.rfft(bed))
    peak_hz = np.fft.rfftfreq(bed.size, 1 / SR)[spectrum.argmax()]
    assert abs(peak_hz - 440) < 5, "the supplied recording, not the generator, should be heard"
    ambience._loop.cache_clear()
