# -*- coding: utf-8 -*-
"""Time-stretch tests.

The identity test is the one that matters. Stretching by exactly 1.0 must
return the input, and an early WSOLA here did not: it used raw cross-correlation,
whose argmax is pulled toward loud regions rather than the true match, so
segments were picked slightly off and a "no-op" quietly rewrote the waveform.
Normalised cross-correlation fixed it, and this test is what would have caught
it immediately.
"""
import numpy as np
import pytest

from maalign import tsm


SR = 22050


def transient_signal(seconds=6.0, sr=SR, bpm=140):
    """Decaying plucks on a steady grid -- sharp attacks, tonal bodies."""
    n = int(seconds * sr)
    y = np.zeros(n)
    rng = np.random.default_rng(0)
    step = int(60.0 / bpm * sr)
    for i, start in enumerate(range(0, n - sr // 2, step)):
        f0 = 220.0 * 2 ** ((i % 5) / 12.0)
        t = np.arange(sr // 2) / sr
        env = np.exp(-9.0 * t) * (1 - np.exp(-t / 0.0015))
        tone = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in (1, 2, 3, 4))
        tone[:64] += 0.5 * rng.standard_normal(64)
        y[start:start + len(tone)] += tone * env
    return y / (np.abs(y).max() + 1e-9) * 0.9


def snr_db(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    return 10 * np.log10(np.sum(a ** 2) / max(np.sum((a - b) ** 2), 1e-12))


def peakiness(x, sr=SR):
    import librosa
    o = librosa.onset.onset_strength(y=x, sr=sr)
    return float(o.std() / max(o.mean(), 1e-9))


@pytest.fixture(scope="module")
def sig():
    return transient_signal()


def test_wsola_identity_is_a_copy(sig):
    src = np.array([0.0, len(sig) / SR])
    z = tsm.wsola(sig, SR, src, src)
    assert snr_db(sig, z) > 15.0


def test_wsola_hits_the_requested_duration(sig):
    src = np.array([0.0, len(sig) / SR])
    for rate in (0.75, 1.0, 1.3):
        z = tsm.wsola(sig, SR, src, src * rate)
        assert len(z) / SR == pytest.approx(len(sig) / SR * rate, abs=0.05)


def test_wsola_preserves_transients_better_than_the_phase_vocoder(sig):
    src = np.array([0.0, len(sig) / SR])
    base = peakiness(sig)
    for rate in (0.88, 1.09):
        dst = src * rate
        p_w = peakiness(tsm.wsola(sig, SR, src, dst))
        p_v = peakiness(tsm.phase_vocoder(sig, SR, src, dst))
        assert p_w > p_v
        assert p_w > 0.85 * base          # near-transparent at musical rates


def test_variable_rate_follows_the_map(sig):
    """A rate that changes mid-file must place events where the map says."""
    dur = len(sig) / SR
    src = np.array([0.0, dur / 2, dur])
    dst = np.array([0.0, dur / 2 * 1.5, dur / 2 * 1.5 + dur / 2 * 0.7])
    z = tsm.wsola(sig, SR, src, dst)
    assert len(z) / SR == pytest.approx(dst[-1], abs=0.05)

    import librosa
    on_src = librosa.onset.onset_detect(y=sig, sr=SR, units="time")
    on_dst = librosa.onset.onset_detect(y=z, sr=SR, units="time")
    expected = np.interp(on_src, src, dst)
    matched = [np.min(np.abs(on_dst - e)) for e in expected
               if e < dst[-1] - 0.25]
    assert np.median(matched) < 0.06


def test_stereo_channels_stay_coherent(sig):
    """Both channels must be displaced identically or the image collapses.

    The inter-channel delay is the thing to check. Search only small lags: the
    test signal is a steady pulse train, so a full-range correlation peaks just
    as happily one whole beat away.
    """
    from scipy.signal import correlate

    delay = 7
    y = np.stack([sig, np.roll(sig, delay)])
    src = np.array([0.0, len(sig) / SR])
    z = tsm.wsola(y, SR, src, src * 1.15)
    assert z.shape[0] == 2

    w, start = 16384, 22050
    a, b = z[0][start:start + w], z[1][start:start + w]
    cc = correlate(a, b, mode="full")
    lags = np.arange(-(len(b) - 1), len(a))
    near = np.abs(lags) <= 64
    found = int(lags[near][int(np.argmax(cc[near]))])
    assert abs(abs(found) - delay) <= 2


def test_unknown_method_rejected(sig):
    src = np.array([0.0, len(sig) / SR])
    with pytest.raises(ValueError):
        tsm.time_stretch(sig, SR, src, src, method="nope")
