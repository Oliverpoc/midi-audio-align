# -*- coding: utf-8 -*-
"""Ground-truth accuracy test.

Against a real recording you cannot measure alignment error, only bound it (see
maalign.validate). Here we can: the test synthesizes a performance from a KNOWN
rubato curve, so the true score-position-to-time mapping is available exactly
and the error is reported in milliseconds.

This needs no downloads and no copyrighted material, so it runs in CI.
"""
import numpy as np
import pytest

from maalign import align, synth
from maalign.score import load as load_score
from maalign.features import SR


BASE_QPM = 96.0


def tempo_curve(q, total_q):
    """A musical-looking rubato: phrase-level ebb, a broadening, a final rit."""
    x = np.asarray(q, dtype=float) / total_q
    t = BASE_QPM
    t = t * (1.0 + 0.08 * np.sin(2 * np.pi * x * 3.0))
    t = t * (1.0 - 0.18 * np.exp(-((x - 0.72) ** 2) / 0.004))
    t = t * (1.0 - 0.35 * np.exp(-((x - 0.99) ** 2) / 0.0006))
    return t


def build_time_map(total_q, step=0.02):
    q = np.arange(0.0, total_q + step, step)
    dt = 60.0 / tempo_curve(q, total_q) * step
    sec = np.concatenate([[0.0], np.cumsum(dt[:-1])])
    return q, sec


def write_test_score(path):
    """A short two-voice piece with enough harmonic motion for chroma to track."""
    from music21 import stream, note as m21note, meter, tempo

    upper = [("C5", 1), ("D5", 1), ("E-5", 2), ("F5", 1), ("G5", 1), ("A-5", 2),
             ("G5", 1), ("F5", 1), ("E-5", 1), ("D5", 1), ("C5", 2),
             ("E-5", 1), ("F5", 1), ("G5", 2), ("B-4", 1), ("A-4", 1), ("G4", 2),
             ("C5", 1), ("E-5", 1), ("G5", 1), ("C6", 1), ("B-5", 2), ("A-5", 2),
             ("G5", 1), ("F5", 1), ("E-5", 1), ("D5", 1), ("C5", 4)]
    lower = [("C3", 2), ("G3", 2), ("A-2", 2), ("E-3", 2),
             ("F2", 2), ("C3", 2), ("G2", 2), ("G3", 2),
             ("C3", 2), ("E-3", 2), ("F3", 2), ("D3", 2),
             ("E-3", 2), ("B-2", 2), ("A-2", 2), ("G2", 2),
             ("C3", 2), ("F3", 2), ("G2", 2), ("C3", 4)]

    s = stream.Score()
    for pitches in (upper, lower):
        p = stream.Part()
        p.append(meter.TimeSignature("4/4"))
        p.append(tempo.MetronomeMark(number=BASE_QPM))
        for name, ql in pitches:
            p.append(m21note.Note(name, quarterLength=ql))
        s.insert(0, p)
    s.write("midi", fp=path)
    return path


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    d = tmp_path_factory.mktemp("maalign")
    score_path = str(d / "test_score.mid")
    write_test_score(score_path)

    sc = load_score(score_path)
    gt_q, gt_sec = build_time_map(sc.quarter_length)
    y = synth.render(sc, BASE_QPM,
                     time_map=lambda q: np.interp(q, gt_q, gt_sec))
    audio_path = str(d / "test_perf.wav")
    synth.write_wav(audio_path, y)
    return score_path, audio_path, sc, gt_q, gt_sec


def _errors(al, sc, gt_q, gt_sec):
    q = np.arange(0, sc.quarter_length + 0.5, 0.5)
    return np.abs(al.time_of(q) - np.interp(q, gt_q, gt_sec)) * 1000.0


def test_score_loads(fixtures):
    _, _, sc, _, _ = fixtures
    assert len(sc) > 40
    assert sc.beats_per_bar == 4
    assert sc.time_signature_is_trustworthy


def test_alignment_beats_the_unaligned_baseline(fixtures):
    score_path, audio_path, sc, gt_q, gt_sec = fixtures
    al = align(score_path, audio_path)
    err = _errors(al, sc, gt_q, gt_sec)

    q = np.arange(0, sc.quarter_length + 0.5, 0.5)
    naive = np.abs(q * (60.0 / al.qpm) - np.interp(q, gt_q, gt_sec)) * 1000.0

    print(f"\n  unaligned median {np.median(naive):7.1f} ms")
    print(f"  aligned   median {np.median(err):7.1f} ms  "
          f"p95 {np.percentile(err, 95):7.1f} ms")

    assert np.median(err) < np.median(naive) / 4
    assert np.median(err) < 60.0
    assert np.mean(err < 100.0) > 0.85


def test_map_is_monotone(fixtures):
    score_path, audio_path, _, _, _ = fixtures
    al = align(score_path, audio_path)
    assert np.all(np.diff(al.seconds) >= -1e-9)
    assert np.all(np.diff(al.quarters) > 0)


def test_roundtrip_position_of_time_of(fixtures):
    score_path, audio_path, sc, _, _ = fixtures
    al = align(score_path, audio_path)
    q = np.linspace(1.0, sc.quarter_length - 1.0, 50)
    assert np.allclose(al.position_of(al.time_of(q)), q, atol=0.25)


def test_barlines_land_inside_the_recording(fixtures):
    score_path, audio_path, _, _, _ = fixtures
    al = align(score_path, audio_path)
    b = al.barlines()
    assert len(b) >= 2
    assert b[0] >= -1e-6
    assert b[-1] <= al.audio_duration + 1e-6
    assert np.all(np.diff(b) > 0)


# --------------------------------------------------------------------------
# Regression: bar length must come from the full time signature, not just the
# numerator. Note onsets are in quarters, so a 7/8 bar is 7*4/8 = 3.5 quarters.
# Using the numerator placed every barline at twice its true position, which
# the alignment itself never notices -- only the click track sounds wrong.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("num,den,expected_bar_q", [
    (4, 4, 4.0), (3, 4, 3.0), (7, 8, 3.5), (6, 8, 3.0),
    (5, 4, 5.0), (12, 8, 6.0), (2, 2, 4.0),
])
def test_bar_length_in_quarters(tmp_path, num, den, expected_bar_q):
    from music21 import stream, note as m21note, meter
    from maalign.score import load as load_score

    n_bars = 6
    p = stream.Part()
    p.append(meter.TimeSignature(f"{num}/{den}"))
    for _ in range(n_bars):
        p.append(m21note.Note("C4", quarterLength=expected_bar_q))
    s = stream.Score()
    s.insert(0, p)
    path = str(tmp_path / f"ts_{num}_{den}.mid")
    s.write("midi", fp=path)

    sc = load_score(path)
    assert sc.bar_length_quarters == pytest.approx(expected_bar_q)
    assert sc.beats_per_bar == num
    assert sc.time_signature == f"{num}/{den}"
    assert sc.n_bars * sc.bar_length_quarters == pytest.approx(
        sc.quarter_length, rel=0.02)


def test_barlines_are_evenly_spaced_in_score_time(fixtures):
    score_path, audio_path, sc, _, _ = fixtures
    al = align(score_path, audio_path)
    q = al.position_of(al.barlines())
    spacing = np.diff(q)
    assert np.allclose(spacing, sc.bar_length_quarters, atol=0.3)
