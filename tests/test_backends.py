# -*- coding: utf-8 -*-
"""Backends must be interchangeable behind one API.

The synctoolbox tests skip when it is not installed -- it is an optional extra,
and CI installs only [dev].
"""
import numpy as np
import pytest

from maalign import align, backends
from tests.test_alignment import BASE_QPM, build_time_map, write_test_score
from maalign import synth
from maalign.score import load as load_score

HAS_STB = "synctoolbox" in backends.available()
skip_stb = pytest.mark.skipif(not HAS_STB, reason="synctoolbox not installed")


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    d = tmp_path_factory.mktemp("backends")
    score_path = str(d / "s.mid")
    write_test_score(score_path)
    sc = load_score(score_path)
    gt_q, gt_sec = build_time_map(sc.quarter_length)
    audio = str(d / "p.wav")
    synth.write_wav(audio, synth.render(sc, BASE_QPM,
                                        time_map=lambda q: np.interp(q, gt_q, gt_sec)))
    return score_path, audio, sc, gt_q, gt_sec


def test_chroma_dtw_is_always_available():
    assert "chroma-dtw" in backends.available()


def test_unknown_backend_rejected(pair):
    score_path, audio, *_ = pair
    with pytest.raises(ValueError):
        align(score_path, audio, backend="nope")


def test_backend_is_recorded_on_the_alignment(pair):
    score_path, audio, *_ = pair
    assert align(score_path, audio).backend == "chroma-dtw"


@skip_stb
def test_synctoolbox_backend_matches_ground_truth(pair):
    """Both engines must land near the known rubato curve, and near each other."""
    score_path, audio, sc, gt_q, gt_sec = pair
    q = np.arange(0, sc.quarter_length + 0.5, 0.5)
    truth = np.interp(q, gt_q, gt_sec)

    a = align(score_path, audio, backend="chroma-dtw")
    b = align(score_path, audio, backend="synctoolbox")
    assert b.backend == "synctoolbox"

    err_a = np.abs(a.time_of(q) - truth) * 1000
    err_b = np.abs(b.time_of(q) - truth) * 1000
    between = np.abs(a.time_of(q) - b.time_of(q)) * 1000
    print(f"\n  chroma-dtw  median {np.median(err_a):6.1f} ms")
    print(f"  synctoolbox median {np.median(err_b):6.1f} ms")
    print(f"  between     median {np.median(between):6.1f} ms")

    assert np.median(err_b) < 80.0
    assert np.median(between) < 80.0


@skip_stb
def test_synctoolbox_path_is_monotone(pair):
    score_path, audio, *_ = pair
    al = align(score_path, audio, backend="synctoolbox")
    assert np.all(np.diff(al.quarters) > 0)
    assert np.all(np.diff(al.seconds) >= -1e-9)
