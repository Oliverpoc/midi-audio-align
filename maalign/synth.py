# -*- coding: utf-8 -*-
"""Render a score to reference audio.

This is what DTW compares the recording against. It does not need to sound
good -- it needs the right pitches at mathematically exact positions, because
all the timing information in the alignment comes from the difference between
this and the real performance.

An additive piano-ish tone is plenty: a handful of harmonics with slight
inharmonicity, an exponential decay that gets faster toward the treble, and a
short filtered-noise hammer transient so the onset channel of the feature
stack has something to lock onto.
"""
from __future__ import annotations

import numpy as np

from .score import Score

SR = 22050


def note(pitch: int, dur: float, vel: int = 90, sr: int = SR,
         n_harmonics: int = 16, inharmonicity: float = 4e-4) -> np.ndarray:
    f0 = 440.0 * 2 ** ((pitch - 69) / 12.0)
    n = max(int(dur * sr), 64)
    t = np.arange(n) / sr
    amp = (vel / 127.0) ** 1.6
    decay = 2.2 + 5.5 * max(0.0, (pitch - 40) / 48.0)
    env = np.exp(-decay * t) * (1.0 - np.exp(-t / 0.0025))

    y = np.zeros(n)
    for k in range(1, n_harmonics + 1):
        fk = f0 * k * np.sqrt(1 + inharmonicity * k * k)
        if fk > sr * 0.45:
            break
        ak = 1.0 / (k ** 1.35) * (0.72 if k % 2 == 0 else 1.0)
        y += ak * np.sin(2 * np.pi * fk * t + k * 0.7)

    nh = int(0.008 * sr)
    rng = np.random.default_rng(pitch * 7919 + int(vel))
    y[:nh] += 0.35 * rng.standard_normal(nh) * np.exp(-np.arange(nh) / (nh / 3.0))
    return y * env * amp * 0.19


def render(score: Score, qpm: float, sr: int = SR,
           time_map=None, velocity: int = 90) -> np.ndarray:
    """Synthesize `score`. `time_map` maps quarters->seconds; default is metronomic."""
    if time_map is None:
        spq = 60.0 / qpm
        def time_map(q):  # noqa: E306
            return np.asarray(q) * spq

    ends = time_map(np.array([n.onset + n.duration for n in score.notes]))
    buf = np.zeros(int((float(ends.max()) + 2.5) * sr))
    starts = time_map(np.array([n.onset for n in score.notes]))
    for n, t0, t1 in zip(score.notes, starts, ends):
        i0 = max(int(float(t0) * sr), 0)
        w = note(n.midi, max(float(t1 - t0), 0.05) + 0.45, velocity, sr=sr)
        end = min(i0 + len(w), len(buf))
        buf[i0:end] += w[:end - i0]

    peak = float(np.abs(buf).max())
    return np.tanh(buf / peak * 1.3) * 0.9 if peak > 0 else buf


def write_wav(path: str, x: np.ndarray, sr: int = SR) -> None:
    from scipy.io import wavfile
    wavfile.write(path, sr, (np.clip(x, -1, 1) * 32767).astype(np.int16))
