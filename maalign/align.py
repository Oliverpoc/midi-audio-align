# -*- coding: utf-8 -*-
"""Top-level API: a score plus a recording in, a time map out."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import librosa

from . import dtw, synth
from .features import features, SR, HOP
from .score import Score, load as load_score


@dataclass
class Alignment:
    """Maps score position (in quarter notes) to time in the recording."""
    score: Score
    quarters: np.ndarray          # knots, in quarters
    seconds: np.ndarray           # the same knots, in recording seconds
    qpm: float                    # global tempo estimate used for the reference
    audio_path: str
    audio_duration: float
    refined: bool = False

    def time_of(self, quarters) -> np.ndarray:
        """Score position -> seconds in the recording."""
        return np.interp(np.asarray(quarters, dtype=float),
                         self.quarters, self.seconds)

    def position_of(self, seconds) -> np.ndarray:
        """Seconds in the recording -> score position in quarters."""
        return np.interp(np.asarray(seconds, dtype=float),
                         self.seconds, self.quarters)

    def barlines(self) -> np.ndarray:
        """Every barline, in recording seconds.

        Uses bar length in QUARTERS, not the time signature numerator -- in 7/8
        a bar is 3.5 quarters, and using 7 would place every other barline.
        """
        return self.time_of(np.arange(0, self.score.n_bars)
                            * self.score.bar_length_quarters)

    def note_times(self) -> np.ndarray:
        return self.time_of([n.onset for n in self.score.notes])

    def local_tempo(self, seconds, window: float = 0.8) -> np.ndarray:
        """Instantaneous tempo (quarters per minute) recovered from the map."""
        t = np.atleast_1d(np.asarray(seconds, dtype=float))
        a = np.clip(t - window, 0, self.audio_duration)
        b = np.clip(t + window, 0, self.audio_duration)
        return (self.position_of(b) - self.position_of(a)) / np.maximum(b - a, 1e-6) * 60.0


def align(score_path: str, audio_path: str, *, refine: bool = False,
          band_rad: float = 0.15, refine_hop: int = 128,
          refine_radius_s: float = 0.40, sr: int = SR, hop: int = HOP,
          qpm: Optional[float] = None, verbose: bool = False) -> Alignment:
    """Align a symbolic score to a recording of it.

    `refine` runs a second, higher-resolution banded pass. It is off by default
    -- see README for why it did not measurably help on real piano audio.
    """
    score = load_score(score_path)
    for w in score.warnings:
        if verbose:
            print(f"  warning: {w}")

    y_perf, _ = librosa.load(audio_path, sr=sr, mono=True)
    duration = len(y_perf) / sr

    # A global tempo estimate keeps the warping path off the matrix corners.
    # DTW absorbs tempo error anyway; this just starts it near the truth.
    if qpm is None:
        qpm = score.quarter_length / duration * 60.0
    y_ref = synth.render(score, qpm, sr=sr)

    if verbose:
        print(f"  score  : {len(score)} notes, {score.n_bars} bars "
              f"({score.time_signature}, {score.bar_length_quarters:g} "
              f"quarters/bar), {score.quarter_length:.0f} quarters")
        print(f"  audio  : {duration:.2f} s")
        print(f"  tempo  : quarter = {qpm:.1f} (global estimate)")

    X = features(y_ref, sr=sr, hop=hop)
    Y = features(y_perf, sr=sr, hop=hop)
    tx, ty = dtw.coarse(X, Y, sr=sr, hop=hop, band_rad=band_rad)

    if refine:
        Xf = features(y_ref, sr=sr, hop=refine_hop)
        Yf = features(y_perf, sr=sr, hop=refine_hop)
        tx, ty = dtw.refine(Xf, Yf, sr, refine_hop, tx, ty,
                            radius_s=refine_radius_s)

    # Reference time is linear in score position, so invert it to get quarters.
    quarters = tx / (60.0 / qpm)
    return Alignment(score=score, quarters=quarters, seconds=ty, qpm=qpm,
                     audio_path=audio_path, audio_duration=duration,
                     refined=refine)
