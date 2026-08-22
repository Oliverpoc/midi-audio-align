# -*- coding: utf-8 -*-
"""Ways to find out whether an alignment is actually right.

Against a real recording there is no ground truth, so this module offers three
checks that fail in different ways:

  onset_agreement   objective, but FLOOR-LIMITED. It compares predicted note
                    onsets to onsets found by a detector that has its own error,
                    so the absolute number is not the alignment's error -- it is
                    an upper bound on it. Always read it next to the unaligned
                    baseline, which is what the same statistic looks like with
                    no alignment at all.

  click_track       audible. The recording with a click on every predicted
                    barline. A human ear settles in ten seconds what the number
                    above cannot, and it is the fastest way to catch the failure
                    that matters most: a structural mismatch (the score takes a
                    repeat the performer does not) makes the clicks drift off
                    and never come back.

  path_plot         visual. A healthy warping path is monotone and close to
                    diagonal. Staircases and long flat runs mean the features
                    lost the thread.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import librosa

from .align import Alignment
from .features import SR, HOP


@dataclass
class OnsetAgreement:
    median_ms: float
    p95_ms: float
    within_50ms: float
    within_100ms: float
    baseline_median_ms: float
    baseline_p95_ms: float
    n_detected: int
    n_score_onsets: int

    def __str__(self) -> str:
        return (f"{'':14s}{'median':>9}{'p95':>9}{'<50ms':>8}{'<100ms':>9}\n"
                f"{'no alignment':14s}{self.baseline_median_ms:8.1f}"
                f"{self.baseline_p95_ms:9.1f}{'':8s}{'':9s}\n"
                f"{'aligned':14s}{self.median_ms:8.1f}{self.p95_ms:9.1f}"
                f"{self.within_50ms:7.1f}%{self.within_100ms:8.1f}%")


def _nearest(pred: np.ndarray, det: np.ndarray) -> np.ndarray:
    if len(det) < 2:
        return np.full(len(pred), np.nan)
    idx = np.clip(np.searchsorted(det, pred), 1, len(det) - 1)
    return np.minimum(np.abs(pred - det[idx - 1]),
                      np.abs(pred - det[idx])) * 1000.0


def onset_agreement(alignment: Alignment, sr: int = SR,
                    hop: int = HOP) -> OnsetAgreement:
    y, _ = librosa.load(alignment.audio_path, sr=sr, mono=True)
    det = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop,
                                     units="time", backtrack=True)
    onsets_q = alignment.score.onsets
    d = _nearest(alignment.time_of(onsets_q), det)
    base = _nearest(onsets_q * (60.0 / alignment.qpm), det)
    return OnsetAgreement(
        median_ms=float(np.median(d)), p95_ms=float(np.percentile(d, 95)),
        within_50ms=float(100 * np.mean(d < 50)),
        within_100ms=float(100 * np.mean(d < 100)),
        baseline_median_ms=float(np.median(base)),
        baseline_p95_ms=float(np.percentile(base, 95)),
        n_detected=len(det), n_score_onsets=len(onsets_q))


def click_track(alignment: Alignment, out_path: str, times=None,
                sr: int = SR, click_freq: float = 2200.0,
                music_gain: float = 0.72, click_gain: float = 0.38) -> str:
    """Write the recording with a click on every predicted barline."""
    from scipy.io import wavfile
    y, _ = librosa.load(alignment.audio_path, sr=sr, mono=True)
    if times is None:
        times = alignment.barlines()
    times = np.asarray(times)
    times = times[(times >= 0) & (times < len(y) / sr)]
    clicks = librosa.clicks(times=times, sr=sr, length=len(y),
                            click_freq=click_freq, click_duration=0.035)
    mix = music_gain * y / (np.abs(y).max() + 1e-9) + click_gain * clicks
    wavfile.write(out_path, sr, (np.clip(mix, -1, 1) * 32767).astype(np.int16))
    return out_path


def path_plot(alignment: Alignment, out_path: str) -> str:
    """Warping path plus recovered local tempo. Needs matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), height_ratios=[2, 1])
    ref = alignment.quarters * (60.0 / alignment.qpm)
    a1.plot([0, max(ref.max(), alignment.seconds.max())],
            [0, max(ref.max(), alignment.seconds.max())],
            color="0.8", lw=1, ls="--", label="no alignment (linear)")
    a1.plot(ref, alignment.seconds, lw=1.6, color="#2f6fbf", label="warping path")
    a1.set_xlabel("reference (score) time, s")
    a1.set_ylabel("recording time, s")
    a1.legend(frameon=False, fontsize=9)
    a1.set_title(f"{alignment.score.source} -> {alignment.audio_path}", fontsize=9)

    t = np.linspace(0, alignment.audio_duration, 600)
    a2.plot(t, alignment.local_tempo(t), lw=1.4, color="#c0873a")
    a2.axhline(alignment.qpm, color="0.8", lw=1, ls="--")
    a2.set_xlabel("recording time, s")
    a2.set_ylabel("local tempo, quarter/min")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def path_report(alignment: Alignment, edge_trim: float = 0.03) -> dict:
    """Sanity checks that catch a broken path without needing any audio.

    Tempo is summarised by robust percentiles over the INTERIOR of the piece.
    Using min/max over the whole span produces false alarms every time: a final
    sustained chord, or a rit., legitimately drives instantaneous tempo toward
    zero, and the clipped local_tempo window degenerates at both edges. Those
    are properties of music, not symptoms of a bad alignment.
    """
    d = np.diff(alignment.seconds)
    dur = alignment.audio_duration
    t = np.linspace(dur * edge_trim, dur * (1 - edge_trim), 400)
    tempo = alignment.local_tempo(t)
    p05, p50, p95 = np.nanpercentile(tempo, [5, 50, 95])
    return {
        "strictly_monotone": bool(np.all(d >= -1e-9)),
        "tempo_p05": float(p05),
        "tempo_median": float(p50),
        "tempo_p95": float(p95),
        "tempo_ratio": float(p95 / max(p05, 1e-6)),
        # A large jump in recording time between consecutive path knots means
        # the path stalled in score time (audio present, no score to match) --
        # the signature of material in the recording that the score lacks.
        "max_audio_gap_s": float(d.max()) if len(d) else 0.0,
        "max_score_gap_q": float(np.diff(alignment.quarters).max())
        if len(alignment.quarters) > 1 else 0.0,
    }


# Kept for backwards compatibility with 0.1.0's name.
monotonicity_report = path_report
