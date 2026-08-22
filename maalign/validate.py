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


# ---------------------------------------------------------------------------
# Cross-recording agreement.
#
# onset_agreement above has a blind spot that only shows up on dense music.
# Its baseline is "predict with no alignment at all", and on a piece with ~14
# notes per second -- onsets every ~70 ms -- even a wrong prediction lands
# within 35 ms of *some* onset by chance. Measured on Prokofiev's Precipitato
# the aligned median was 37 ms against an unaligned baseline of 49 ms: a real
# improvement, but far too little separation to conclude anything from.
#
# When you have two or more recordings of the same score, this is the check to
# use instead. It needs no onset detector. If the alignments are right then at
# the same SCORE position every recording must contain the same harmony, so
# sample chroma from each at its own aligned time and compare. Scored against
# the same statistic at deliberately wrong offsets, which is chance level.
# ---------------------------------------------------------------------------

@dataclass
class CrossAgreement:
    pairs: dict                 # "a~b" -> similarity at the aligned positions
    offset_pairs: dict          # offset seconds -> {"a~b": similarity}
    mean_aligned: float
    mean_chance: float
    by_bar: np.ndarray = None   # mean pairwise agreement per bar, if requested

    @property
    def lift(self) -> float:
        return self.mean_aligned - self.mean_chance

    def __str__(self) -> str:
        offs = sorted(self.offset_pairs)
        head = f"{'pair':>18}{'aligned':>10}" + "".join(f"{'+'+str(o)+'s':>9}" for o in offs)
        rows = [head]
        for k in self.pairs:
            rows.append(f"{k:>18}{self.pairs[k]:10.3f}"
                        + "".join(f"{self.offset_pairs[o][k]:9.3f}" for o in offs))
        rows.append(f"\n  mean aligned {self.mean_aligned:.3f}   "
                    f"mean chance {self.mean_chance:.3f}   lift {self.lift:+.3f}")
        return "\n".join(rows)


def cross_recording_agreement(alignments: dict, n_samples: int = 1500,
                              offsets=(0.5, 2.0, 8.0), by_bar: bool = False,
                              sr: int = SR, hop: int = HOP) -> CrossAgreement:
    """Compare recordings of one score to each other through their alignments.

    `alignments` maps a name to an Alignment. All must be of the same score.

    A correct set of alignments shows high similarity at offset 0 that falls
    off steeply within half a second. A flat profile means the alignments are
    not actually locating the same music.

    Similarity is bounded above by how differently the performers voice the
    same chords, so it will not reach 1.0 even when the alignment is perfect.
    Read the falloff, not the absolute number.
    """
    import itertools

    names = list(alignments)
    if len(names) < 2:
        raise ValueError("need at least two alignments")

    chroma, scores = {}, []
    for n in names:
        al = alignments[n]
        y, _ = librosa.load(al.audio_path, sr=sr, mono=True)
        C = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop)
        chroma[n] = C / (np.linalg.norm(C, axis=0, keepdims=True) + 1e-9)
        scores.append(al.score.quarter_length)

    q_end = min(scores)
    grid = np.linspace(q_end * 0.02, q_end * 0.98, n_samples)

    def vecs(n, q, shift=0.0):
        t = alignments[n].time_of(q) + shift
        f = np.clip((t * sr / hop).astype(int), 0, chroma[n].shape[1] - 1)
        return chroma[n][:, f]

    def pairwise(q, shift=0.0):
        return {f"{a}~{b}": float(np.mean(np.sum(vecs(a, q) * vecs(b, q, shift), axis=0)))
                for a, b in itertools.combinations(names, 2)}

    aligned = pairwise(grid)
    off = {o: pairwise(grid, o) for o in offsets}
    chance = [np.mean(list(off[o].values())) for o in offsets if o >= 2.0] or \
             [np.mean(list(off[o].values())) for o in offsets]

    per_bar = None
    if by_bar:
        al0 = alignments[names[0]]
        bar_q = al0.score.bar_length_quarters
        nb = al0.score.n_bars
        per_bar = np.array([
            np.mean(list(pairwise(np.linspace(b * bar_q, (b + 1) * bar_q, 12)).values()))
            for b in range(nb)])

    return CrossAgreement(pairs=aligned, offset_pairs=off,
                          mean_aligned=float(np.mean(list(aligned.values()))),
                          mean_chance=float(np.mean(chance)), by_bar=per_bar)
