# -*- coding: utf-8 -*-
"""Alignment backends.

`chroma-dtw` is this package's own: CENS chroma plus an onset channel, one
banded global DTW, optionally refined. Self-contained, and quadratic in memory.

`synctoolbox` delegates to the reference implementation. Prefer it for whole
movements. MrMsDTW is anchor-based and never materialises a full cost matrix,
so memory stays roughly linear where the built-in backend goes quadratic --
measured on a 10-minute input, 1.14 GB against ~17 GB. The built-in backend can
be brought down to the same footprint with a larger `hop`, but it has to be
told, and being safe by default is worth more than that on long material.

Install with `pip install 'midi-audio-align[synctoolbox]'`. It is optional
because it pulls a considerably larger dependency tree.
"""
from __future__ import annotations

import numpy as np

from . import dtw
from .features import features, SR, HOP

SYNCTOOLBOX_FEATURE_RATE = 50


def available() -> list:
    """Backends importable in this environment."""
    out = ["chroma-dtw"]
    try:
        import synctoolbox  # noqa: F401
        out.append("synctoolbox")
    except ImportError:
        pass
    return out


def chroma_dtw(y_ref, y_perf, *, sr=SR, hop=HOP, band_rad=0.15,
               refine=False, refine_hop=128, refine_radius_s=0.40):
    """Return (t_ref, t_perf) in seconds."""
    X = features(y_ref, sr=sr, hop=hop)
    Y = features(y_perf, sr=sr, hop=hop)
    tx, ty = dtw.coarse(X, Y, sr=sr, hop=hop, band_rad=band_rad)
    if refine:
        Xf = features(y_ref, sr=sr, hop=refine_hop)
        Yf = features(y_perf, sr=sr, hop=refine_hop)
        tx, ty = dtw.refine(Xf, Yf, sr, refine_hop, tx, ty,
                            radius_s=refine_radius_s)
    return tx, ty


def synctoolbox(y_ref, y_perf, *, sr=SR, feature_rate=SYNCTOOLBOX_FEATURE_RATE,
                **_ignored):
    """Delegate to synctoolbox's MrMsDTW. Returns (t_ref, t_perf) in seconds.

    Follows synctoolbox's own sync_audio_audio_full recipe: multi-rate
    filterbank pitch features, quantised chroma, DLNCO onset features, MrMsDTW,
    then a strictly monotonic path.
    """
    try:
        import librosa
        from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
        from synctoolbox.dtw.utils import make_path_strictly_monotonic
        from synctoolbox.feature.chroma import pitch_to_chroma, quantize_chroma
        from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
        from synctoolbox.feature.pitch import audio_to_pitch_features
        from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
    except ImportError as e:                       # pragma: no cover
        raise ImportError(
            "backend='synctoolbox' needs the synctoolbox package: "
            "pip install 'midi-audio-align[synctoolbox]'") from e

    def feats(y):
        tuning = librosa.estimate_tuning(y=y, sr=sr)
        f_chroma = quantize_chroma(f_chroma=pitch_to_chroma(
            f_pitch=audio_to_pitch_features(f_audio=y, Fs=sr,
                                            tuning_offset=tuning,
                                            feature_rate=feature_rate,
                                            verbose=False)))
        f_onset = audio_to_pitch_onset_features(f_audio=y, Fs=sr,
                                                tuning_offset=tuning,
                                                verbose=False)
        f_dlnco = pitch_onset_features_to_DLNCO(
            f_peaks=f_onset, feature_rate=feature_rate,
            feature_sequence_length=f_chroma.shape[1], visualize=False)
        return f_chroma, f_dlnco

    c1, d1 = feats(y_ref)
    c2, d2 = feats(y_perf)
    wp = sync_via_mrmsdtw(f_chroma1=c1, f_onset1=d1, f_chroma2=c2, f_onset2=d2,
                          input_feature_rate=feature_rate, verbose=False)
    wp = make_path_strictly_monotonic(wp)
    tx = wp[0] / feature_rate
    ty = wp[1] / feature_rate
    keep = np.concatenate([np.diff(tx) > 0, [True]])
    return tx[keep], ty[keep]


BACKENDS = {"chroma-dtw": chroma_dtw, "synctoolbox": synctoolbox}


def get(name: str):
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; "
                         f"choose from {sorted(BACKENDS)}")
    return BACKENDS[name]
