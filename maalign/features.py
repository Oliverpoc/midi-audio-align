# -*- coding: utf-8 -*-
"""Chroma features for score-to-audio alignment.

The feature choice is the part that decides whether this works at all.

Plain chroma is not enough. During a repeating figure -- an ostinato, an
Alberti bass, a toccata pattern -- the harmony barely changes, so the chroma
vector is nearly constant and DTW has no gradient to follow. Whole bars become
equally good matches and the warping path drifts.

The fix is to add a channel that carries note *attacks*: half-wave-rectify the
temporal difference of raw chroma, then smear it forward with an exponential
decay. That is a simplified DLNCO feature (Ewert, Mueller & Grosche, 2009),
which is what synctoolbox uses for the same reason.
"""
from __future__ import annotations

import numpy as np
import librosa

SR = 22050
HOP = 512


def chroma_onset(y: np.ndarray, sr: int = SR, hop: int = HOP,
                 decay: float = 0.62) -> np.ndarray:
    """Onset-weighted chroma: where in pitch class did energy just *start*."""
    C = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    D = np.maximum(0.0, np.diff(C, axis=1, prepend=C[:, :1]))
    for t in range(1, D.shape[1]):          # exponential forward smear
        D[:, t] = np.maximum(D[:, t], D[:, t - 1] * decay)
    return D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-6)


def features(y: np.ndarray, sr: int = SR, hop: int = HOP,
             w_chroma: float = 1.0, w_onset: float = 1.0) -> np.ndarray:
    """Stack normalised CENS chroma with the onset channel, column-normalised.

    Returns a (24, n_frames) array ready for cosine-distance DTW.
    """
    C = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop)
    D = chroma_onset(y, sr=sr, hop=hop)
    F = np.vstack([w_chroma * C, w_onset * D])
    return F / (np.linalg.norm(F, axis=0, keepdims=True) + 1e-9)


def load_features(path: str, sr: int = SR, hop: int = HOP, **kw):
    """Load an audio file and return (features, waveform)."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return features(y, sr=sr, hop=hop, **kw), y
