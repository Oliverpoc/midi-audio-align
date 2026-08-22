# -*- coding: utf-8 -*-
"""Dynamic time warping: one coarse global pass, one optional banded refinement.

`coarse` runs librosa's DTW over the whole cost matrix under a Sakoe-Chiba
band. `refine` re-runs DTW at higher time resolution inside a narrow corridor
around that first path -- the multiscale idea from MrMsDTW (Praetzlich, Driedger
& Mueller, 2016). Cost is O(n * band) instead of O(n * m), so 4x resolution is
affordable.

Whether refinement actually helps is a question about your material, not a
given. See README: on a real piano recording it changed the median onset
agreement by 3 ms, which is inside the measurement floor of the onset detector
used to score it. It is off by default for that reason.
"""
from __future__ import annotations

import numpy as np
import librosa

INF = 1e18


def coarse(X: np.ndarray, Y: np.ndarray, sr: int, hop: int,
           band_rad: float = 0.15):
    """Global DTW under a Sakoe-Chiba band. Returns (t_x, t_y) in seconds."""
    _, wp = librosa.sequence.dtw(
        X=X, Y=Y, metric="cosine",
        step_sizes_sigma=np.array([[1, 1], [1, 0], [0, 1]]),
        weights_add=np.array([0, 0, 0]),
        weights_mul=np.array([1, 1, 1]),
        global_constraints=True, band_rad=band_rad)
    wp = wp[::-1]
    tx = librosa.frames_to_time(wp[:, 0], sr=sr, hop_length=hop)
    ty = librosa.frames_to_time(wp[:, 1], sr=sr, hop_length=hop)
    return _monotone(tx, ty)


def refine(Xf: np.ndarray, Yf: np.ndarray, sr: int, hop: int,
           tx: np.ndarray, ty: np.ndarray, radius_s: float = 0.40):
    """Banded DTW at hop `hop`, centred on the path (tx, ty)."""
    n, m = Xf.shape[1], Yf.shape[1]
    fi = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop)
    fj = librosa.frames_to_time(np.arange(m), sr=sr, hop_length=hop)
    centre = np.interp(np.interp(fi, tx, ty), fj, np.arange(m))
    R = max(int(radius_s * sr / hop), 2)

    lo = np.clip((centre - R).astype(int), 0, m - 1)
    hi = np.clip((centre + R).astype(int) + 1, 1, m)
    W = int((hi - lo).max())

    cost = np.full((n, W), INF)
    for i in range(n):
        seg = Yf[:, lo[i]:hi[i]]
        cost[i, :seg.shape[1]] = 1.0 - Xf[:, i] @ seg

    back = np.zeros((n, W), dtype=np.int8)
    acc = np.full((n, W), INF)
    w0 = hi[0] - lo[0]
    acc[0, :w0] = np.cumsum(cost[0, :w0])
    for i in range(1, n):
        w = hi[i] - lo[i]
        shift = lo[i] - lo[i - 1]
        prev = acc[i - 1]
        d_idx = np.arange(w) + shift - 1
        v_idx = np.arange(w) + shift
        diag = np.where((d_idx >= 0) & (d_idx < W), prev[np.clip(d_idx, 0, W - 1)], INF)
        vert = np.where((v_idx >= 0) & (v_idx < W), prev[np.clip(v_idx, 0, W - 1)], INF)
        cur = np.full(W, INF)
        cur[:w] = np.minimum(diag, vert) + cost[i, :w]
        back[i, :w] = (vert < diag).astype(np.int8)
        for j in range(1, w):                       # horizontal step, sequential
            if cur[j - 1] + cost[i, j] < cur[j]:
                cur[j] = cur[j - 1] + cost[i, j]
                back[i, j] = 2
        acc[i] = cur

    i, j = n - 1, int(hi[n - 1] - lo[n - 1]) - 1
    path = []
    while i > 0 or j > 0:
        path.append((i, j + lo[i]))
        b = back[i, j]
        if b == 2 and j > 0:
            j -= 1
        else:
            shift = lo[i] - lo[i - 1] if i > 0 else 0
            j = j + shift - (0 if b == 1 else 1)
            i -= 1
        j = max(j, 0)
        if i < 0:
            break
    path.append((0, lo[0]))
    path = np.array(path[::-1])
    tx2 = librosa.frames_to_time(path[:, 0], sr=sr, hop_length=hop)
    ty2 = librosa.frames_to_time(path[:, 1], sr=sr, hop_length=hop)
    return _monotone(tx2, ty2)


def _monotone(tx: np.ndarray, ty: np.ndarray):
    """Collapse many-to-one steps so tx is strictly increasing."""
    keep = np.concatenate([np.diff(tx) > 0, [True]])
    return tx[keep], ty[keep]
