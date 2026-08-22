# -*- coding: utf-8 -*-
"""Variable-rate time-scale modification.

An alignment tells you *when* each performance plays each bar. Acting on that
-- putting several recordings on one clock so they can be compared, or fitting
a performance to a fixed timeline -- needs a time-stretch whose rate changes
continuously. `librosa.effects.time_stretch` takes a single scalar, so this
module drives the stretch from an arbitrary time map instead.

Two algorithms, and the choice matters more than it looks:

  wsola (default)   Copies real waveform segments and overlap-adds them,
                    choosing each segment by cross-correlating against the
                    natural continuation of the previous one. Local waveform
                    shape survives, so attacks stay sharp.

  phase_vocoder     Resynthesizes from accumulated phase. Smooth on sustained
                    tone, but it smears transients, and for piano that is
                    exactly the wrong trade. Measured on a Prokofiev toccata it
                    kept 73% of onset sharpness at a rate of 1.09 where WSOLA
                    kept 100%, and it was audibly bad. Kept for reference and
                    for material where it genuinely suits.

If a rubberband binary is available, it will beat both of these; this module
exists so the package has no non-Python dependency.
"""
from __future__ import annotations

import numpy as np
import librosa


# --------------------------------------------------------------------- WSOLA
def wsola(y: np.ndarray, sr: int, src_times, dst_times, frame: int = 2048,
          hop_syn: int = 512, tol: int = 512,
          centre_bias: float = 0.02) -> np.ndarray:
    """Warp `y` so the moment at src_times[i] lands at dst_times[i].

    `y` may be mono (n,) or multichannel (ch, n). Channels are displaced by the
    SAME offsets, computed once on their mean: running the search per channel
    would decorrelate them and collapse the stereo image.

    `centre_bias` mildly prefers matches near the requested position so that
    ties do not accumulate into a slow drift away from the time map.
    """
    from scipy.signal import correlate

    src = np.asarray(src_times, dtype=float)
    dst = np.asarray(dst_times, dtype=float)
    multi = y.ndim > 1
    ref = y.mean(axis=0) if multi else y
    n_in = ref.shape[-1]
    if n_in < frame + 2:
        raise ValueError("input shorter than one analysis frame")

    n_out = int(float(dst[-1]) * sr)
    n_frames = n_out // hop_syn + 1
    t_out = np.arange(n_frames) * hop_syn / sr
    pos = np.clip((np.interp(t_out, dst, src) * sr).astype(int),
                  0, n_in - frame - 1)

    win = np.hanning(frame + 1)[:frame]
    overlap = frame - hop_syn
    ch = y.shape[0] if multi else 1
    out = np.zeros((ch, n_out + frame))
    norm = np.zeros(n_out + frame)

    # Sliding energy, for NORMALISED cross-correlation. Raw correlation is
    # biased toward loud regions, so its argmax drifts off the true match --
    # badly enough that even a rate of exactly 1.0 stopped being a copy.
    csum = np.concatenate([[0.0], np.cumsum(ref.astype(np.float64) ** 2)])
    tmpl_energy_floor = 1e-12

    last = int(pos[0])
    for m in range(n_frames):
        target = int(pos[m])
        if m == 0:
            best = target
        else:
            ts = last + hop_syn                      # natural continuation
            tmpl = ref[ts:ts + overlap]
            lo = max(target - tol, 0)
            hi = min(target + tol + overlap, n_in)
            region = ref[lo:hi]
            L = len(tmpl)
            if L < 8 or len(region) < L + 1:
                best = target
            else:
                cc = correlate(region, tmpl, mode="valid", method="fft")
                starts = lo + np.arange(len(cc))
                energy = csum[np.minimum(starts + L, n_in)] - csum[starts]
                ncc = cc / np.sqrt(np.maximum(energy, tmpl_energy_floor)
                                   * max(float(tmpl @ tmpl), tmpl_energy_floor))
                if centre_bias:
                    ncc = ncc - centre_bias * np.abs(starts - target) / max(tol, 1)
                best = int(starts[int(np.argmax(ncc))])
        best = int(np.clip(best, 0, n_in - frame - 1))

        seg = y[:, best:best + frame] if multi else ref[best:best + frame][None, :]
        k = m * hop_syn
        out[:, k:k + frame] += seg * win
        norm[k:k + frame] += win
        last = best

    norm[norm < 1e-6] = 1.0
    z = (out / norm)[:, :n_out]
    return z if multi else z[0]


# ------------------------------------------------------------- phase vocoder
def _peak_map(mag_col: np.ndarray) -> np.ndarray:
    n = len(mag_col)
    peaks = np.flatnonzero((mag_col[1:-1] > mag_col[:-2]) &
                           (mag_col[1:-1] >= mag_col[2:])) + 1
    if peaks.size == 0:
        return np.arange(n)
    idx = np.searchsorted(peaks, np.arange(n))
    lo = np.clip(idx - 1, 0, peaks.size - 1)
    hi = np.clip(idx, 0, peaks.size - 1)
    return np.where(np.abs(np.arange(n) - peaks[lo]) <=
                    np.abs(np.arange(n) - peaks[hi]), peaks[lo], peaks[hi])


def phase_vocoder(y: np.ndarray, sr: int, src_times, dst_times,
                  n_fft: int = 2048, hop: int = 512,
                  phase_lock: bool = True) -> np.ndarray:
    """Variable-rate phase vocoder with identity phase locking. Mono only.

    Prefer `wsola` for anything percussive. See the module docstring.
    """
    if y.ndim > 1:
        raise ValueError("phase_vocoder takes mono; call it per channel")
    src = np.asarray(src_times, dtype=float)
    dst = np.asarray(dst_times, dtype=float)

    D = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    mag, phase = np.abs(D), np.angle(D)
    n_bins, n_frames = D.shape

    n_out = int(np.ceil(float(dst[-1]) * sr / hop)) + 1
    t_out = np.arange(n_out) * hop / sr
    pos = np.clip(np.interp(t_out, dst, src) * sr / hop, 0, n_frames - 1.001)

    omega = 2.0 * np.pi * hop * np.arange(n_bins) / n_fft
    out = np.zeros((n_bins, n_out), dtype=np.complex128)
    acc = phase[:, int(pos[0])].copy()

    for i in range(n_out):
        p = pos[i]
        k = int(p)
        k1 = min(k + 1, n_frames - 1)
        frac = p - k
        m = (1.0 - frac) * mag[:, k] + frac * mag[:, k1]
        if phase_lock:
            owner = _peak_map(m)
            out[:, i] = m * np.exp(1j * (acc[owner] +
                                         phase[:, k] - phase[:, k][owner]))
        else:
            out[:, i] = m * np.exp(1j * acc)
        if i + 1 < n_out:
            dphi = phase[:, k1] - phase[:, k] - omega
            dphi -= 2.0 * np.pi * np.round(dphi / (2.0 * np.pi))
            acc = acc + (pos[i + 1] - p) * (omega + dphi)

    return librosa.istft(out, hop_length=hop, n_fft=n_fft)


# -------------------------------------------------------------------- facade
def time_stretch(y: np.ndarray, sr: int, src_times, dst_times,
                 method: str = "wsola", **kw) -> np.ndarray:
    if method == "wsola":
        return wsola(y, sr, src_times, dst_times, **kw)
    if method == "phase_vocoder":
        if y.ndim > 1:
            chans = [phase_vocoder(c, sr, src_times, dst_times, **kw) for c in y]
            n = min(len(c) for c in chans)
            return np.stack([c[:n] for c in chans], axis=0)
        return phase_vocoder(y, sr, src_times, dst_times, **kw)
    raise ValueError(f"unknown method {method!r}")


def stretch_file(in_path: str, out_path: str, src_times, dst_times,
                 sr: int = 44100, mono: bool = False,
                 method: str = "wsola", peak: float = 0.94, **kw) -> str:
    """Load, warp, normalise, write a 16-bit wav."""
    from scipy.io import wavfile

    y, _ = librosa.load(in_path, sr=sr, mono=mono)
    z = time_stretch(y, sr, src_times, dst_times, method=method, **kw)
    p = float(np.abs(z).max())
    if p > 0:
        z = z / p * peak
    data = z.T if z.ndim > 1 else z
    wavfile.write(out_path, sr, (np.clip(data, -1, 1) * 32767).astype(np.int16))
    return out_path


def warp_to_timeline(alignment, target_seconds, out_path: str,
                     quarters=None, sr: int = 44100, **kw) -> str:
    """Warp an aligned recording onto `target_seconds`.

    `target_seconds` is the wanted arrival time of each score position in
    `quarters` (default: the alignment's own knots). The output starts at the
    first sampled score position, so several recordings warped to the same
    target begin together and stay together.
    """
    q = alignment.quarters if quarters is None else np.asarray(quarters, float)
    src = alignment.time_of(q)
    dst = np.asarray(target_seconds, dtype=float)
    dst = dst - dst[0]
    return stretch_file(alignment.audio_path, out_path, src, dst, sr=sr, **kw)
