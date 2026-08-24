# -*- coding: utf-8 -*-
"""Command line interface: `maalign <score> <recording>`."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from .align import align
from . import validate


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="maalign",
        description="Align a symbolic score (MIDI / MusicXML / kern) to a "
                    "recording of it, and check the result.")
    p.add_argument("score", help="MIDI, MusicXML, Humdrum kern, MEI or ABC")
    p.add_argument("audio", help="wav / flac / mp3 / m4a")
    p.add_argument("-o", "--outdir", default="out", help="output directory")
    p.add_argument("--backend", default="chroma-dtw",
                   help="alignment engine: 'chroma-dtw' (built in) or "
                        "'synctoolbox' (needs the extra; memory-safe on long "
                        "pieces)")
    p.add_argument("--refine", action="store_true",
                   help="second high-resolution banded DTW pass "
                        "(see README: rarely helps on real audio)")
    p.add_argument("--qpm", type=float, default=None,
                   help="override the global tempo estimate, quarters/min")
    p.add_argument("--band", type=float, default=0.15,
                   help="Sakoe-Chiba band radius, fraction of sequence length. "
                        "Prunes the search; does NOT reduce memory")
    p.add_argument("--hop", type=int, default=512,
                   help="STFT hop for the coarse pass. Memory scales as "
                        "1/hop^2 -- raise it for long pieces (see README) and "
                        "add --refine to win the resolution back")
    p.add_argument("--no-click", action="store_true", help="skip the click track")
    p.add_argument("--no-plot", action="store_true", help="skip the path plot")
    p.add_argument("-q", "--quiet", action="store_true")
    a = p.parse_args(argv)

    v = not a.quiet
    os.makedirs(a.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.audio))[0]

    if v:
        print("aligning ...")
    al = align(a.score, a.audio, backend=a.backend, refine=a.refine,
               band_rad=a.band, hop=a.hop, qpm=a.qpm, verbose=v)

    if not al.score.time_signature_is_trustworthy and v:
        print("  ! no time signature in the score -- bar numbers are guesses")

    rep = validate.path_report(al)
    if v:
        print(f"  path   : monotone={rep['strictly_monotone']}  "
              f"tempo {rep['tempo_p05']:.0f}-{rep['tempo_p95']:.0f} qpm "
              f"(median {rep['tempo_median']:.0f}, ratio {rep['tempo_ratio']:.2f})")
        if rep["tempo_ratio"] > 2.5 or rep["max_audio_gap_s"] > 3.0:
            print("  ! the path is not behaving like ordinary rubato "
                  "(wide tempo spread, or a long stall).")
            print("    Most often this means the score and the performance "
                  "disagree structurally -- a repeat taken on one")
            print("    side only, a cut, a different edition. "
                  "Listen to the click track.")

    oa = validate.onset_agreement(al)
    if v:
        print(f"\ndetected {oa.n_detected} onsets in the recording; "
              f"{oa.n_score_onsets} distinct score onsets\n")
        print(oa)
        print("\nnote: onset agreement is floor-limited by the onset detector.")
        print("      It bounds the alignment error from above; it is not it.")

    out = {
        "score": a.score, "audio": a.audio, "refined": al.refined,
        "backend": al.backend,
        "qpm": al.qpm, "audio_duration": al.audio_duration,
        "time_signature": al.score.time_signature,
        "bar_length_quarters": al.score.bar_length_quarters,
        "n_bars": al.score.n_bars,
        "time_signature_is_trustworthy": al.score.time_signature_is_trustworthy,
        "warnings": al.score.warnings,
        "quarters": al.quarters.tolist(), "seconds": al.seconds.tolist(),
        "barline_seconds": al.barlines().tolist(),
        "onset_agreement": oa.__dict__, "path": rep,
    }
    jf = os.path.join(a.outdir, f"{stem}.alignment.json")
    with open(jf, "w") as f:
        json.dump(out, f)
    if v:
        print(f"\nmap    -> {jf}")

    if not a.no_click:
        cf = validate.click_track(al, os.path.join(a.outdir, f"{stem}.click.wav"))
        if v:
            print(f"click  -> {cf}")
            print("          clicks should land on every downbeat. If they "
                  "drift off and stay off,\n"
                  "          the score and the performance disagree "
                  "structurally -- check repeats.")
    if not a.no_plot:
        try:
            pf = validate.path_plot(al, os.path.join(a.outdir, f"{stem}.path.png"))
            if v:
                print(f"plot   -> {pf}")
        except ImportError:
            if v:
                print("plot   -- skipped (matplotlib not installed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
