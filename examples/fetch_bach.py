# -*- coding: utf-8 -*-
"""Download the example pair: Bach WTC I, Fugue No. 2 in C minor, BWV 847.

Nothing this script downloads is committed to the repository, and that is
deliberate -- the two files have different licences and only one of them is
freely redistributable:

  RECORDING   Kimiko Ishizaka, "Open Well-Tempered Clavier" (2015).
              Released CC0 / Public Domain Mark. Genuinely free; kept out of
              git only because a 31 MB FLAC does not belong in a source repo.
              https://archive.org/details/bach-well-tempered-clavier-book-1

  SCORE       Humdrum **kern encoding from humdrum-tools/bach-wtc.
              Bach's music is public domain, but the ENCODING carries
              "Copyright 1994, David Huron" and "Rights to all derivative
              electronic formats reserved". Fetching it for your own analysis
              is ordinary scholarly use of KernScores; redistributing it, or
              publishing files derived from it, is not something this project
              will do for you.

If you need an unencumbered score for the same music, the Open Well-Tempered
Clavier project also released the notation under CC0 (musescore.com/openscore),
from the same recording sessions.

Usage:  python examples/fetch_bach.py [--outdir assets]
"""
from __future__ import annotations

import argparse
import os
import urllib.parse
import urllib.request

IA_ITEM = "bach-well-tempered-clavier-book-1"
IA_BASE = f"https://archive.org/download/{IA_ITEM}/"
TRACKS = {
    "prelude": "Kimiko Ishizaka - Bach- Well-Tempered Clavier, Book 1 - "
               "03 Prelude No. 2 in C minor, BWV 847.flac",
    "fugue":   "Kimiko Ishizaka - Bach- Well-Tempered Clavier, Book 1 - "
               "04 Fugue No. 2 in C minor, BWV 847.flac",
}
KERN_BASE = "https://raw.githubusercontent.com/humdrum-tools/bach-wtc/master/kern/"
KERN = {"prelude": "wtc1p02.krn", "fugue": "wtc1f02.krn"}


def _get(url: str, dest: str, retries: int = 3) -> str:
    if os.path.exists(dest) and os.path.getsize(dest) > 2048:
        print(f"  have  {os.path.basename(dest)} "
              f"({os.path.getsize(dest)/1e6:.1f} MB)")
        return dest
    last = None
    for k in range(retries):
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  got   {os.path.basename(dest)} "
                  f"({os.path.getsize(dest)/1e6:.1f} MB)")
            return dest
        except Exception as e:                       # IA 502s under load
            last = e
            print(f"  retry {k+1}/{retries}: {e}")
    raise SystemExit(f"failed to download {url}: {last}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="assets")
    p.add_argument("--which", choices=["fugue", "prelude", "both"],
                   default="fugue")
    a = p.parse_args()

    audio_dir = os.path.join(a.outdir, "audio")
    score_dir = os.path.join(a.outdir, "score")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(score_dir, exist_ok=True)

    which = ["fugue", "prelude"] if a.which == "both" else [a.which]

    print("recording  Kimiko Ishizaka, Open Well-Tempered Clavier -- CC0")
    for w in which:
        _get(IA_BASE + urllib.parse.quote(TRACKS[w]),
             os.path.join(audio_dir, f"bwv847_{w}.flac"))

    print("score      humdrum-tools/bach-wtc -- encoding (c) 1994 David Huron;")
    print("           scholarly use only, do not redistribute or commit")
    for w in which:
        _get(KERN_BASE + KERN[w], os.path.join(score_dir, KERN[w]))

    w0 = which[0]
    print("\nnow run:")
    print(f"  maalign {score_dir}/{KERN[w0]} {audio_dir}/bwv847_{w0}.flac")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
