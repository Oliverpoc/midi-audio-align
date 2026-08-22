# -*- coding: utf-8 -*-
"""Read a symbolic score into a plain note list.

Accepts anything music21 parses: MIDI (.mid), MusicXML (.musicxml/.mxl),
Humdrum (.krn), MEI, ABC.

A caution about MIDI specifically. Two very different things arrive with that
extension:

  * a QUANTISED score MIDI, exported from notation. Its time axis *is* score
    time, so bar and beat positions come for free.
  * a PERFORMANCE MIDI, captured from someone playing. Its time axis is that
    player's rubato, not score time. Alignment still works -- DTW does not care
    which side carries the rubato -- but `beats_per_bar` and bar numbers are
    only meaningful if the file has sane tempo and time-signature meta events,
    which downloaded performance MIDI very often does not.

`Score.time_signature_is_trustworthy` flags the obvious failure so you find out
before the bar numbers in your output turn out to be fiction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Note:
    onset: float          # in quarter notes from the start
    duration: float       # in quarter notes
    midi: int
    part: int


@dataclass
class Score:
    notes: List[Note]
    n_bars: int
    beats_per_bar: int          # the time signature's numerator, for display
    quarter_length: float
    source: str
    # Note onsets are measured in QUARTER notes, so a bar is only
    # `beats_per_bar` long when the denominator is 4. In 7/8 a bar is
    # 7 * 4/8 = 3.5 quarters. Getting this wrong silently doubles or halves
    # every barline, which the alignment itself never notices.
    bar_length_quarters: float = 4.0
    time_signature_denominator: int = 4
    time_signature_is_trustworthy: bool = True
    warnings: List[str] = field(default_factory=list)

    @property
    def time_signature(self) -> str:
        return f"{self.beats_per_bar}/{self.time_signature_denominator}"

    @property
    def onsets(self) -> np.ndarray:
        return np.unique([n.onset for n in self.notes])

    def __len__(self) -> int:
        return len(self.notes)


def load(path: str) -> Score:
    from music21 import converter

    s = converter.parse(path).stripTies()
    notes, warnings = [], []
    for pi, part in enumerate(s.parts if s.parts else [s]):
        for n in part.flatten().notes:
            pitches = n.pitches if n.isChord else [n.pitch]
            for p in pitches:
                notes.append(Note(float(n.offset), float(n.quarterLength),
                                  p.midi, pi))
    notes.sort(key=lambda n: (n.onset, n.midi))
    if not notes:
        raise ValueError(f"no notes found in {path}")

    ts = s.flatten().getElementsByClass("TimeSignature")
    trustworthy = len(ts) > 0
    num = int(ts[0].numerator) if trustworthy else 4
    den = int(ts[0].denominator) if trustworthy else 4
    bar_q = num * 4.0 / den
    if not trustworthy:
        warnings.append(
            "no time signature found; assuming 4/4. Bar numbers derived from "
            "this score are guesses. Common with downloaded performance MIDI.")
    if len({(t.numerator, t.denominator) for t in ts}) > 1:
        warnings.append(
            "the score changes time signature; bar positions are computed from "
            f"the first one ({num}/{den}) and will drift after the change.")

    try:
        n_bars = len(s.parts[0].getElementsByClass("Measure"))
    except Exception:
        n_bars = 0
    ql = float(s.duration.quarterLength)
    if n_bars <= 1:
        n_bars = int(np.ceil(ql / bar_q))
        warnings.append("no measure structure in the file; bars inferred from "
                        "total length and the time signature.")

    return Score(notes=notes, n_bars=n_bars, beats_per_bar=num,
                 quarter_length=ql, source=path,
                 bar_length_quarters=bar_q, time_signature_denominator=den,
                 time_signature_is_trustworthy=trustworthy, warnings=warnings)
