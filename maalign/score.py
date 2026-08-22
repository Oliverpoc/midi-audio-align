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
    beats_per_bar: int
    quarter_length: float
    source: str
    time_signature_is_trustworthy: bool = True
    warnings: List[str] = field(default_factory=list)

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
    bpb = int(ts[0].numerator) if len(ts) else 4
    trustworthy = len(ts) > 0
    if not trustworthy:
        warnings.append(
            "no time signature found; assuming 4/4. Bar numbers derived from "
            "this score are guesses. Common with downloaded performance MIDI.")

    try:
        n_bars = len(s.parts[0].getElementsByClass("Measure"))
    except Exception:
        n_bars = 0
    ql = float(s.duration.quarterLength)
    if n_bars <= 1:
        n_bars = int(np.ceil(ql / bpb))
        warnings.append("no measure structure in the file; bars inferred from "
                        "total length and the time signature.")

    return Score(notes=notes, n_bars=n_bars, beats_per_bar=bpb,
                 quarter_length=ql, source=path,
                 time_signature_is_trustworthy=trustworthy, warnings=warnings)
