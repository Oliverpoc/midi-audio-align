# -*- coding: utf-8 -*-
"""Align a symbolic score (MIDI / MusicXML / kern) to a recording of it."""
from .align import align, Alignment
from .score import load as load_score, Score, Note
from . import validate, features, synth, dtw, tsm

__version__ = "0.1.0"
__all__ = ["align", "Alignment", "load_score", "Score", "Note",
           "validate", "features", "synth", "dtw", "tsm"]
