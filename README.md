# midi-audio-align

Align a symbolic score — MIDI, MusicXML, Humdrum `**kern` — to a recording of
someone playing it, and get back a map from score position to time in the
recording. Then check whether the map is any good.

```bash
maalign score.mid recording.flac
```

```
  score  : 754 notes, 31 bars, 124 quarters
  audio  : 115.93 s
  tempo  : quarter = 64.2 (global estimate)
  path   : monotone=True  tempo 33-75 qpm (median 71, ratio 2.30)

                 median      p95   <50ms   <100ms
no alignment      73.3    923.8
aligned           27.3    135.9   73.0%    92.9%

map    -> out/recording.alignment.json
click  -> out/recording.click.wav
plot   -> out/recording.path.png
```

Written for driving score-synchronised visuals, where every bar has to land on
the frame it belongs to. It works equally well for building training data,
comparing performances, or score following after the fact.

---

## Install

```bash
pip install git+https://github.com/Oliverpoc/midi-audio-align
```

Python ≥3.9. Pulls numpy, scipy, librosa and music21. `pip install
'midi-audio-align[plot]'` adds matplotlib for the diagnostic plot.

---

## Use it

**Command line**

```bash
maalign score.mid recording.flac -o out/
maalign score.musicxml recording.wav --refine     # second, finer DTW pass
maalign score.krn recording.mp3 --qpm 72          # override tempo estimate
```

**Python**

```python
from maalign import align, validate

al = align("score.mid", "recording.flac")

al.time_of(48.0)        # score quarter 48 -> seconds in the recording
al.position_of(31.7)    # 31.7 s into the recording -> score quarter
al.barlines()           # every barline, in recording seconds
al.local_tempo(31.7)    # instantaneous tempo there, quarters/min

print(validate.onset_agreement(al))
validate.click_track(al, "check.wav")
```

---

## Try it on real music

```bash
python examples/fetch_bach.py
maalign assets/score/wtc1f02.krn assets/audio/bwv847_fugue.flac
```

Downloads Bach's Fugue in C minor, BWV 847, played by Kimiko Ishizaka — a real
performance with real rubato, released CC0 by the *Open Well-Tempered Clavier*
project. Nothing is committed to this repository; see
[Example assets](#example-assets) for why the score half in particular is
fetched rather than vendored.

---

## Does it work?

Two different answers, because they are answerable to different degrees.

### On synthetic audio, where ground truth exists

`tests/test_alignment.py` renders a performance from a *known* rubato curve, so
the true mapping is available exactly and error is measurable in milliseconds.
No downloads, so it runs in CI.

| | median | p95 |
|---|---|---|
| no alignment | 1022.4 ms | — |
| aligned | **11.6 ms** | 99.5 ms |

### On a real recording, where it does not

There is no ground truth for a human performance, so the number below is **not
the alignment's error**. It is the distance from each predicted onset to the
nearest onset a detector finds in the audio — and that detector has error of
its own. Read it as an upper bound, and always next to the unaligned baseline.

Bach BWV 847, Ishizaka, 754 notes over 116 s:

| | median | p95 | <50 ms | <100 ms |
|---|---|---|---|---|
| no alignment | 73.3 ms | 923.8 ms | 34.1% | 68.1% |
| aligned | **27.3 ms** | 135.9 ms | 73.0% | 92.9% |
| aligned, `--refine` | 30.7 ms | 113.4 ms | 69.6% | 92.9% |

**That third row is the interesting one.** `--refine` re-runs DTW at four times
the time resolution inside a narrow corridor around the first path. It moved
the median by 3 ms — nothing. Quadrupling temporal resolution and seeing no
change means the residual is not timing resolution; it is the measuring
instrument. The true error is somewhere at or below ~27 ms and this metric
cannot see it.

Which is why `--refine` is off by default, and why the click track exists.

---

## How to tell whether *your* alignment worked

Three checks that fail in different ways. Use all three; the cheap one first.

**1. Listen to the click track.** `out/<name>.click.wav` is your recording with
a click on every predicted barline. Ten seconds of listening settles what the
statistics cannot. This is the fastest way to catch the failure that matters
most — see below.

**2. Read the path summary.** A healthy path is monotone with tempo variation
in the range ordinary rubato produces. The CLI warns when the spread is wide or
the path stalls.

**3. Look at the plot.** `out/<name>.path.png` draws the warping path and the
recovered local tempo. Near-diagonal and smooth is good. Staircases, long flat
runs, or a tempo trace that leaps between wildly different values means the
features lost the thread somewhere.

---

## How it fails

**Structural mismatch is the failure that will actually bite you.** If the
score takes a repeat the performer does not — or the performer takes a cut, or
plays a different edition — DTW cannot recover. It is not that error grows; the
path commits to a wrong branch and everything after it is wrong. A handful of
wrong notes is fine, but a missing sixteen bars is not.

Symptom: the clicks track correctly, then drift off at some point and never
come back. Fix: edit the symbolic score so its repeat structure matches the
performance, and run again. (There is research on doing this automatically —
Shan & Tsai, *Just Label the Repeats*, ISMIR 2024 — not implemented here.)

**Long sustained notes with extreme rubato.** Final chords under a big
ritardando are where the path is loosest — there is little spectral change for
the features to lock onto. Expect the worst residuals at the very end.

**Transposition** breaks it. Chroma is octave-invariant, so octave differences
cost nothing, but a performance in a different key will not align.

---

## A caution about MIDI

Two very different things arrive with a `.mid` extension.

A **quantised score MIDI**, exported from notation software, has score time as
its time axis. Bar and beat positions come for free and everything downstream
works.

A **performance MIDI**, captured from someone playing, has *that player's*
rubato as its time axis. Alignment still works — DTW does not care which side
carries the rubato, and if anything the path is closer to diagonal. But bar
numbers are only meaningful if the file carries sane tempo and time-signature
meta events, and downloaded performance MIDI very often does not. `maalign`
warns when a score has no time signature, because the bar numbers in your
output would otherwise be quiet fiction.

MIDI also loses slurs, dynamics, real voice separation, and note spelling. None
of that affects alignment, which needs only pitches and approximate timing. It
costs you downstream, in anything that wants phrase boundaries or voices.

---

## How it works

1. **Synthesize the score** to reference audio — additive tone, a few
   harmonics, slight inharmonicity, a filtered-noise hammer transient. It does
   not need to sound good. It needs the right pitches at exact positions.
2. **Extract features** from both signals.
3. **DTW** under a Sakoe-Chiba band, cosine distance.
4. **Invert**: reference time is linear in score position, so the warping path
   becomes score-position-to-recording-time.

The feature choice is the part that decides whether any of this works.

Plain chroma is not enough. Through a repeating figure — an ostinato, an
Alberti bass, a toccata pattern — harmony barely changes, chroma is nearly
constant, and DTW has no gradient to follow: whole bars are equally good
matches and the path drifts. So a second channel carries note *attacks*:
half-wave-rectified temporal difference of raw chroma, smeared forward with an
exponential decay. That is a simplified DLNCO feature, and it is the difference
between an alignment that holds through repetitive material and one that does
not.

---

## Example assets

`examples/fetch_bach.py` downloads at run time and commits nothing. The two
halves have different licences and only one is freely redistributable.

**Recording** — Kimiko Ishizaka, *Open Well-Tempered Clavier* (2015), released
**CC0 / Public Domain Mark**. Genuinely free; kept out of git only because a
31 MB FLAC does not belong in a source repository.
[archive.org](https://archive.org/details/bach-well-tempered-clavier-book-1)

**Score** — Humdrum `**kern` from
[humdrum-tools/bach-wtc](https://github.com/humdrum-tools/bach-wtc). Bach's
music is public domain, but the *encoding* carries `Copyright 1994, David
Huron` and *"Rights to all derivative electronic formats reserved"*. Fetching
it for your own analysis is ordinary scholarly use of KernScores. Redistributing
it, or publishing files derived from it, is not — so this project does neither.

If you need an unencumbered score for the same music, the Open Well-Tempered
Clavier project also released the notation CC0, at
[musescore.com/openscore](https://musescore.com/openscore).

---

## Prior art

This is a small, dependency-light implementation of well-established ideas, not
a new method. If you need the mature toolkit, use
[synctoolbox](https://github.com/meinardmueller/synctoolbox).

- Ewert, Müller & Grosche (2009), *High resolution audio synchronization using
  chroma onset features* — the DLNCO idea this borrows.
- Prätzlich, Driedger & Müller (2016), *Memory-restricted multiscale DTW* — the
  coarse-then-banded-refinement structure.
- Müller et al. (2021), *Sync Toolbox* — the reference implementation.
- Shan & Tsai (2024), *Just Label the Repeats* — on the repeat problem above.

## License

MIT. See [LICENSE](LICENSE).
