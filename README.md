# midi-audio-align

[![tests](https://github.com/Oliverpoc/midi-audio-align/actions/workflows/test.yml/badge.svg)](https://github.com/Oliverpoc/midi-audio-align/actions/workflows/test.yml)
[![python](https://img.shields.io/badge/python-3.9%20%7C%203.11%20%7C%203.12-blue)](https://github.com/Oliverpoc/midi-audio-align)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Align a symbolic score — MIDI, MusicXML, Humdrum `**kern` — to a recording of
someone playing it, and get back a map from score position to time in the
recording. Then check whether the map is any good.

```bash
maalign score.mid recording.flac
```

```
  score  : 754 notes, 31 bars (4/4, 4 quarters/bar), 124 quarters
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

Roughly 0.07x realtime on a laptop CPU: a 2-minute recording aligns in about
8 seconds.

**Memory is the real constraint, and it grows with the square of the piece.**
librosa builds the full n x m cost matrix before applying any band, so
`--band` prunes the *search* but does not reduce the *allocation*. At the
default `--hop 512` that is about 28 bytes per cell:

| length | coarse matrix | memory |
|---|---|---|
| 2 min | 5.2k x 5.2k | 0.7 GB |
| 4 min | 10k x 10k | 3.0 GB |
| 10 min | 26k x 26k | 19 GB |
| 20 min | 52k x 52k | 75 GB — will not run |

So past about four minutes, **raise `--hop` and add `--refine`**. Memory falls
with the square of the hop, and the banded refinement pass costs only
O(n x band), so it wins the resolution back. Measured on the Bach example:

| | matrix | onset median |
|---|---|---|
| `--hop 512` | 0.70 GB | 27.3 ms |
| `--hop 2048` | 0.04 GB | 39.6 ms |
| `--hop 2048 --refine` | 0.04 GB | **30.7 ms** |

Seventeen times less memory for the same answer. A 20-minute movement at
`--hop 2048 --refine` needs about 4.7 GB.

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

`maalign.tsm` warps a recording onto any timeline you can express as a time
map -- see [Putting several recordings on one
clock](#putting-several-recordings-on-one-clock).

---

## What comes out

The CLI writes three files into `-o/--outdir`, named after the recording.

**`<name>.alignment.json`** — the machine-readable result.

| key | |
|---|---|
| `quarters`, `seconds` | the map itself: matching arrays of knots. `seconds[i]` is when the performer reached score position `quarters[i]`. Interpolate between them, or use the `Alignment` methods below. |
| `barline_seconds` | every barline, already resolved to recording time |
| `qpm` | the global tempo estimate used for the reference synthesis |
| `time_signature`, `bar_length_quarters`, `n_bars` | `bar_length_quarters` is what you multiply bar numbers by — in 7/8 it is 3.5, not 7 |
| `time_signature_is_trustworthy` | false when the score carried no time signature and 4/4 was assumed, which makes every bar number a guess |
| `warnings` | anything odd found while parsing the score |
| `onset_agreement` | `median_ms`, `p95_ms`, `within_50ms`, `within_100ms`, and the same statistics for the unaligned baseline. Read the two together — see [Does it work?](#does-it-work) |
| `path` | `strictly_monotone`, tempo percentiles, `max_audio_gap_s` (a long stall means audio with no score to match) and `max_score_gap_q` |
| `refined` | whether the second DTW pass ran |

**`<name>.click.wav`** — the recording with a click on every predicted barline.
Suppress with `--no-click`.

**`<name>.path.png`** — warping path and recovered local tempo. Needs
matplotlib; suppress with `--no-plot`.

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

### On dense music, onset agreement stops working entirely

Bach's fugue has about 6 notes per second. Prokofiev's *Precipitato* has 14 —
an onset every ~70 ms — and at that density a prediction lands within 35 ms of
*some* onset by chance. Three different pianists, aligned to the same score:

| | median | baseline |
|---|---|---|
| Pollini | 37.0 ms | 49.0 ms |
| Horowitz 1953 | 36.5 ms | 53.5 ms |
| Sokolov | 37.3 ms | 57.9 ms |

All three alignments were in fact good — verified other ways below — but the
metric has almost no room left to show it. **The denser the music, the less
onset agreement tells you.** Check the gap between the two columns before
believing either.

---

## When you have more than one recording

`cross_recording_agreement` is the check to reach for when onset agreement runs
out of room. It uses no onset detector at all. If two alignments are right then
at the same *score* position both recordings contain the same harmony, so it
samples chroma from each at its own aligned time and compares — scored against
the same statistic at deliberately wrong offsets, which is chance level.

```python
from maalign import align, validate

als = {n: align("score.mxl", f"{n}.mp3")
       for n in ("pollini", "horowitz1953", "sokolov")}
print(validate.cross_recording_agreement(als))
```

```
              pair   aligned    +0.5s    +2.0s    +8.0s
pollini~horowitz1953   0.933    0.852    0.783    0.691
   pollini~sokolov     0.943    0.879    0.793    0.733
horowitz1953~sokolov   0.927    0.860    0.780    0.731

  mean aligned 0.935   mean chance 0.752   lift +0.183
```

**Read the falloff, not the absolute number.** Similarity is capped by how
differently the performers voice the same chords, so it never reaches 1.0. What
proves the alignment is the shape: a sharp peak at zero that has already lost
0.08 by half a second and decays monotonically. A flat profile means the
alignments are not locating the same music.

`by_bar=True` returns the same statistic per bar, which localises weakness:
here it found bars 176–177 (the three pianists end the final chord differently)
and bar 1 (different lead-ins), with a mean of 0.934 and a minimum of 0.857 —
no bar anywhere near the 0.75 chance level, so nothing lost the thread.

---

## Putting several recordings on one clock

With every recording mapped to score position, you can warp them onto a shared
timeline so bar 40 arrives at the same second in all of them — for A/B
listening, for stacking, for driving one set of visuals from any of them.

```python
from maalign import align, tsm
import numpy as np

als = {n: align("score.mxl", f"{n}.mp3") for n in names}
grid = np.linspace(0, als[names[0]].score.quarter_length, 4000)
target = np.mean([a.time_of(grid) for a in als.values()], axis=0)

for n, a in als.items():
    tsm.warp_to_timeline(a, target, f"{n}_ontime.wav", quarters=grid)
```

Two notes from doing this in anger:

**Use the mean, not the median, as the target.** With three performances the
elementwise median is just whichever one sits in the middle at each point — in
our case Horowitz everywhere. That leaves him untouched and time-stretches only
the other two, which is not a fair comparison if the performances are what you
are judging.

**Trim to the score, not to the path.** Live recordings run on into applause,
and warping a span that includes eight seconds of clapping stretches the
clapping too.

`tsm` defaults to WSOLA. See its docstring for why not the phase vocoder.

---

## How to tell whether *your* alignment worked

Three checks that fail in different ways. Use all three; the cheap one first.

**0. If you have a second recording of the same score**, use
[`cross_recording_agreement`](#when-you-have-more-than-one-recording). It is
the strongest of these and the only one that stays sharp on dense music.

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

**A whole movement may not fit in memory.** This is the failure you are most
likely to hit after a structural mismatch, and it looks like a `MemoryError` or
the machine swapping rather than anything musical. The coarse DTW allocates a
full n x m matrix, so cost grows with the *square* of the length: fine at four
minutes, impossible at twenty. Raise `--hop` and add `--refine` — see
[Install](#install) for the measured numbers. `--band` does not help; it prunes
the search, not the allocation.

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

## Compared with synctoolbox

[synctoolbox](https://github.com/meinardmueller/synctoolbox) is the reference
implementation from the group that developed most of the methods this borrows.
Running both on the same material is the closest thing to ground truth
available for a real recording, so here it is.

Prokofiev, Sonata No. 7 Op. 83 III *Precipitato* — 177 bars of 7/8, 3734 notes,
three pianists. Both engines were given the **same** reference audio
(synthesised once from the MusicXML) and the same recordings, so what is
compared is the alignment engine and nothing else.

**Do they agree with each other?**

| | median | p95 | within 50 ms |
|---|---|---|---|
| Pollini | 14.9 ms | 131.6 ms | 86.6% |
| Horowitz 1953 | 17.1 ms | 243.2 ms | 83.0% |
| Sokolov | 27.1 ms | 165.9 ms | 72.6% |

Two independent implementations landing within 15–27 ms of each other is
stronger evidence for both than either one's self-assessment.

**Against synthetic ground truth**, where the true mapping is known exactly
(`tests/test_backends.py`), synctoolbox is slightly more accurate:

| | median error |
|---|---|
| `chroma-dtw` | 11.6 ms |
| `synctoolbox` | **8.9 ms** |

**On real recordings, neither dominates.** synctoolbox wins on the Prokofiev;
the built-in backend wins on the Bach fugue (onset median 27.3 ms against
33.1 ms). Which is ahead depends on the material.

**Are the results equally good?**

| | maalign | synctoolbox |
|---|---|---|
| onset agreement, median | 37.0 / 38.7 / 37.8 ms | **36.7 / 29.8 / 32.7 ms** |
| cross-recording agreement | 0.934 (lift +0.182) | 0.934 (lift +0.183) |
| runtime, one 193 s alignment | **13.6 s** | 16.0 s |
| peak memory, 193 s | 1.95 GB | **0.57 GB** |

By the strong metric they are indistinguishable. On onset agreement synctoolbox
is slightly ahead. Runtime is comparable — an earlier version of this table
claimed synctoolbox was twice as fast, which was an artifact of amortising its
reference-feature extraction across three performances while charging maalign
for a fresh reference each time; measured per single alignment they are within
20% of each other.

**Memory is where the difference is real, and it decides long pieces.**
MrMsDTW is anchor-based and never builds a full cost matrix, so it stays
roughly linear. maalign's coarse pass is quadratic. On a 10-minute input:

| | time | peak memory |
|---|---|---|
| synctoolbox, defaults | 43 s | 1.14 GB |
| maalign, `--hop 2048` | 9 s | 1.18 GB |
| maalign, default `--hop 512` | — | ~17 GB, not attempted |

With the documented `--hop` workaround maalign matches it and is faster. Without
knowing about that flag, a 10-minute movement fails and an 18-minute one is
hopeless. **synctoolbox is safe by default; maalign has to be told.** That is a
robustness difference, not a capability one, and it is the reason to prefer
synctoolbox for whole movements.

**So use synctoolbox** for the alignment itself if you can — it is the mature,
better-tested option, and `backend="synctoolbox"` below wires it in without
giving up the rest of this package. `maalign` is competitive, not better; what
it adds is a small default dependency footprint, the validation tooling in
`maalign.validate`, and the time-stretch in `maalign.tsm`, none of which
synctoolbox provides.

**Where they disagree is interesting.** Thirteen of 177 bars carry 47% of the
total disagreement; across the other 164 the median is 23.3 ms. Boundaries
account for most of it (lead-in silence and the final decay — expected). But a
cluster at m.73–77 shows up in all three performances and is **unexplained**.
Three hypotheses were tested and two failed:

| hypothesis | correlation with disagreement |
|---|---|
| thinner texture is harder to align | +0.120 — rejected, and the sign is backwards |
| texture-change points are harder | −0.163 — rejected |
| where the performers themselves differ most | +0.301 — weakly supported |

Reproduce with `examples/` and the scripts in the companion project; the
comparison is not vendored here because it needs commercial recordings.

---

## API

```python
from maalign import align, load_score, validate, tsm, synth, features, dtw
```

**`align(score_path, audio_path, *, backend="chroma-dtw", refine=False,
band_rad=0.15, qpm=None, sr=22050, hop=512, verbose=False) -> Alignment`**

`backend` selects the engine: `"chroma-dtw"` (built in, no extra dependency,
quadratic memory) or `"synctoolbox"` (needs
`pip install 'midi-audio-align[synctoolbox]'`; memory-safe on long pieces).
`backends.available()` lists what this environment can use. `refine` applies
only to the built-in backend.

```bash
maalign score.mxl recording.mp3 --backend synctoolbox
```

**`Alignment`**

| | |
|---|---|
| `.time_of(quarters)` | score position → seconds in the recording. Accepts scalars or arrays. |
| `.position_of(seconds)` | the inverse |
| `.barlines()` | every barline in recording seconds |
| `.note_times()` | every score note onset in recording seconds |
| `.local_tempo(seconds, window=0.8)` | instantaneous tempo there, quarters/min |
| `.quarters`, `.seconds` | the raw knots |
| `.score`, `.audio_path`, `.audio_duration`, `.qpm`, `.refined` | |

**`load_score(path) -> Score`** — MIDI, MusicXML, `**kern`, MEI, ABC.

| | |
|---|---|
| `.notes` | list of `Note(onset, duration, midi, part)`, onsets in quarter notes |
| `.onsets` | unique onset positions |
| `.n_bars`, `.bar_length_quarters`, `.beats_per_bar`, `.time_signature` | |
| `.quarter_length` | total length in quarters |
| `.time_signature_is_trustworthy`, `.warnings` | |

**`validate`**

| | |
|---|---|
| `onset_agreement(al)` | → `OnsetAgreement`. Floor-limited; see the caveats. |
| `cross_recording_agreement({name: al}, n_samples=1500, offsets=(0.5,2,8), by_bar=False)` | → `CrossAgreement`. The strong check when you have two or more recordings. |
| `click_track(al, out_path, times=None)` | |
| `path_plot(al, out_path)` | needs matplotlib |
| `path_report(al)` | dict of monotonicity and tempo statistics |

**`tsm`** — variable-rate time-stretch.

| | |
|---|---|
| `wsola(y, sr, src_times, dst_times, ...)` | default; mono or `(ch, n)` |
| `phase_vocoder(y, sr, src_times, dst_times, ...)` | mono only; see the module docstring for why it is not the default |
| `time_stretch(y, sr, src, dst, method="wsola")` | dispatcher |
| `stretch_file(in_path, out_path, src, dst, sr=44100, ...)` | |
| `warp_to_timeline(al, target_seconds, out_path, quarters=None)` | warp an aligned recording onto a shared clock |

Lower-level pieces, if you want to build your own pipeline: `features.features`,
`synth.render`, `dtw.coarse`, `dtw.refine`.

---

## Development

```bash
git clone https://github.com/Oliverpoc/midi-audio-align
cd midi-audio-align
pip install -e ".[dev]"
pytest -q
```

The suite synthesizes its own audio from a known rubato curve, so it needs no
downloads and no copyrighted material — which is why accuracy can be asserted
in milliseconds and still run in CI. `pytest -q -s` prints the measured
numbers. `examples/fetch_bach.py` is only needed to try it on real music.

---

## Prior art

This is a small, dependency-light implementation of well-established ideas, not
a new method. If you need the mature toolkit, use
[synctoolbox](https://github.com/meinardmueller/synctoolbox) — see
[Compared with synctoolbox](#compared-with-synctoolbox) for measured numbers.

- Ewert, Müller & Grosche (2009), *High resolution audio synchronization using
  chroma onset features* — the DLNCO idea this borrows.
- Prätzlich, Driedger & Müller (2016), *Memory-restricted multiscale DTW* — the
  coarse-then-banded-refinement structure.
- Müller et al. (2021), *Sync Toolbox* — the reference implementation.
- Verhelst & Roelands (1993), *An overlap-add technique based on waveform
  similarity (WSOLA)* — the time-stretch in `maalign.tsm`.
- Laroche & Dolson (1999), *Improved phase vocoder time-scale modification* —
  the identity phase locking in the fallback method.
- Shan & Tsai (2024), *Just Label the Repeats* — on the repeat problem above.

## License

MIT. See [LICENSE](LICENSE).
