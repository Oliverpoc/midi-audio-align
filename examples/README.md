# Examples

`fetch_bach.py` downloads Bach's Prelude and Fugue No. 2 in C minor, BWV 847 —
Kimiko Ishizaka's recording and a Humdrum encoding of the score.

```bash
python examples/fetch_bach.py                # fugue only
python examples/fetch_bach.py --which both   # prelude too
maalign assets/score/wtc1f02.krn assets/audio/bwv847_fugue.flac
```

Nothing downloaded here is committed, and `assets/` is gitignored. The recording
is CC0; the Humdrum encoding is not freely redistributable. See the licence
notes at the top of `fetch_bach.py` and in the main README.
