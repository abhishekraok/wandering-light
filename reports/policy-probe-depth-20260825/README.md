# Does the front/back asymmetry persist inward?

`reports/policy-probe-20260824/` measured the two ends of a program and found
they disagree. At the front a dict on `(input type, output type)` matched a
trained encoder; at the back the encoder nearly doubled top-1. That asymmetry
is the argument for bidirectional search in #33 — but only if it survives more
than one step. The cheap check, run before building anything:

```
uv run python experiments/policy_probe_depth.py <corpus> --depths=0,1,2
```

**It does not survive. The asymmetry is a first-step artifact of the corpus's
root types, and learnability at the front goes *up* one step in, not down.**

## How a sub-task is peeled

A witness `w_1..w_d` certified shortest gives an exact sub-task at each end:

    front, depth k:  (s_k, output)      distance d-k, predict its FIRST action
    back,  depth k:  (input, s_{d-k})   distance d-k, predict its LAST action

`s_j` is the witness replayed `j` steps; the distance is exact by the prefix
argument (see `reports/value-probe-20260825/`). Peeling the *target* end for the
back is what makes the two comparable — a second-to-last action must produce
`s_{d-1}`, so masking it on the original output type would mask the wrong
field. At `k = 0` both reduce exactly to the published probes.

Every predictor and the scorer are imported from `policy_probe` unchanged. All
depths are scored on the same task population (`d >= 3`), so a difference
between rows is depth and not task length.

## Result

    Depth policy probe -- witness-derived labels, tasks with d >= 3

    probe          predictor                 top-1  mean rank
    front k=0      type mask                 0.0%       9.50
    front k=0      conditional lookup       39.7%       3.38
    front k=0      learned (conv)           38.9%       3.28
    front k=1      type mask                 0.0%      18.41
    front k=1      conditional lookup       26.1%       4.79
    front k=1      learned (conv)           32.0%       4.04
    front k=2      type mask                 0.0%      19.90
    front k=2      conditional lookup       29.3%       4.37
    front k=2      learned (conv)           37.0%       3.58
    back  k=0      type mask                 0.0%      25.87
    back  k=0      conditional lookup       21.8%       4.48
    back  k=0      learned (conv)           40.9%       3.38
    back  k=1      type mask                 0.0%      26.71
    back  k=1      conditional lookup       23.0%       4.83
    back  k=1      learned (conv)           41.4%       3.74
    back  k=2      type mask                 0.3%      26.87
    back  k=2      conditional lookup       23.7%       4.97
    back  k=2      learned (conv)           43.2%       3.84

What the encoder buys over the dict, which is the comparison that decides #33:

| probe | mask width | dict rank / top-1 | conv rank / top-1 | rank gain | top-1 gain |
|---|---|---|---|---|---|
| front k=0 | 9.50 | 3.38 / 39.7% | 3.28 / 38.9% | **3%** | **−2%** |
| front k=1 | 18.41 | 4.79 / 26.1% | 4.04 / 32.0% | 16% | +23% |
| front k=2 | 19.90 | 4.37 / 29.3% | 3.58 / 37.0% | 18% | +26% |
| back k=0 | 25.87 | 4.48 / 21.8% | 3.38 / 40.9% | 25% | +88% |
| back k=1 | 26.71 | 4.83 / 23.0% | 3.74 / 41.4% | 23% | +80% |
| back k=2 | 26.87 | 4.97 / 23.7% | 3.84 / 43.2% | 23% | +82% |

## The mechanism: the front's type mask only works at step 1

    front k=0: mean compatible  9.50/118  (6.88 -> 2.90 bits)  types: bytes 12%, dict 11%, range 8%, set 8%
    front k=1: mean compatible 18.41/118  (6.88 -> 4.07 bits)  types: int 51%, float 18%, str 12%, list 10%
    front k=2: mean compatible 19.90/118  (6.88 -> 4.24 bits)  types: int 55%, float 24%, str 14%, bool 4%
    back  k=0: mean compatible 25.87/118  (6.88 -> 4.56 bits)  types: int 47%, float 24%, str 22%, list 6%
    back  k=1: mean compatible 26.71/118  (6.88 -> 4.61 bits)  types: int 51%, float 29%, str 16%, list 3%
    back  k=2: mean compatible 26.87/118  (6.88 -> 4.60 bits)  types: int 55%, float 29%, str 13%, bool 3%

`deep-corpus-v1`'s `deep_root_types` is `bool, list, tuple, set, dict, bytes,
bytearray, complex, range` — it **excludes `int`, `str` and `float`**. So a
task's input type is always one of nine unusual types, and few functions
consume those: 9.50 of 118. One op later the state distribution has collapsed
onto `int`/`float`/`str` (51/18/12%), which is what most functions consume, and
the mask widens to 18.41. By `k = 2` it has settled at 19.90 — close to the
20.52 applicable-functions figure already on file for the basis as a whole.

The back never has this advantage and never loses it: 25.87 → 26.71 → 26.87.

So masking is worth **3.98 bits at the first step and ~2.7 bits everywhere
else** at the front, and a flat ~2.3 bits at the back.

## What this means for #33

**"BFS ordered by a lookup table" is not enough.** That option was on the table
because the dict matched the encoder at the front. It matches at exactly one
position — `k = 0`, the only position where the mask does the work. At the
other five measured positions the encoder beats the dict by 16–25% on rank and
23–88% on top-1.

**Bidirectional search is not specially motivated by learnability.** The back is
not uniquely learnable; it is *uniformly* learnable, and the front joins it one
step in. The back keeps a real but smaller edge — top-1 41–43% against 32–37%,
not the ~2x that `k = 0` implied. If bidirectional search is built, it should be
for the branching-factor argument, not this one.

**A forward policy head is worth more than the first-action probe suggested.**
Its apparent uselessness there was the type mask doing the work for free, and
the mask stops doing that after one step.

## Calibration: witness labels are a subset, and it costs little

Only the two outermost actions ship a certified-complete optimal set. Deeper
in, all that is known is that the witness's op is optimal — so labels here are
a subset, which can only *understate* a predictor. Sizing that before reading
any `k = 1` number:

* On test, the complete optimal set is a **singleton for 93.0% of first actions
  and 86.2% of last ones**, and the witness op is in the set **2905/2905** times
  at both ends. So a witness label is exactly right most of the time and never
  wrong, only incomplete.
* Rerunning `k = 0` with witness labels (`policy_probe_depth_d2.json`, `d >= 2`)
  gives front dict 3.36 / conv 3.11 against the published 3.18 / 2.86, and back
  dict 4.73 / conv 3.55 against the published 4.80 / 3.43.

Every predictor moves in the same direction by a similar amount, so **within-run
relative comparisons are safe and cross-run absolute ones are not**. Every
conclusion above is a within-run comparison.

## What this does not establish

Encoders are the same small, untuned, 12-epoch, byte-level-JSON models as the
original probe; they set a bar, not a ceiling. Depth is measured to `k = 2`
only, though the mask width and every gain column are flat from `k = 1` to
`k = 2`, which is what "stabilised" would look like.

The mechanism is a property of **this corpus's root types**. A corpus rooted at
`int`/`str`/`float` would show no front/back asymmetry at `k = 0` either — which
is the point: the asymmetry was never about the ends of a program.
