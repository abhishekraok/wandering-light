# Is distance-to-go learnable? The value half of #33

#33 proposes two heads: a policy that orders expansion, and a value that
estimates distance to the target. `reports/policy-probe-20260824/` measured the
policy half. This measures the value half — the one an A*-style best-first
search actually needs, since `f = g + h` is only as good as `h`.

Reproduce:

```
uv run python experiments/value_probe.py <path-to-deep_corpus_v1> --epochs=40
```

## The labels are free and exact

A `deep-corpus-v1` task carries a witness program of length `d` certified
shortest. Replay its first `j` ops on the input to reach `s_j`; the true
remaining distance from `s_j` to the target is **exactly `d - j`**. If some `p`
reached the target from `s_j` in fewer than `d - j` steps, then `w_1..w_j`
followed by `p` would solve the task in fewer than `d`, contradicting
certification.

So every prefix of every witness is an exactly labelled point, with no search
and no bootstrapping — 137,140 of them (107,049 discovery / 15,046 validation /
15,045 test), split by root, not reshuffled. Replaying all 31,350 witnesses
reproduces the stored output on **31,350 of 31,350** tasks and takes 3.8 s.

The full witness (`j = d`) is excluded: its state *is* the target, so search
settles it with a goal test and never asks `h`.

## The bar, stated before looking

BFS spends ~805k executions at distance 7 and ~4.4M at 8. If `h` cannot predict
remaining distance to within roughly **1 step at remaining distance 6 and
beyond**, best-first will not rescue the search. Accuracy near the goal is not
worth much — that is where search is cheap anyway.

Absolute error is scored against the **median**, not the mean: under L1 the
optimal constant is the median, so a mean baseline would have flattered the
learned model. Pinned in `tests/test_value_probe.py`.

## Result

    Value probe -- 15045 test prefix points

    predictor                   MAE   within 1     bias  train MAE
    constant (median)         1.460     53.8%   -0.122         --
    type pair lookup          1.232     68.3%   -0.225         --
    type + size lookup        1.242     67.9%   -0.264         --
    learned (mean-pool)       1.182     50.8%   -0.223      1.152
    learned (conv)            0.978     61.6%   -0.179      0.830

    MAE by true remaining distance (the axis that matters)

    predictor                   r1      r2      r3      r4      r5      r6      r7      r8
    constant (median)         2.00    1.00    0.00    1.00    2.00    3.00    4.00    5.00
    type pair lookup          1.18    0.71    0.79    1.08    1.72    2.35    3.00    3.20
    type + size lookup        1.13    0.72    0.80    1.14    1.74    2.38    3.05    3.52
    learned (mean-pool)       1.10    0.74    0.78    1.10    1.68    2.16    2.54    2.77
    learned (conv)            0.74    0.69    0.77    1.01    1.35    1.66    2.08    2.51

**Learning pays here, unlike at the front of the policy probe.** The conv
encoder cuts MAE from the dict's 1.232 to 0.978 — a 21% improvement, against
the 10% the same architecture managed on first-action ranking. Types alone are
not the whole signal for `h`.

**And it still fails the bar.** At remaining distance 6, 7, 8 the best
predictor is off by 1.66, 2.08, 2.51 steps. The bar was ~1 step at 6 and
beyond, and nothing on the ladder comes close. `h` is accurate exactly where
search is already cheap and degrades monotonically into the regime that
motivated #33.

**The dicts are barely better than a constant far from the goal.** At r7 the
constant scores 4.00 and the type-pair lookup 3.00; both are just regressing to
the median. The conv encoder's 2.08 is a real gain over that, and still not a
usable `h`.

**Bias is negative throughout (−0.18 for the conv encoder): every predictor
systematically *under*-estimates.** That is the admissible direction, so
weighted A* remains available — but the underestimate is a hedge toward the
median, not a bound, and it is worst precisely at large distances.

**This is not an undertrained model.** Train MAE is 0.830 against test 0.978
for the conv encoder, and 1.152 against 1.182 for the mean-pool — the mean-pool
model cannot fit even the data it has seen. Both are representation-limited,
not generalization-limited, so more epochs will not move this. What would move
it is a better encoding than truncated JSON bytes; that is the open question,
and it is the same one the policy probe left open.

## Successor discrimination: what search actually asks

Every training state sits *on* an optimal path. Search does not — it spends its
time on states that fell off one. On-path MAE alone would therefore mislead, so
the probe also ranks the distinct successors of each test input by `h` and asks
whether an optimal one comes first. Only tasks with a certified-complete
optimal set are used (2,905 of them, 9.06 distinct successors each).

    predictor                 top-1  mean rank
    constant (median)         0.0%       8.99
    type pair lookup         21.5%       5.32
    type + size lookup       21.2%       5.44
    learned (mean-pool)      31.7%       3.67
    learned (conv)           38.1%       3.35

**`h` used as an expansion order is no better than the lookup table the policy
probe already measured.** Against `reports/policy-probe-20260824/`, whose
first-action numbers are over a near-identical denominator (9.78
type-compatible actions there against 9.06 distinct successor states here —
they differ because two ops can produce the same state):

| ordering signal | mean rank | top-1 |
|---|---|---|
| policy, conditional lookup | 3.18 | 42.1% |
| policy, conv encoder | 2.86 | 42.2% |
| **value `h`, conv encoder** | **3.35** | **38.1%** |

So the value head, converted into an expansion order, is *worse* than a
two-field dict on the policy side, and clearly worse than the policy encoder.
It would earn its place in #33 only if `g + h` beat a pure policy ordering, and
nothing measured here suggests it does.

Per-step top-1 of 38.1% also bounds greedy descent: following only the
top-ranked child gives roughly `0.381^d` for a full length-`d` path — about 2%
at `d = 4`. Best-first keeps a frontier rather than committing, so this is not
the search's solve rate, but it is the compounding the frontier has to absorb.

## What this establishes

1. **The value half of #33 is measurable for free and now measured.**
   137,140 exact labels, no search.
2. **`h` fails the stated bar.** MAE 1.66–2.51 at remaining distance 6–8,
   against a bar of ~1. Best-first on this `h` will not rescue distance-7 and
   distance-8 search, which is what #33 was for.
3. **Learning does pay for `h` — 21% over a dict — but into a floor that is
   still too high.** The direction is right and the magnitude is not enough.
4. **As an expansion orderer, `h` is beaten by the policy lookup table already
   on file.** If #33 keeps one head, the evidence favours the policy, not the
   value.

## What it does not establish

The encoder is small, untuned, and reads truncated JSON bytes — 224 per side.
Train MAE says that representation is the binding constraint, so a structured
encoding could move these numbers; nothing here bounds what a better one does.

Successor discrimination is measured at **step 1 only**, from the task input.
Deeper states are further off-distribution and were not tested, and the
compounding across levels — the thing that decides whether best-first beats BFS
— is not measured at all.
