# Was the policy probe's encoder starved, or wrongly shaped?

`experiments/policy_probe.py` encoded a task as `f"{input}\x00{output}"` cut to
256 bytes. That overruns on **46.0% of `deep-corpus-v1`** (14,423 of 31,350), so
every published number from the probe was a floor on input bandwidth rather than
on architecture. This re-runs the depth probe with the confound removed, and
with a control that says which of the two things was actually binding.

```
uv run python experiments/policy_probe_depth.py <corpus> --depths=0,1,2 \
    --min-distance=3 --seeds=0,1,2 --epochs=12
```

## Verdict

**Truncation mattered enormously at the back and not measurably at the front,
and that is mostly a fact about which end needs the target, not about how much
was cut.**
Giving the identical model a 1024-byte window instead of 256 moves the back
probes by 1.06–1.46 mean rank and 14.6–17.2 points of top-1, and moves the front
probes by nothing at any depth (+0.03 to +0.06 rank, all inside seed spread).
Front k=0 and back k=0 are *the same 2,700 tasks with the same bytes hidden*, so
this is not a property of the corpus slice: truncation eats the **target** end of
the string, a last action must produce the target, and a first action need only
consume the input, which is never what gets cut.

**Separately, and everywhere, the per-item encoder buys more than the wider
window does at the front.** Encoding a task as its 3–16 aligned example pairs
rather than as one string beats the widened flat encoder at all six probes,
by 0.25–0.61 mean rank, including at the two probes where nothing was truncated
in the first place. So the answer to "was representation the bottleneck" is yes
twice over, for two unrelated reasons, and only one of them is bandwidth.

The brief gated this stage on front k=1 and k=2. Both move — 4.03 → 3.63 and
3.53 → 3.00 — but entirely on the architecture arm, with the visibility arm flat.

## The three arms

    flat   both sides concatenated, cut at 256 bytes -- the published encoder
    wide   the same module at 1024 bytes             -- truncation removed
    pairs  each aligned example pair encoded on its own, pooled across items

`wide` is what makes the result readable. `pairs` changes what the model can see
*and* how it is shaped, and those license different follow-ons. A convolution
and a max-pool do not know how long their input is, so `wide` has parameter
shapes **identical** to `flat` (90,582 both, pinned by a test) and differs only
in whether the task fits. `pairs` is 96,214, +6.2%.

    wide - flat    removing truncation, architecture fixed
    pairs - wide   the architecture, truncation already gone

**Why pairs is shaped this way.** Every basis function is applied elementwise
(`executor.py:118`), and input and output item counts match on 31,350 of 31,350
tasks. A task is not one string pair; it is 3–16 *aligned* pairs `(x_i, y_i)`,
and this is programming by example. Each pair is encoded on its own, the shared
trunk runs over items in the batch dimension, and a permutation-invariant
mean-and-max pool collapses them. At 96 bytes per item, 0.26% of items are cut
instead of 46.0% of tasks.

## The harness still measures what it did before

`flat` is the published model, bit-identical, re-run here on the same device
class. It reproduces `reports/policy-probe-depth-20260825/`:

| probe | published | `flat` here |
|---|---|---|
| front k=0 | 3.28 / 38.9% | 3.27 / 38.9% |
| front k=1 | 4.04 / 32.0% | 4.06 / 32.3% |
| front k=2 | 3.58 / 37.0% | 3.58 / 37.6% |
| back k=0 | 3.38 / 40.9% | 3.41 / 41.4% |
| back k=1 | 3.74 / 41.4% | 3.72 / 43.2% |
| back k=2 | 3.84 / 43.2% | 3.87 / 42.9% |

Worst deviation is 1.8 points of top-1 at back k=1. The bits/step column
independently lands on all six of the published mask-width figures to two
decimals (below), which is a second, unrelated check on the same harness.

## The decomposition

Mean rank and top-1, mean ± sd over **3 seeds**, differenced **pair by pair**
(seed *i* of one arm against seed *i* of the other; `flat` and `wide` are the
same module, so seed *i* of each starts from identical weights). `(ns)` marks a
gap inside twice the pooled seed spread.

| probe | target seen / blind | flat | wide | pairs | wide − flat | pairs − wide |
|---|---|---|---|---|---|---|
| front k=0 | 62.5% / 19.3% | 3.27 / 38.9% | 3.21 / 41.1% | 2.96 / 45.8% | +0.06 ±0.09 (ns) | +0.25 ±0.10 |
| front k=1 | 92.4% / 2.7% | 4.06 / 32.3% | 4.03 / 32.3% | 3.63 / 37.5% | +0.03 ±0.05 (ns) | +0.40 ±0.08 |
| front k=2 | 94.1% / 0.4% | 3.58 / 37.6% | 3.53 / 38.3% | 3.00 / 47.6% | +0.05 ±0.07 (ns) | +0.53 ±0.07 |
| back k=0 | 62.5% / 19.3% | 3.41 / 41.4% | 2.35 / 56.0% | 1.84 / 65.2% | **+1.06 ±0.06** | +0.50 ±0.06 |
| back k=1 | 63.6% / 19.3% | 3.72 / 43.2% | 2.34 / 59.7% | 1.79 / 69.5% | **+1.38 ±0.09** | +0.55 ±0.06 |
| back k=2 | 64.4% / 19.3% | 3.87 / 42.9% | 2.41 / 60.2% | 1.79 / 71.0% | **+1.46 ±0.11** | +0.61 ±0.08 |

Top-1 gaps: `wide − flat` is −0.0% to +2.1% on the front (all ns) and +14.6% /
+16.5% / +17.2% on the back. `pairs − wide` is +4.7% to +10.8%, significant at
all six.

## The mechanism, on identical tasks

The obvious reading of the front/back split — that the front probes were simply
less truncated — is wrong. It is true that they are (92–94% of the target
visible against 62–64%), but **front k=0 and back k=0 are the same tasks**. Same
input, same output, same 19.3% of tasks with the target entirely invisible. Only
the predicted action differs. Cutting each by how much of the target the
256-byte budget showed:

| stratum | n | front k=0 `wide − flat` | back k=0 `wide − flat` |
|---|---|---|---|
| target entirely unseen | 520 | +0.06 ±0.03 (ns) | **+2.63 ±0.13** |
| target partly seen | 799 | +0.15 ±0.05 | **+1.69 ±0.14** |
| target fully seen | 1381 | +0.01 ±0.11 (ns) | +0.11 ±0.07 (ns) |

Both ends are flat on the fully-visible stratum, as they must be. On the blind
stratum the back gains 2.63 mean rank and the front gains 0.06.

**Truncation is `[:256]` applied to `input + "\0" + output`, so a long input
starves the target rather than losing its own tail.** A first action has to
consume the input, which survives whole. A last action has to produce the
target, which is precisely what was cut. The 46% figure describes a deprivation
that only one of the two ends could ever have felt.

**Two things covary with the end, and they weaken this from a clean experiment
to a strong one.** front k=0 and back k=0 differ in which action is predicted,
but also in mask width — 9.50 compatible ops against 25.87 — and in how hard
their blind tasks are to begin with. Those are not independent of each other: a
task is blind precisely because its input is enormous, an enormous input has a
rare type, and a rare type is what makes the front's mask tight. So the front's
blind tasks are its *easiest* (top-1 56.3% against the probe's 38.9%), and there
is less room to gain there whatever the reason.

Normalising each gain by its own headroom — the distance from `flat`'s rank to a
perfect 1.0 — and adding front k=1, whose mask at 18.41 is most of the way to the
back's:

| stratum, target invisible | mask | n | flat rank | `wide − flat` | share of headroom |
|---|---|---|---|---|---|
| front k=0 | 9.50 | 520 | 1.73 | +0.06 (ns) | 8% |
| front k=1 | 18.41 | 72 | 1.51 | +0.18 | 35% |
| back k=0 | 25.87 | 520 | 4.98 | +2.63 | 66% |
| back k=2 | 26.87 | 520 | 5.61 | +3.31 | 72% |

So **"the front never benefits" is too strong**. front k=1's invisible tasks do
gain, significantly, and they sit at a mask width between the two ends. What
survives is the ordering and its size: the back gains 15–50x more in absolute
rank, and roughly twice as much as a share of what was available. Whether the
residual front/back difference is about which end needs the target or about how
much uncertainty is left after masking is **not separated by this run**, because
the corpus does not offer a front probe with a back-sized mask.

This also explains the published front/back asymmetry differently from
`reports/policy-probe-depth-20260825/`. That report attributed the back's
learnability headroom to mask width (25.87 functions produce a given output type
against 9.50 consuming a given input type). Mask width is still real, but a
large part of what looked like "the back has more left to learn" was the back
being **the only end penalised by the encoder's truncation** — and it goes away
when the window is widened, without touching the mask.

## Dose–response, within one probe

back k=2, cut by how much target the wider window newly reveals. The anchor bin
is tasks where 1024 bytes shows nothing 256 did not, so it measures transfer —
whatever the arms gained from having trained on different data elsewhere — and
the visibility effect is the slope above it.

| newly revealed | n | flat | wide | wide − flat |
|---|---|---|---|---|
| nothing new (anchor) | 1468 | 2.87 / 52.6% | 2.60 / 56.2% | +0.27 ±0.05 |
| 0–25% | 140 | 2.31 / 66.0% | 2.04 / 66.7% | +0.27 ±0.09 |
| 25–50% | 78 | 1.93 / 73.5% | 1.62 / 76.5% | +0.30 ±0.17 (ns) |
| 50–75% | 161 | 4.89 / 34.2% | 2.13 / 64.4% | +2.76 ±0.16 |
| 75–100% | 853 | 5.82 / 21.3% | 2.25 / 63.6% | **+3.57 ±0.05** |

Monotone, and the top bin is 13x the anchor. **Transfer is real but small**:
+0.27 of a +3.57 effect, about 8%. It was worth measuring rather than assuming
away — the arms are separately trained models, so the anchor bin is not a
placebo with a known-zero effect, and reading the two-stratum cut as
difference-in-differences would have assumed transfer was common across strata
in exactly the case where it is least likely to be.

The same cut on front k=2 is flat everywhere (anchor +0.02, ns) except its
75–100% bin, n=49, at +0.38 ±0.17. So even at the front, revealing a target that
was actually hidden helps a little — there are just almost no such tasks there.

## Bits per step

`H = -log2` of the mass on the *optimal set* over the type-compatible ops,
averaged over test tasks. Summing over the set rather than one witness op is
deliberate: a policy should not be punished for preferring an equally short
route.

| probe | type mask | conditional | flat | wide | pairs |
|---|---|---|---|---|---|
| front k=0 | 2.90 | 2.38 | 2.34 | 2.28 | 2.15 |
| front k=1 | 4.07 | 3.28 | 2.98 | 2.98 | 2.78 |
| front k=2 | 4.24 | 3.20 | 2.83 | 2.81 | 2.45 |
| back k=0 | 4.56 | 3.26 | 2.66 | 1.95 | 1.48 |
| back k=1 | 4.61 | 3.40 | 2.66 | 1.92 | 1.41 |
| back k=2 | 4.60 | 3.41 | 2.80 | 1.93 | 1.35 |

(ε = 0.01; the ε = 0 and ε = 0.1 columns are in `depth*.json` and differ by at
most 0.05, because **no predictor puts zero mass on a reachable optimal op on
any task** — the unreachable fraction is 0.0% everywhere, so the interpolation
weight changes nothing here and the calibration diagnostic came back clean.)

**Absolute bits are not portable; bits *saved over the type mask* are.** A
singleton witness label and a graded optimal set put the same policy at
different absolute bits, and the mask width differs by probe, so only the saving
within a population compares across rows:

| probe | mask | saved by flat | by wide | by pairs | `pairs` as compute at d=7 |
|---|---|---|---|---|---|
| front k=0 | 2.90 | 0.56 | 0.62 | 0.75 | 38x |
| front k=1 | 4.07 | 1.09 | 1.09 | 1.29 | 523x |
| front k=2 | 4.24 | 1.41 | 1.43 | 1.79 | 5,914x |
| back k=0 | 4.56 | 1.90 | 2.61 | 3.08 | 3.1M x |
| back k=1 | 4.61 | 1.95 | 2.69 | 3.20 | 5.5M x |
| back k=2 | 4.60 | 1.80 | 2.67 | **3.25** | **7.1M x** |

The last column is `2^(saved x 7)` — how much smaller a depth-7 enumeration
gets, against a searcher that has the type mask and nothing else. The front buys
two to four orders of magnitude; the back buys six to seven.

**Reading the two columns.** `H` conditions on the type mask for every rung,
because the mask is substrate rather than predictor — no real searcher
enumerates 118 ops when most are type errors. So `uniform over all` and
`type mask` are the same distribution once conditioned and their `H` is
identical by construction; the rank column's `uniform over all` row at 118.00 is
a scale reference, not a predictor anyone would run. The type mask lands on
exactly `log2(mask width)` — 2.90 / 4.07 / 4.24 / 4.56 / 4.61 / 4.60 against the
published 9.50 / 18.41 / 19.90 / 25.87 / 26.71 / 26.87 compatible ops — which is
what makes this column subtract against substrate branching.

Rungs that already return probabilities are renormalised over the compatible
set; only the learned logits are softmaxed. Softmaxing a distribution living in
[0, 0.1] would flatten it to nearly uniform and every rung would report the mask
width.

**One place the two columns disagree, and it is the argument for having both.**
`conditional lookup` at front k=0 ranks *better* than `flat` (3.38 against 3.27
is a near tie) while costing more bits (2.38 against 2.34), and at back k=0 it
ranks far worse (4.48 against 3.41) while costing 0.60 bits more. A lookup table
orders acceptably and is badly calibrated. Which of those matters depends on the
search: a policy-weighted best-first search consumes path probability, so bits is
its unit; a fixed-width beam only sees the top-*b* cut, so rank is. A method that
sharpens without reordering helps the first and does nothing for the second.

## The margin is the pooling, not the type tag

`pairs` adds two things to `wide` at once: per-item pooling, and an explicit
type-tag embedding per side. The flat arms were never blind to type — the
`{"type": "builtins.range"` header is the first thing in their window and is
never the part that gets cut — but "at most nothing" is not a measurement.
`pairs_untyped` is `pairs` with the tag removed, run on the same population,
seeds and epochs:

| probe | wide | pairs_untyped | pairs | from pooling | from the tag |
|---|---|---|---|---|---|
| front k=0 | 3.21 / 41.1% | 2.94 / 45.6% | 2.96 / 45.8% | +0.27 (109%) | −0.02 |
| front k=2 | 3.53 / 38.3% | 3.05 / 47.5% | 3.00 / 47.6% | +0.49 (92%) | +0.04 |
| back k=0 | 2.35 / 56.0% | 1.85 / 64.7% | 1.84 / 65.2% | +0.49 (98%) | +0.01 |
| back k=2 | 2.41 / 60.2% | 1.81 / 71.1% | 1.79 / 71.0% | +0.59 (97%) | +0.02 |

**The tag is worth nothing — between −9% and +8% of the margin, which is noise.**
The whole effect is the per-item pooling. In bits the untyped arm is within 0.02
of `pairs` everywhere (back k=2: 1.37 against 1.35).

This matters for the parameter argument as well as the mechanism. Dropping the
tag puts the arm at **90,374 parameters against the published model's 90,582** —
0.2% smaller. So the strongest claim in this report does not need the +6.2%: an
encoder the same size as the published one, differing only in that it reads a
task as aligned example pairs, is what produces the gain.

## What a forward solver gets from this

A forward rollout predicts the *next* action from a materialized state, so it is
a **front** probe. Every dramatic number above is on the end it never uses. The
sober version, for anything reading forward:

* **Removing truncation does nothing for a forward solver.** `wide − flat` is
  inside seed spread at all three front depths.
* **The per-item encoder buys +0.25 / +0.40 / +0.53 mean rank** at k=0/1/2, and
  +4.7 / +5.1 / +9.3 points of top-1. Real, replicated across three seeds, and
  modest.
* In bits per step the front ladder is **2.34 → 2.28 → 2.15** at k=0,
  **2.98 → 2.98 → 2.78** at k=1, **2.83 → 2.81 → 2.45** at k=2 (flat → wide →
  pairs). The best front number in the study is 2.15 bits/step, at the depth
  whose type mask is unusually tight and therefore least representative.

Against the back's 1.35 at k=2, the front is a different regime, and a back
number should not be quoted as evidence about a forward rollout.

## The back is now the best direction in the project

`back k=2` with the pair encoder reaches **1.79 mean rank, 71.0% top-1, and 1.35
bits/step** — 3.25 bits saved over its own type mask. That is the strongest
policy number recorded anywhere in this repo, and it beats every front number by
a wide margin: at d=7 it is worth ~7.1M x against the front's best ~5,900x.

The back was the *harder* end in `reports/policy-probe-20260824/`. It looks that
way only through an encoder that could not see the target, which is the one
thing a backward prediction depends on.

The design implication runs against a working assumption, so it is worth stating
plainly rather than leaving in a caveat: **if a forward rollout is capped near
2.5 bits/step while the backward direction reaches 1.35, the answer is not to
make the front better — it is to search from the target, or from both ends.**
Bidirectional search was #33's original motivation and has been dormant since
`reports/policy-probe-depth-20260825/` concluded the asymmetry was a first-step
artifact. That conclusion was about mask width and it still holds; the
learnability conclusion drawn alongside it does not. This is outside the present
brief and is not a result about search — no bidirectional search was run — but it
is the largest thing this run changes about what to build next.

## An observation without a story

`pairs − wide` rises with depth at both ends: front +0.25 / +0.40 / +0.53, back
+0.50 / +0.55 / +0.61. The obvious explanation is item composition — a per-item
encoder should like the short regular `int`/`float`/`str` items that states
collapse onto — but composition moves in **opposite** directions at the two ends.
Front items get shorter and more regular with depth; back targets get longer and
more root-like as they peel back toward the root. The same trend from opposite
composition changes is not composition. We do not have a mechanism for this and
are not proposing one.

## What this does not establish

**The two budgets carry different amounts of padding.** At 256 bytes most tasks
fill or overrun the window, so `flat` often max-pools over no padding; at 1024
every task carries roughly three quarters padding. This is not dilution — a run
of identical pad bytes gives the same convolution activation at every interior
pad position, so the max sees one constant however much surrounds it. What
remains is that `wide` always includes that constant and a truncated `flat` task
does not, which can bite only in a channel where real signal falls below it.
Masking `flat`'s padding would have moved it off the published configuration.

**Roughly 30 cells carry a 2σ flag over three seed pairs.** Running the identical
stratified analysis on **random ranks** produced one cell flagged at 2σ. A single
flagged cell here is not evidence; the pattern across bins is.

**Same small, untuned, 12-epoch byte-level models as the original probe.** They
set a bar, not a ceiling. Three seeds size the noise; they do not make a
96k-parameter model a strong one. The epoch count and learning rate were held at
the published values for all arms, which may not suit all three equally.

**Labels are witness-derived**, as in the published depth probe, so they are a
subset of the optimal set — which understates every arm the same way. Within-run
comparisons are safe; absolute numbers are not comparable across runs.

**This prices the supervision these probes were given, not the supervision
available.** Every arm here trains on one op per example — the op the witness
happened to take. A graded label from backward BFS over the trajectory graph
would score *every* applicable op by the true distance of its successor, roughly
18 values per example instead of 1, and would label off-path states too. That is
a different supervision regime, not a bigger model, and nothing here tests it. So
"the front is capped near 2.45–2.78 bits/step" is a statement about this
supervision at this scale. What the run does establish is narrower and still
useful: **the front's ceiling is not raised by letting this encoder see more.**

**Populations.** The 46.0% overrun and the dose figures quoted per probe are on
the *peeled test splits* under `min_distance = 3`, not on the whole corpus. The
corpus-wide figures differ (13.5% of all tasks across all splits see none of
their target, against 19.3% on these test splits).

## Files

* `depth{0,1,2}.json.gz` — full rows including per-task ranks and optimal-set
  mass, in split order, so any cut by task property is a rescoring rather than a
  rerun. The stratified tables above come from these:

      uv run python experiments/policy_probe_strata.py <corpus> depth2.json.gz

* `untyped{0,2}.json.gz` — the type-tag ablation, same population and seeds
  (`--arms=pairs_untyped`).
* `dose.json` — per-probe truncation and target-visibility on the peeled splits.
* `depth{0,1,2}.err` — per-epoch validation loss for every arm and seed.
