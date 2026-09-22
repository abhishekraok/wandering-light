# E1 of inner-frames (#160): how often a committed search could execute the body it is writing

witness-trajectory-shape (#150) found that 84% of `stack_pbe_v4` witness ops
sit under an open mark, where the machine state does not move and a ranker over
states has nothing to look at. inner-frames (#160) proposes committing to the
consuming combinator when a quotation opens, so the partial body can be run on
the frame that combinator will feed it. E1 asks only whether that frame is
**computable**, on the data a search would actually have. It trains nothing, so
no number here says the frames help.

Driver `experiments/inner_frames.py`, output `frames.json`, 25 s for every
corpus on CPU. The unit is the **distinct witness program**, replayed on one
task's shown inputs (the first in corpus order). Coverage is over the
**product**: `stack_beam` applies an op to all k shown examples and drops the
candidate if any one refuses, so a frame defined on one example is not a frame
the search has; the per-example rate is reported beside it as the price of that
rule. Consumers are attributed per example and must agree.

Two things about the method matter for reading the numbers:

- **The frame chain.** A depth-1 frame comes from the real stack below the
  mark. A deeper one does not: the inner body runs on whatever the enclosing
  body's prefix leaves on the enclosing frame, so a deep frame is undefined the
  moment any ancestor's prefix refuses. The driver carries one virtual stack per
  open mark and advances it one item per written op. Nothing had executed a
  depth-2 frame before this; the prior figure of 88.1% was a **text-level**
  statistic (is the consumer the next token?) and the prior executed figure
  covered depth 1 only.
- **Attribution is structural.** Following a finished quotation through later
  stack states can only ever attribute a top-level one: a nested quotation is
  absorbed when its parent closes, so value-tracking reports every one of them
  as `mutated_or_folded`. Consumers are therefore found by bracket matching —
  the first token after the matching `]` that is not a sibling quotation — and
  the value-tracking method is kept as a depth-1 cross-check.

## Coverage

| corpus | programs | under-mark ops | share of all ops | **coverage (product)** | per example | depth 1 | depth 2 | depth 3+ |
|---|---:|---:|---:|---:|---:|---|---|---|
| `stack_pbe_v4` train | 175 | 7,157 | 0.843 | **0.700** | 0.792 | 0.874 (23% of ops) | 0.756 (45%) | 0.499 (32%) |
| `stack_pbe_v4` eval | 45 | 1,897 | 0.850 | **0.674** | 0.758 | 0.746 (22%) | 0.730 (47%) | 0.543 (32%) |
| `stack_pbe_world_v3` train | 482 | 109,300 | 0.888 | **0.430** | 0.561 | 0.705 (20%) | 0.456 (25%) | 0.318 (55%) |
| `stack_pbe_world_v3` eval | 44 | 8,353 | 0.844 | **0.564** | 0.606 | 0.847 (27%) | 0.596 (31%) | 0.351 (41%) |
| staged (#155), train | 128 | 4,135 | 0.705 | **0.790** | 0.852 | 0.825 (76%) | 0.681 (24%) | — |

**Recursion is worth about 3.5×, and it is not 84%.** Previewing at depth 1
only, as the prior probe did, defines a frame for 0.230 × 0.874 = **0.20** of
v4 train's under-mark ops. The frame chain takes that to **0.70**. The
predictions registered on #160 before the run were 80–85% for v4 and 55–60% for
world-v3 train; the measured values are 0.700 and 0.430, so both were
optimistic, and the prediction that v4 would land below 80% is the one that
holds.

**The frame chain is the failure mode, exactly where the design is weakest.**
`parent_refused` is the largest undefined class everywhere (v4 train 1,844 of
2,149 undefined ops; world-v3 train 46,906 of 62,264). And the asymmetry the
design leans on is real and large: under a `map` / `filter` / `fold` parent the
inner body opens on a **fresh** one- or two-value stack, and coverage at depth
≥ 2 is 0.801 on v4 train; under `i` / `dip` / `ifte` it inherits the whole outer
stack, and coverage is 0.493. World-v3: 0.512 against 0.301. Staged: 0.789
against 0.250.

**But the most common consumer is an inheriting one.** Of v4 train's 5,008
previewable under-mark ops, 2,866 (57%) sit inside an `ifte` body and only 87
(1.7%) inside `map` or `filter`; `fold` has 837 and `dip` 1,218. Counted in ops
rather than quotations, the design's fresh-stack story covers `fold` and little
else, and the combinator that dominates the corpus is precisely the one whose
children preview worst.

## Commitment ambiguity

At a `[`, what the machine's own preconditions rule out, against what the
hindsight distribution of consumers says:

| corpus | mean legal tags (of 6) | ln(mean legal) | H(tag) | **H(tag \| shape)** |
|---|---:|---:|---:|---:|
| `stack_pbe_v4` train | 4.55 | 1.52 | 1.062 | **0.306** |
| `stack_pbe_v4` eval | 4.67 | 1.54 | 1.039 | **0.287** |
| `stack_pbe_world_v3` train | 4.68 | 1.54 | 1.170 | **0.347** |
| staged (#155), train | 4.82 | 1.57 | 1.478 | **0.458** |

The issue prices commitment at `ln 6` = 1.79 nats per quotation if the tag is
uninformative. **Legality is not what saves it**: a mean of 4.55 tags of 6 are
legal at a real `[`, worth `ln 4.55` = 1.52 nats, because `i` and an `ifte`
condition are legal on any stack. What saves it is the prior over consumers:
conditioned on a cheap descriptor of the stack under the mark (its depth and the
types of its top three values), the tag costs **0.29–0.46 nats**, about a fifth
of the issue's figure. Read that as an upper bound on what a ranker could
achieve *if it gets the shape for free*: it is measured in-sample, on hindsight
tags, with a hand-made descriptor.

## The unchained-fold objection is empirically dead

For each fold write step, the partial body was run on `(acc₀, itemⱼ)` for every
item, and a disagreement counted when the accept set differs across items —
the only form of the objection that can change what a beam keeps. Chaining the
partial body instead is not an option: a partial body leaves the wrong arity and
refuses at once, so no better mid-body frame exists.

| corpus | fold write steps with a frame | accept set differs |
|---|---:|---:|
| `stack_pbe_v4` train | 533 | **0** |
| `stack_pbe_v4` eval | 111 | **0** |
| `stack_pbe_world_v3` train | 8,847 | 75 (0.85%) |
| `stack_pbe_world_v3` eval | 480 | **0** |
| staged (#155), train | 1,002 | **0** |

Folds in these corpora run over homogeneous lists, so the first item is
representative. The registered prediction of "under 5%" holds with room.

## What the staged corpus says, and it is the awkward finding

Staged witnesses have the **highest** coverage (0.790) and the **highest**
ambiguity (H(tag | shape) 0.458), and their largest undefined class after empty
lists is `other_op_first` — 365 ops whose quotation is not consumed promptly.
Their idiom is `[ body ] swap cons filter`: a closure built before the consumer
runs. A search committed to a tag at `[` cannot express that at all under the
design as written, because the thing the quotation becomes is not run by the
next combinator but consed into a value first. Staged text is the training data
behind `solve.BEST` and the shape witness-trajectory-shape (#150) argues the
project wants, so the corpus the design most needs to handle is the one whose
central idiom it does not cover.

## Cross-check, and what E1 does not settle

Structural and value-tracking attribution agree on 1,446 of 1,596 depth-1
quotations in v4 train; every disagreement is `other_op_first` against
`mutated_or_folded`, which are the same substantive case (not promptly
consumed) under two names, so the two methods do not disagree about any
quotation's fate.

E1 is coverage, not learnability. A frame being computable says nothing about
whether a ranker over inner states beats a goal-blind baseline, which is E2 and
the gate that matters. Nothing here licenses a claim that inner frames help.
The numbers are witness statistics on one task's examples per program, not
cohort statistics, and `world-v3` train is dominated by a few very long
programs (median witness 136 ops, longest 4,063), so its aggregate is not a
per-behaviour average.
