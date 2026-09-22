# E1 of inner-frames (#160): how often a committed search could execute the body it is writing

witness-trajectory-shape (#150) found that 84% of `stack_pbe_v4` witness ops sit
under an open mark, where the machine state does not move and a ranker over
states has nothing to look at. inner-frames (#160) proposes committing to the
consuming combinator when a quotation opens, so the partial body can be run on
the frame that combinator will feed it. E1 asks only whether that frame is
**computable**, on the data a search would actually have. It trains nothing, so
no number here says the frames help.

Driver `experiments/inner_frames.py`, output `frames.json`, 59 s for every
corpus on CPU. The unit is the **distinct witness program**, replayed on one
task's shown inputs (the first in corpus order). Coverage is over the
**product**: `stack_beam` applies an op to all k shown examples and drops the
candidate if any one refuses, so a frame defined on one example is not a frame
the search has; the per-example rate is reported beside it as the price of that
rule. Consumers are attributed per example and must agree.

Three things about the method matter for reading the numbers.

- **The frame chain.** A depth-1 frame comes from the real stack below the mark.
  A deeper one does not: the inner body runs on whatever the enclosing body's
  prefix leaves on the enclosing frame, so a deep frame is undefined the moment
  any ancestor's prefix refuses. The driver carries one virtual stack per open
  mark and advances it one item per written op. Nothing had executed a depth-2
  frame before this; the prior 88.1% figure was **text-level** (is the consumer
  the next token?) and the prior executed figure covered depth 1 only.
- **Attribution is structural.** Following a finished quotation through later
  stack states can only ever attribute a top-level one: a nested quotation is
  absorbed when its parent closes. Consumers are therefore found by bracket
  matching, and value-tracking is kept as a depth-1 cross-check.
- **Two readings of the design.** *Strict* is #160 exactly as written: the
  consumer is the next token after the quotation. *Capture-aware* also looks
  through the closure idiom `x [ body ] swap cons filter`, where the quotation
  is consed into a closure before the combinator runs it; the captured values
  are pushed onto the frame. Both are reported, because the repair is not in the
  issue and the difference between them is what it would be worth.

**Every frame is checked against the machine.** Recording
`stack_machine.run_body` observes the stack each combinator actually hands a
body, and every computed frame is compared with it. **6,447 of 6,447 match**,
across both readings and every capture count that occurs. That check found a
real error while it was being built: the captured values were being reversed,
which is a no-op at one capture and wrong at two, and the 27 two-capture frames
are what settle the order.

## Coverage

| corpus | programs | under-mark ops | **strict** | **capture-aware** | per example (strict → capture) |
|---|---:|---:|---:|---:|---|
| `stack_pbe_v4` train | 175 | 7,157 | **0.700** | **0.768** | 0.792 → 0.856 |
| `stack_pbe_v4` eval | 45 | 1,897 | **0.674** | **0.756** | 0.758 → 0.877 |
| `stack_pbe_world_v3` train | 482 | 109,300 | **0.430** | **0.636** | 0.561 → 0.803 |
| `stack_pbe_world_v3` eval | 44 | 8,353 | **0.564** | **0.782** | 0.606 → 0.858 |
| staged (#155), train | 128 | 4,135 | **0.790** | **0.875** | 0.852 → 0.937 |

**Recursion is worth about 3.5×, and it is not 84%.** Previewing at depth 1
only, as the prior probe did, defines a frame for 0.230 × 0.874 = **0.20** of
v4 train's under-mark ops. The frame chain takes that to **0.70**. The
predictions registered on #160 before the run were 80–85% for v4 and 55–60% for
world-v3 train; the measured strict values are 0.700 and 0.430, so both were
optimistic, and the prediction that v4 would land below 80% is the one that
holds. On v4 train the ops divide 23% / 45% / 32% across depth 1 / 2 / 3+, with
coverage 0.874 / 0.756 / 0.499.

**The frame chain is the failure mode.** `parent_refused` is the largest
undefined class on v4 and world-v3 — 1,844 of v4 train's 2,149 undefined ops,
46,906 of world-v3 train's 62,267 — though **not** on the staged corpus, where
`other_op_first` (365) and `empty_list` (290) both exceed it (213).

**The closure repair is worth more than anything else measured here.** Going
capture-aware cuts `other_op_first` on v4 train from 169 to 43, on world-v3
train from 8,882 to 932, and on staged from 365 to 75, and it drags
`parent_refused` down with it (world-v3 train 46,906 → 30,474) because a
closure that could not be tagged was blocking every frame beneath it. World-v3
gains 0.21 of coverage from a repair that is three lines of attribution.

## Which combinator, and the asymmetry the design leans on

Under a `map` / `filter` / `fold` parent the inner body opens on a **fresh**
one- or two-value stack; under `i` / `dip` / `ifte` it inherits the whole outer
stack. Coverage at depth ≥ 2, strict → capture-aware:

| corpus | fresh parent | inherited parent |
|---|---|---|
| `stack_pbe_v4` train | 0.801 → 0.815 | 0.493 → 0.607 |
| `stack_pbe_world_v3` train | 0.512 → 0.628 | 0.301 → 0.559 |
| staged (#155) | 0.789 → 0.873 | 0.250 → 0.444 |

The asymmetry is real and large under the strict reading and **halves** under
the capture-aware one, because the closure idiom sits disproportionately under
inheriting parents. Per consuming combinator on v4 train (strict), the ops
divide very unevenly:

| tag | frame | coverage | under-mark ops |
|---|---|---:|---:|
| `ifte` | inherited | 0.682 | 4,200 |
| `dip` | inherited | 0.693 | 1,758 |
| `fold` | fresh | 0.888 | 943 |
| `filter` | fresh | 1.000 | 46 |
| `map` | fresh | 1.000 | 41 |

`map` and `filter` bodies — the case the design's fresh-stack argument is built
on — are **87 ops of 7,113**, while `ifte` alone is 4,200. On the staged corpus
the ordering is different again (`dip` 0.951 out-covers `fold` 0.856 and
`filter` 0.921), so "inherited previews worst" is a tendency across corpora and
not a rule within one.

## Commitment ambiguity

| corpus | mean legal tags (of 6) | ln(mean legal) | H(tag) | **H(tag \| shape)** strict → capture |
|---|---:|---:|---:|---|
| `stack_pbe_v4` train | 4.54 | 1.51 | 1.06 | **0.306 → 0.369** |
| `stack_pbe_v4` eval | 4.67 | 1.54 | 1.04 | **0.287 → 0.377** |
| `stack_pbe_world_v3` train | 4.68 | 1.54 | 1.17 | **0.347 → 0.475** |
| `stack_pbe_world_v3` eval | 4.59 | 1.52 | 1.10 | **0.324 → 0.462** |
| staged (#155), train | 4.83 | 1.57 | 1.48 | **0.458 → 0.624** |

The issue prices commitment at `ln 6` = 1.79 nats per quotation if the tag is
uninformative. **Legality is not what saves it**: a mean of 4.54 tags of 6 are
legal at a real `[`, worth `ln 4.54` = 1.51 nats, because `i` and an `ifte`
condition are legal on any stack. What saves it is the prior over consumers:
conditioned on a cheap descriptor of the stack under the mark, the tag costs
**0.29–0.46 nats** strict and **0.37–0.62** capture-aware, since the repair adds
tag variants to distinguish. Against the beam's `ln 64` = 4.16-nat budget, a
five-quotation witness spends 1.5–2.3 nats on commitment at these rates and
9 nats at the issue's figure — the difference between fitting inside the beam
and not. That makes the caveat load-bearing rather than decorative: these are
**in-sample** entropies over hindsight tags with a hand-made shape descriptor,
so they bound what a ranker could reach only if it gets the shape for free. E2
must measure this out of sample, on the shape the ranker actually sees.

## The unchained-fold objection is empirically dead

For each fold write step the partial body was run on `(acc₀, itemⱼ)` for every
item, counting a disagreement when the accept set differs across items — the
only form of the objection that can change what a beam keeps. Chaining the
partial body instead is not an option: it leaves the wrong arity and refuses at
once, so no better mid-body frame exists.

| corpus | fold write steps with a frame | accept set differs |
|---|---:|---:|
| `stack_pbe_v4` train | 533 | **0** |
| `stack_pbe_v4` eval | 111 | **0** |
| `stack_pbe_world_v3` train | 8,847 | 75 (0.85%) |
| `stack_pbe_world_v3` eval | 480 | **0** |
| staged (#155), train | 1,002 | **0** |

Folds in these corpora run over homogeneous lists, so the first item is
representative. The registered prediction of "under 5%" holds with room.

## Cross-check, and what E1 does not settle

Structural and value-tracking attribution agree on 1,446 of 1,596 depth-1
quotations in v4 train. Of the 150 disagreements, 50 are `other_op_first`
against `mutated_or_folded` — the same substantive case under two names — and
100 are the structural method naming a real consumer (`map` 50, `filter` 42,
`fold` 8) where value-tracking reports absorption. That is the closure idiom
again: a quotation consed into a closure *is* mutated, so value-tracking is
right that the object moved and wrong that nothing runs it.

E1 is coverage, not learnability. A frame being computable says nothing about
whether a ranker over inner states beats a goal-blind baseline, which is E2 and
the gate that matters. The numbers are witness statistics on one task's examples
per program, not cohort statistics; `world-v3` train is dominated by a few very
long programs (median distinct witness 145 ops, longest 4,136), so its aggregate is not a
per-behaviour average. The machine check covers the frames this driver computed
and no others: `no_observation` exceeds `compared` in most cells, because a
quotation whose frame was never computed has nothing to compare.
