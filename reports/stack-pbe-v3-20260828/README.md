# A stack PBE corpus written behaviour-first

Date: 2026-08-28 · [pbe-generator-behaviour-first (#67)](../../issues/67) · branch point `bf19d33`

The corpus deleted in `bf19d33` was *harvested*: enumerate every program up to
depth *d* from a few seed stacks, keep the behaviourally-distinct ones, call
each survivor a task. Three defects followed, all recorded in `docs/STATUS.md`
and argued in [stack-corpus-no-higher-order (#59)](../../issues/59) — seventeen
distinct input sets across 1,857 tasks, a quotation handed to the solver inside
36.6% of task inputs, and a certified-shortest label that made *building* a
quotation strictly dominated by the one supplied for free.

The root cause under all three is that harvesting makes "this task is useless"
unfalsifiable: usefulness was *defined* by reachability, which is why
`quote quote swap quote cons` was a task.

This replaces it with the inversion. A person writes the behaviour; the machine
is asked to meet it.

## How a task is made now

A `Behaviour` in `wandering_light/stack_pbe.py` is four things: a one-line
summary, a plain-Python **reference** (`sum`, `max`, keep the evens), an input
**sampler**, and a stack-machine **witness**. Reference and witness are written
independently and every task checks that they agree, so a wrong label needs the
same mistake made twice, in two languages, on data the sampler chose after both
were written. All 40 behaviours agree on 500 fresh inputs each.

Two consequences that do the real work:

- **Inputs are drawn per example, never from a table.** There is no `FAMILIES`
  dict to be seventeen rows long.
- **Every task carries hidden held-out examples.** Eight pairs, drawn from the
  same sampler and disjoint from the shown ones, excluded from the prompt. A
  candidate program is graded on those. Overfitting the shown examples is
  therefore impossible by construction rather than merely unmeasured.

## The three defects, measured

`experiments/stack_pbe_scope.py`, the same audit `stack_corpus_scope.py` ran on
v2, so the columns are comparable.

| | v3 | v2 ([#59](../../issues/59)) |
|---|---:|---:|
| tasks | 1,600 | 1,857 |
| behaviours | 40 | — |
| **distinct input stacks** | **15,550** | **17** |
| distinct input sets (one per task) | 1,600 | 17 |
| **quotation in the task input** | **0.0%** | 36.6% |
| **program builds a quotation** | **87.5%** | 0.0% |
| program runs code at all | **87.5%** | 11.8% |
| program uses a combinator | **87.5%** | 5.7% |
| builds a closure over task data | 15.0% | — |
| no arithmetic, predicate or combinator | 10.0% | 55.3% |

The residual 10% is deliberate: four first-order behaviours (`head`, `tail`,
`push_front`, `append`) keep a floor of 1–3 op tasks so the corpus is not
uniformly deep.

Witness length: median 6.5, max 35. Histogram —
`{1: 120, 2: 40, 3: 40, 5: 320, 6: 280, 7: 160, 10: 40, 11: 40, 12: 120, 13: 80, 14: 40, 15: 80, 16: 120, 22: 40, 25: 40, 35: 40}`.

**The split is on the behaviour, not the task.** v2 hashed the task, which
leaves the same behaviour on both sides of the boundary with different data —
that asks whether a solver can redo a problem it has already seen, which is a
much weaker question than PBE is for. 11 of 40 behaviours are held out (440
tasks), spanning witness lengths 5 to 16. The task-level rule is still recorded
per row as `task_hash_split` so the older protocol stays reproducible.

## The basis had to change, and that bounds #60's gate

`executable-26` cannot host this corpus. Not "does it badly" — cannot.

No op in `executable-26` puts an op *name* on the stack, and `quote`/`cons`/`cat`
only combine values already there. So from a data-only input every reachable
quotation is a list of data, and a data-only body pushes values without
consuming any — which `map`, `filter` and `fold` all reject, because each
requires a body leaving exactly one value.

Measured rather than argued, by expanding the depth-5 ball around four data-only
roots drawn from the catalogue's own samplers
(`experiments/stack_pbe_basis_floor.py`):

| | `executable-26` | `mark-28` |
|---|---:|---:|
| states | 307,707 | 4,368,089 |
| distinct quotations reached | 6,058 | 85,333 |
| of which **executable** | **0** | 78,731 |
| legal applications of `i`/`dip`/`map`/`filter`/`fold`/`ifte` | 33,468 | 36,351 |
| …that **execute any op** | **0** | 2,289 |

Thirty-three thousand legal applications of the six code-running ops, not one
of which runs any code. They are legal only vacuously — `filter` over an empty
list never reaches its body, and `map` with an empty body is the identity.

This does **not** overturn
[solver-build-quotations (#60)](../../issues/60)'s measured gate, and the
`reports/mark-28-corpus-20260828/` result stands exactly as written: on the
*harvesting* generator, `mark-28` bought 0.3 points of combinator share for
10.5x the generation cost. That report's own conclusion was **"the lever is the
generator, not the op set."** Pull that lever and the op set becomes
load-bearing: `mark-28` is not a preference here, it is the floor.

## Are the tasks pinned down by their examples?

The failure mode to rule out is a short program that reproduces the shown
outputs without computing anything — pushing a constant, projecting part of the
input. `experiments/stack_pbe_search_audit.py` enumerates *every* `mark-28`
program up to depth *d* from a task's input stacks, then re-checks each fit
against that task's held-out examples.

Forty tasks sampled with a fixed seed, one core:

| | depth ≤ 4 | depth ≤ 5 |
|---|---:|---:|
| states searched per task | ~41,000 | ~981,000 |
| tasks with any fit | 1 / 40 | **7 / 40** |
| …fitting the shown examples and **failing the held-out ones** | **0** | **0** |
| …with a fit shorter than the authored witness | 0 | 0 |
| witnesses certified shortest by the sweep | 1 | 7 |
| wall clock | 80 s | 1,688 s |

Every fit found is the authored witness itself — `0 [ add ] fold`,
`[ dup add ] map`, `[ 0 lt? ] filter`, `1 [ mul ] fold`, `uncons drop`. Search
either recovers the intended program or returns nothing; it never returns
something else that happens to work. **No spurious fit exists at either depth.**

The seven are all 1–5 op tasks — `sum`, `product`, `tail`, `double_each`,
`square_each`, `keep_negative` — that is, the deliberate floor. Nothing with a
6+ op witness is touched, which is thirty-three of the forty. So the audit is
simultaneously the evidence that the tasks are well posed and the measured
death certificate for BFS on them: 28 minutes of exhaustive search across the
sample buys the easiest 17.5% and nothing else.

BFS appears here as an *auditor*, run once offline at corpus-build time, not as
a solver. "No program of five ops or fewer fits these examples" is a claim about
the whole space, and only exhaustive enumeration can establish it.

Two generator-side guards work upstream of this: an example set is rejected if
every output is equal (a constant fits) or if the output is some fixed position
of the input stack on every example (a projection fits).

## Where forward search stands

Effective branching from *data-only* stacks measures ~19–24x per step
(2,100 states at depth 3, 41,000 at depth 4, 980,000 at depth 5, per task with
its examples carried as one product state) — near the published `mark-28`
`b_eff` of 23.11. Against that, half the catalogue needs 10 ops or more, and
`maximum` needs 16, `count_evens` 22, `prefix_sums` 25, `span` 35. At 19x per
step a 16-op program sits around 10^20 nodes.

So this corpus is outside forward search by construction and by a stateable
margin. That is the property the old corpus lacked: its 75.2% BFS solve rate
measured search over an enumeration of what BFS already reaches.

It also lines up with what `docs/STATUS.md` already records for this substrate —
300/300 random walks land on distinct endpoints, every one escapes the depth-5
ball — and with the bound placed on it there: **neither substrate measurement
shows that any *directed* mechanism reaches further.** Nothing here changes
that. This corpus makes the gap measurable; it does not close it.

## What this does not say

- **Witnesses are authored, not certified shortest.** Every row is
  `witness_source: "authored"`. The search audit certifies the ones within its
  swept depth and nothing beyond.
- **Forty behaviours is small**, and 11 of them are the entire eval split. Any
  cross-behaviour generalisation number off this corpus carries that variance.
  The catalogue is built to be extended: a behaviour is four fields and its
  acceptance test is mechanical, so proposals can be gated by execution rather
  than review.
- **No solver has been run on it.** There is no solve rate here, deliberately.
  The v2 numbers were quoted for two months before anyone measured what the
  corpus contained.
- **The samplers are narrow** — small integers, lists of 1–7 — so nothing here
  speaks to scale in the data, only to variety.

## Reproduce

```bash
# The payload is on the Hub; the tracked manifest pins its digests.
uv run python -c "from wandering_light.corpus_hub import fetch_corpus; fetch_corpus('wandering_light/evals/data/stack_pbe_v4/manifest.json')"
uv run python -m wandering_light.stack_pbe --per-behaviour 40 --seed 17
uv run python experiments/stack_pbe_scope.py
uv run python experiments/stack_pbe_basis_floor.py --depth 5
uv run python experiments/stack_pbe_search_audit.py --depth 4 --tasks 40
```

Generation is under two seconds and verifies all 40 behaviours first. The
basis floor is ~70 s. The search audit is ~2 s per task at depth 4 and ~40 s at
depth 5.
