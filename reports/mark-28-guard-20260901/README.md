# Guarding the states inside an open quotation

Date: 2026-09-01 · [prune-open-mark-states (#70)](../../issues/70) · branch point `2881637`

Under an open `[`, the `mark-28` machine captures every op as a symbol instead
of applying it. Every guard returns `None`, so all 28 ops are legal where 13
were before:

```
[ (1 2 3) 5 ]            13 of 28 ops legal
[ (1 2 3) 5 MARK ]       28 of 28 ops legal
[ (1 2 3) 5 MARK mul ]   28 of 28 ops legal
```

That is the mechanism behind `mark-28`'s `b_eff` of 23.11 against
`executable-26`'s 13.38, and it fires exactly where a higher-order program must
pass — 87.5% of v3 witnesses build a quotation. This is route 2 from the issue:
search-side only, machine unchanged, so no substrate measurement moves.

## What the guard is

`wandering_light/quotation_guard.py`. Two checks on the body of the innermost
open quotation, plus one on the brackets.

- **type** — an op meets a value whose type it can never accept. `0 null?` is
  dead because `null?` wants a list; `0 i` is dead because `i` wants a
  quotation. Only values the *body itself* put on the stack are judged;
  anything a consumer will supply is unknown and therefore unconstrained.
- **reach** — how far the body dips below its own contribution. Computed and
  reported, but **not pruned on**; see below.
- **close** — a state holding more open marks than it has ops left can never get
  back to a mark-free stack. Sound only because every target is mark-free: 0 of
  1,600 v3 tasks and 0 of 8,800 v4 tasks carry a `MARK` in any input, output,
  held-out input or held-out output.

All three are monotone: appending an op can only add a type error and can only
increase reach, and spending an op can only reduce the budget. A body that fails
one can never be repaired, which is what makes these prunes rather than a prior
over bodies.

Deliberately **not** checked: a *closed* nested quotation.
`[ [ 0 null? ] drop ]` builds a body that cannot run and then throws it away,
which is legal, so a type error inside a completed inner quotation is not fatal
to the outer one.

## The reach check was wrong first, and v4 caught it

The first version bounded reach by what the consumer hands over. That prunes 85
of v4's 220 witnesses. They build closures:

```
replace_value:  [ [ drop eq? ] ... ] swap cons swap cons map
```

The body reaches three deep and `map` supplies one — the other two arrive from
the `cons`es that run *after* the quotation closes. Bounding reach by the
consumer alone loses the witness. The shipped bound adds the remaining op
budget, since each widening costs at least one op.

The type half was unaffected: **0 type failures across all 220 v4 witnesses**,
and 0 across v3's 40.

The correction weakens the reach check — it now has to allow for widening that
has not happened yet — and every number in this report is measured after it.

## The gate

Pre-registered in the issue: states per task drops at least 2x, all 40 authored
witnesses stay reachable, and the depth-4 and depth-5 spurious-fit results
(0/40) are unchanged.

Met, on all three.

**States per task.** `--guard off` against `--guard both`, 40 tasks, seed 17:

| | off | both | reduction | wall clock |
|---|---:|---:|---:|---|
| depth 4 | 41,917 | 9,742 | **4.30x** | 78 s → 39 s |
| depth 5 | 1,001,797 | 110,266 | **9.09x** | 1,851 s → 682 s |

**Nothing found changes.** Per-task rows are identical on fits-shown,
fits-held-out, spurious and shortest-fit, for all 40 tasks at both depths: still
1/40 and 7/40 tasks admitting a fit, still **0/40 spurious**, still 0 with a fit
shorter than the authored witness. The unpruned arm reproduces
`reports/stack-pbe-v3-20260828/` exactly (1/40 at 78 s against its 80 s, 7/40 at
1,851 s against its 1,688 s), so before and after are the same machine and run.
Identical counts here are evidence, not proof: `body` is only guaranteed to
preserve the *shortest* fit, and at these depths there is no room for a program
that both fits and carries a dead quotation.

**Witnesses stay reachable.** Established by replay rather than by the sweep,
because the depth-5 sweep only reaches 7 of the 40 witnesses and cannot speak
for the other 33. Replaying every authored witness op by op, consulting the
guard at each prefix with the tightest budget a sound search could use:

| corpus | behaviours | open-mark prefix states | pruned |
|---|---:|---:|---:|
| v3 | 40 | 705 | **0** |
| v4 | 220 | 26,013 | **0** |

## The body guard strengthens with depth, because it strengthens with length

The prune rate is a function of **body length**, not of search depth. A
depth-5 budget cannot afford a body longer than about three ops, and at three
ops only about half the space is provably dead — which is why the body arm buys
1.30x at depth 4 and 2.30x at depth 6. `stack_pbe_v4` has a median witness of 43
ops, so this is the regime that matters and it is off the right-hand edge of
every search reported here.

Measured directly against body length — 50,000 random bodies per length, supply
2, seed 17, so no search budget is involved:

| body length | pruned (type) | reaches deeper than 2 | surviving b_eff |
|---:|---:|---:|---:|
| 1 | 0.0% | 7.2% | 28.00 |
| 2 | 18.2% | 17.9% | 25.32 |
| 4 | 52.4% | 14.7% | 23.26 |
| 8 | 85.2% | 5.2% | 22.05 |
| 12 | 95.3% | 1.6% | 21.69 |
| 16 | 98.6% | 0.5% | 21.43 |

Branching under an open mark settles at **~21.4 against the machine's 28** —
1.31x per op *inside* a quotation, compounding with body length. At length 16,
98.6% of the legal space cannot run. The right-hand column is the diagnostic
nothing prunes on.

The same curve seen through a search budget, states per task on v3:

| depth | unpruned | body | close | both |
|---:|---:|---:|---:|---:|
| 4 | 40,807 | 31,274 — 1.30x | 9,506 — 4.29x | 9,218 — 4.43x |
| 5 | 980,688 | 572,474 — 1.71x | 125,375 — 7.82x | 105,549 — 9.29x |
| 6 | 25,647,169 | 11,147,652 — **2.30x** | 2,365,275 — 10.84x | 1,509,390 — **16.99x** |

Three tasks, seed 17. The body arm clears the pre-registered 2x on its own by
depth 6 and is still climbing there — which is where the budget runs out, not
where the mechanism does.

The body arm strengthens with depth because it is really strengthening with body
length. `close` strengthens faster because the last ply is where unclosed states
pile up: at depth 4, 79.8% of states sit inside an open mark and every one of
them is unclosable within the remaining budget.

## What this does not say

- **It bounds the legal space, not solver success.** Nothing here is a solve
  rate. What a fixed *node* budget buys with the guard on is
  [rerun-baselines-v4 (#85)](../../issues/85)'s measurement, not this one.
- **Random bodies are not the distribution a solver walks.** The `dead` column
  describes the space; the `surviving b_eff` column is the honest read, since it
  is the branching of the set that survives.
- **The 2x gate does not discriminate the issue's thesis.** `close` clears it
  alone, for a reason that has nothing to do with whether a body makes sense.
  The two arms are reported separately for exactly that reason.
- **Success on v4 remains evidence of composition, not invention.** On
  `mark-28` a witness shares a median 79.8% of its 8-op windows with another
  witness, and eval witnesses share 76.8% with train.

## Reproduce

```bash
uv run python -c "from wandering_light.corpus_hub import fetch_corpus; fetch_corpus('wandering_light/evals/data/stack_pbe_v3/manifest.json')"

# the gate: 40 tasks, seed 17, both arms at both depths
for d in 4 5; do for g in off both; do
  uv run python experiments/stack_pbe_search_audit.py \
    --corpus wandering_light/evals/data/stack_pbe_v3 \
    --depth $d --tasks 40 --seed 17 --guard $g \
    --out reports/mark-28-guard-20260901/audit-depth$d-guard$g.json
done; done

# prune rate against body length, no search budget involved
uv run python experiments/stack_pbe_guard_probe.py \
  --out reports/mark-28-guard-20260901/body-lengths.json \
  lengths --max-length 16 --samples 50000

# states per task per arm
for d in 4 5 6; do
  uv run python experiments/stack_pbe_guard_probe.py \
    --out reports/mark-28-guard-20260901/states-depth$d.json \
    states --depth $d --tasks 3
done
```

Runtimes on one core of this desktop (Ubuntu 22.04 under WSL2, CPU only —
nothing here uses the GPU): the depth-4 gate is 78 s unpruned and 39 s guarded,
depth 5 is 1,851 s and 682 s. The length sweep is ~3 min. The states probe is
~2 s per task at depth 4, ~16 s at depth 5 and ~7 min at depth 6, per arm.

Raw results in this directory: `audit-depth{4,5}-guard{off,both}.json`,
`body-lengths.json`, `states-depth{4,5,6}.json`, `witnesses-v{3,4}.json`.
