# Metrics

**This file defines the yardsticks. It carries no measurements.**

`STATUS.md` says what is true; issues say why; this file says *how any of it is
measured*. Numbers belong in the other two — if a constant appears here it has
gone stale.

## Cost units

Two, always reported together.

| unit | charges | role |
|---|---|---|
| `wall_clock_seconds` | the timed solve call — search **and** model inference | **primary** |
| `basis_function_executions` | one per type-compatible basis function attempted, including one that raises | diagnostic |

Wall-clock is primary because execution counting charges the executor but not
the policy, so a larger model can win on executions while losing on real
compute. The sandbox is local Python, so both draw on the same processor:
there is one scarce resource, not two.

Executions stay because they are hardware-independent, and because
certification, `b_eff` and the growth model are stated in them.

Wall-clock is comparable only within one host. Results record an
`evaluation_host` fingerprint, and seed aggregation refuses to average across
hosts rather than silently mixing them.

Device matters as much as size — measure it, do not assume it. See
[make-solver-baselines-comparable (#25)](../../issues/25).

## Solve cost

**Area under the solve-rate vs log-budget curve, per certified distance.**

```
for each solver, each certified distance d:
    sweep budget t; record executor-verified solve_rate(t)
    score = AUC(solve_rate vs log2 t)
```

An area under a *rate* curve, not a time. One number per (solver, distance).

- **Per distance, never aggregated.** Aggregating lets gains at low distance
  mask a wall at high distance.
- **Not mean time-to-solve.** That is undefined under censoring, and averaging
  over solved tasks only is gameable — time out on everything hard and report a
  superb mean. An unsolved cohort sweeps out no area, so AUC needs no timeout
  constant.
- **Log budget axis**, because search cost is exponential in depth and a linear
  axis hides everything below the top decade.
- **Fix the sweep range** across everything compared; AUC scales with it.
- **Keep the curve, not only the scalar.** Its right-hand end is the final solve
  rate, which is what [raise-solve-rate-d7-8 (#74)](../../issues/74) targets.

A faster solver can beat a more accurate one on AUC. That is intended for a
*cost of solving* metric, and it is the reason to read the curve too.

For a frozen range `[t_min, t_max]`, compute the integral exactly from each
task's first-hit cost `tau`, rather than on a sampled budget grid:

```
AUC = mean_tau clip(log2(t_max / max(t_min, tau)), 0, log2(t_max / t_min))
```

A miss contributes zero. A hit at or above `t_max` contributes zero; a hit at
or below `t_min` contributes the full log-budget span. Report both this area
and the area divided by the span. The displayed curve is the empirical
first-passage CDF, with its discontinuities retained.

## Bits are log-time

Levin's universal search with a prior `P` finds `p` in time `O(t(p) / P(p))`:

```
log2 T  =  log2 t(p)  +  (-log2 P(p))  =  log2(fuel) + |p| under the prior
```

That is Kt, with code length under a **learned** prior rather than a uniform
one. Kt is not an alternative currency to time — `2^Kt` **is** the time, so
bits saved convert to speedup by exponentiation.

What Kt does not charge is the cost of *computing* the prior. See
[weight-paths-by-kt (#45)](../../issues/45).

## Where total time goes

```
T(N)  =  D . c_inf(N)   +   c_exec . 2^(H(N).d)
         |___________|       |________________|
          inference               search
```

| symbol | meaning |
|---|---|
| `N` | policy parameter count — the variable being chosen |
| `T(N)` | expected wall-clock to solve one task; the quantity minimised |
| `D` | scoring calls per solve; the beam batches into one call per layer, so ~= depth |
| `c_inf(N)` | wall-clock of one scoring call at size `N` |
| `c_exec` | wall-clock of one basis execution |
| `H(N)` | residual uncertainty per step, in **bits**, after the policy ranks actions |
| `d` | certified distance (shortest program length) |

With `H` bits left per step, effective branching is `2^H`, so the search tree
has `2^(H.d)` leaves. **One bit saved per step divides search by `2^d`.**

The two terms scale oppositely in `N`: inference grows with size, search shrinks
exponentially as `H` falls. Where they cross is why an optimal size exists —
[compute-optimal-policy-size (#80)](../../issues/80).

**Two caveats.** It is a reasoning model, not a law: it assumes uniform `H` and a
tree, while the solver deduplicates structurally, so real node counts run below
`2^(H.d)`. Use it to decide which term dominates, not to predict a runtime. And
`H` must be measured **in situ**, along states rollouts actually visit —
improving on-distribution bits moves `H` on paper and nothing on the metric
above. At a visited state, let `C` be the type-compatible actions and `O` the
complete set of actions that begin a shortest path to the target. The measured
code length is the policy's total masked mass on that set, not the chosen
action or one recorded witness:

```
H = -log2((1 - epsilon) * sum(a in O) P(a | C) + epsilon * |O| / |C|)
```

An exact bounded search must certify all of `O`; unresolved states are excluded
and their coverage and stopping reasons reported. When scoring a bounded
sample of rollout states, freeze the sampling rule before evaluation and report
both state-weighted and task-balanced means so long rollouts do not silently
dominate the result. Gate each headline on at least 80% certified coverage in
its own weighting; retain the conditional estimate and censoring counts below
that threshold, but do not present it as headline `H`.

## Protocol

- Never score an unopened test split more than once. Select on validation, lock
  the choice, then score once.
- Three paired seeds; single-seed fine grain is noise against a zero baseline.
- Report both cost units, the host, and the device.
- Complete and seal every timed cell before observer replay or oracle labeling;
  untimed analysis must not alter the load between primary timing cells.
- Findings go on the issue and, once merged, as a `STATUS.md` row — not here.
