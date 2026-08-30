# Metrics

**This file defines the yardsticks. It does not record findings.**

`STATUS.md` says what is true; issues say why; this file says *how any of it is
measured*. If you are about to quote a cost or a solve rate, the units are
defined here. Calibration constants below are measured on the reference host
and each points at the issue that argues it.

Last updated: 2026-08-30

## 1. Cost units

Two units, always reported together.

| unit | what it charges | role |
|---|---|---|
| `wall_clock_seconds` | the timed solve call — search **and** model inference | **primary** |
| `basis_function_executions` | one per type-compatible basis function attempted, including one that raises | diagnostic |

**Why wall-clock is primary.** Execution counting charges the executor but not
the policy, so a larger model can win on executions while losing on every real
measure of compute. In this repo the sandbox is local Python: an execution and
a forward pass draw on the same CPU, so there is one scarce resource, not two.

**Why executions survive anyway.** They are hardware-independent, and they are
the unit that certification, `b_eff`, and the `~5.8^d` growth model are stated
in. Dropping them would break more than it fixed.

**Host pinning.** Wall-clock is comparable only within one host. Every result
records an `evaluation_host` fingerprint, and seed aggregation refuses to
average across hosts rather than silently mixing them.

See [make-solver-baselines-comparable (#25)](../../issues/25), which argues the
shared cost model; the wall-clock flip landed in PR #73.

## 2. Calibration constants

Measured on `UltraDesk` (RTX 4090, WSL2, Ubuntu 22.04). Reproduce before
quoting on another host.

### What one "execution" actually costs

| component | time |
|---|---:|
| `execute(f, state)` — the basis function itself | 3.78 us |
| `search_key()` on the result | 2.85 us |
| queue, visited-set, list copies | ~25.9 us |
| **total, as measured end to end** | **32.5 us** (30,733/s) |

Only **12%** of an execution is the function call. The rest is structural
identity and search bookkeeping. Worth remembering whenever an
executions-per-second figure is compared against anything else.

### Policy inference, `score_actions_batch` at beam 64

The full call the rollout makes per layer: serialization, tensor build,
host-to-device transfer, forward pass.

| params | CPU ms | CUDA ms | best |
|---:|---:|---:|---|
| 90,374 | **3.37** | 4.64 | CPU |
| 724,022 | 4.95 | **4.58** | CUDA |
| 6,504,310 | 19.72 | **5.01** | CUDA |
| 25,329,270 | 80.81 | **9.22** | CUDA |
| 75,504,758 | 308.25 | **5.98** | CUDA |

CPU scales roughly linearly with parameters. **CUDA is flat** across three
orders of magnitude, because the loop is dominated by Python serialization and
kernel-launch overhead rather than matmul. **Crossover is ~1M parameters** —
below it, use CPU. A size comparison run at one fixed device measures the
device as much as the size.

See [compute-optimal-policy-size (#80)](../../issues/80).

## 3. Solve-cost metric

**Area under the solve-rate vs log-wall-clock-budget curve, reported per
certified distance.**

```
for each solver, each certified distance d:
    sweep budget t; record executor-verified solve_rate(t)
    score = AUC(solve_rate vs log2 t)
```

It is an area under a *rate* curve, not a time. One number per
(solver, distance) cell.

**Per distance, never aggregated.** Aggregating lets gains at d1–4 mask a wall
at d7–8, which is the failure mode the standing goal exists to prevent.

**Why not mean time-to-solve.** It is undefined under censoring — at d7–8
nothing solves — and averaging over solved tasks only is gameable: time out on
everything hard and report a superb mean. AUC handles this by construction, since
an unsolved cohort simply sweeps out no area.

**Why a log budget axis.** Search cost is exponential in depth, so budgets span
orders of magnitude and a linear axis hides everything below the top decade. It
also puts the axis in Kt's own currency (section 4).

**Fix the sweep range** across everything being compared; AUC scales with it.

Worked example — one distance, 12 tasks, budget swept 1 ms to 1 s
(`log2 t` from 0 to 10):

| solver | solves | first reaches it at | AUC |
|---|---|---|---:|
| A | 7/12 | 100 ms | 0.583 x (10 - 6.64) = **1.96** |
| B | 7/12 | 512 ms | 0.583 x (10 - 9) = **0.58** |
| C | 4/12 | 4 ms | 0.333 x (10 - 2) = **2.67** |

A and B end at the same solve rate; AUC separates them 3.4x because B is late.
C wins outright despite solving fewer, because it is fast. That is the intended
behaviour for a *cost of solving* metric — but keep the curve, not only the
scalar: its right-hand end is the final solve rate, which is what
[raise-solve-rate-d7-8 (#74)](../../issues/74) actually targets.

## 4. Bits are log-time: the Kt bridge

Levin's universal search with a prior `P` over programs finds `p` in time
`O(t(p) / P(p))`. In logs:

```
log2 T  =  log2 t(p)  +  (-log2 P(p))
        =  log2(fuel) + |p| under the prior
```

That is Kt, with code length measured under a **learned** prior instead of a
uniform one. So Kt is not an alternative currency to time — `2^Kt` **is** the
time, and "bits saved" converts to "search-time speedup" by exponentiation.
The policy-probe work already reports in this direction: 3.08–3.25 bits saved
at the back, quoted as ~500–5,900x at d=7.

What Kt does **not** charge is the cost of *computing* the prior. That omission
is the entire reason an optimal model size exists.

See [weight-paths-by-kt (#45)](../../issues/45) and the probes on
[replace-autoregressive-solver (#33)](../../issues/33).

## 5. Where the total time goes

```
T(N)  =  D . c_inf(N)   +   c_exec . 2^(H(N).d)
         |___________|       |________________|
          inference               search
```

| symbol | meaning | measured |
|---|---|---|
| `N` | policy parameter count — the variable being chosen | 90,374 today |
| `T(N)` | expected wall-clock to solve one task; the thing minimised | — |
| `D` | scoring calls per solve; the beam is batched into one call per layer, so ~= depth | <= 8 |
| `c_inf(N)` | wall-clock of one scoring call at size `N` | section 2 |
| `c_exec` | wall-clock of one basis execution | 32.5 us |
| `H(N)` | residual uncertainty per step in **bits**, after the policy ranks actions | 6.9 uniform -> ~4.3 type-mask -> ~1.35 probe |
| `d` | certified distance (shortest program length) | 1–8 |

With `H` bits of uncertainty left per step, effective branching is `2^H`; over
`d` steps the tree has `2^(H.d)` leaves, each costing `c_exec`. **One bit saved
per step divides search by `2^d`** — 128x at d=7.

At d=7 with measured constants:

| `H` | search term | inference term |
|---|---|---|
| 4.3 (type mask only) | 2^30.1 ~= 1.1e9 executions ~= **10 hours** | ~35 ms |
| 1.35 (probe) | 2^9.45 ~= 700 executions ~= **23 ms** | ~35 ms |

At the good end inference *exceeds* search. **That crossover is why the optimal
size is not infinite**: once the policy is good enough you stop being
search-bound, and further capacity must pay for itself against a term that is
no longer shrinking.

### Two caveats that decide whether the model is worth using

**It is a reasoning model, not a law.** It assumes uniform `H` across steps and
a tree; the solver deduplicates structurally, so real node counts run below
`2^(H.d)`. Use it to decide which term dominates, not to predict a runtime.

**`H` must be measured in situ.** The 1.35 figure is on-distribution, on
certified witness prefixes. The observed failure is off-distribution: the beam
retains the certified witness for 24, 22, 17, 7, 2, then 0 of 24 d7–8 tasks
across depths 1–6. A model that improves on-distribution bits and not in-situ
bits moves `H` on paper and nothing on the metric in section 3.

See [solve-step-by-step (#55)](../../issues/55).

## 6. Protocol rules

- Never score an unopened test split more than once. `deep-corpus-v1` test has
  already been read twice; `deep-corpus-v2` test is the only clean one left.
- Select on validation, lock the choice, then score once.
- Three paired seeds. Single-seed fine grain is noise against a zero baseline.
- Report both cost units, the host, and the device.
- Findings go on the issue and, once merged, as a row in `STATUS.md` — not here.
