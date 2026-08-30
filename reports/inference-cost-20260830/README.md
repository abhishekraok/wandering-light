# What an execution costs, and what inference costs

Two questions behind the cost model in [`docs/metrics.md`](../../docs/metrics.md):
what is actually inside one `basis_function_executions` unit, and does policy
capacity cost wall-clock in the solver's inner loop.

Reproduce with `uv run python experiments/inference_cost.py`. Figures below are
two consecutive runs on `UltraDesk` (RTX 4090, WSL2, Ubuntu 22.04, torch 2.8.0,
8 CPU threads). **Absolute numbers move with machine load; the shape does not.**

## One charged execution is mostly not the function

| component | run 1 | run 2 |
|---|---:|---:|
| `execute(f, state)` — the basis function | 4.04 us | 4.13 us |
| `search_key()` on the result | 2.87 us | 2.89 us |
| queue, visited-set, list copies | 19.85 us | 20.00 us |
| **total, as BFS achieves it** | **26.76 us** | **27.02 us** |
| | 37,375/s | 37,012/s |

**The basis function is ~15% of a charged execution.** The rest is structural
identity and search bookkeeping. Anything compared against an
executions-per-second figure is being compared against a unit that is mostly
overhead, which is worth stating whenever such a ratio is quoted.

## Inference latency is flat in model size on GPU

End-to-end `score_actions_batch` at beam 64 — the exact call the rollout makes
per layer, including serialization, tensor build, host-to-device transfer and
forward pass. Freshly initialised models at each width; latency depends on
architecture and batch shape, not weight values.

| size | parameters | CPU ms (r1 / r2) | CUDA ms (r1 / r2) | best |
|---|---:|---:|---:|---|
| published | 90,374 | 4.63 / 4.42 | 8.70 / 8.35 | **CPU** |
| 1M | 724,022 | 7.67 / 7.19 | 9.44 / 8.23 | **CPU** |
| 10M | 6,504,310 | 35.25 / 27.45 | 8.91 / 8.60 | **CUDA** |
| 40M | 25,329,270 | 95.17 / 91.47 | 8.83 / 9.01 | **CUDA** |
| 125M | 75,504,758 | 369.02 / 385.53 | 10.30 / 10.07 | **CUDA** |

Two findings, both stable across runs:

- **CPU scales roughly linearly** with parameters — ~80x from 90K to 125M.
- **CUDA is flat** — 8.2 to 10.3 ms across three orders of magnitude, because
  the call is dominated by Python serialization and kernel-launch overhead
  rather than matmul.

**The crossover sits between ~1M and ~10M parameters.** Below it use CPU; above
it use GPU. A size comparison run at one fixed device measures the device as
much as the size.

The consequence for capacity: on GPU, going from 90K to 125M costs ~1.2x wall
clock per scoring call. Inference latency is not what bounds model size in this
loop — training cost and VRAM are.

## Bearing on the published rollout run

`reports/materialized-rollout-20260829/test/summary.json` records
`protocol.device = cuda` with the 90,374-parameter policy, which is on the CPU
side of the crossover. That run paid the GPU penalty for no benefit; the table
above puts it at roughly 1.9x on the scoring call.

This moves the wall-clock axis only. Solve counts (60/96 overall, 12/48 at
d5-8) and execution counts are device-independent up to floating-point
tie-breaking in the beam ordering.

## Caveats

- **Load-sensitive.** These were measured while other work ran on the same box.
  An earlier ad-hoc measurement posted on [solve-step-by-step (#55)](../../issues/55)
  gave 3.37 ms CPU / 4.64 ms CUDA at the published size and 30,733 exec/s for
  BFS, against 4.4-4.6 / 8.4-8.7 and ~37,000 here. Direction and crossover are
  unchanged; treat the absolutes as within a factor of ~2, not as constants.
- **Not the trained checkpoint.** Latency is weight-independent, but no
  accuracy claim is made here.
- **One host.** Every number is `UltraDesk`-specific by construction.
