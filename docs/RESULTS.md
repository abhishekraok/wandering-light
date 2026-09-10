# Results

**The headline numbers.** Every row is copied from an artifact under
`reports/`; the full ledger, every run and every benchmark, is
[`RESULTS_DETAILED.md`](RESULTS_DETAILED.md). `STATUS.md` says what these
mean; issues say why.

## Stack PBE v4 — 220 behaviours, eval split 1,800 tasks

Held-out grading, one sample per task. "Sample 40" is the seed-17 eval sample;
it is the easy end of the split (median witness 26 ops against 40).
[report](../reports/stack-pbe-v4-baseline-20260901/README.md)

| solver | config | tasks | solved | wall (s) |
|---|---|---:|---:|---:|
| `gpt-5.6-luna` | effort high | sample 40 | **15 (37.5%)** | 2,286 |
| BFS | 100,000 nodes | sample 40 | 6 (15.0%) | 3,917 |
| `qwen3:30b-a3b` (local, 4090) | 300 s deadline | sample 40 | 6 (15.0%) | 3,381 |
| BFS | 10,000 nodes | full 1,800 | 120 (6.7%) | 22,803 |
| beam 64 × depth 48, 1m next-op policy trained on v4 train | seeds 0 / 1, CPU | sample 40 | 7 (17.5%) / **10 (25.0%)** | 131 / 126 |

One luna solve and one BFS solve are the `aligned_matches` sampler shortcut,
so 14 and 5 compute the behaviour. BFS solves nothing at 6+ ops; qwen3
nothing at 16+; luna's wall is ~40 ops.

**v3 anchor, same protocol on the 40-behaviour predecessor:** luna 30/40
(75.0%) on its sample 40, `gpt-5.6-terra` 377/440 (85.7%) on the full split,
BFS at 100,000 nodes 11/40 (27.5%). The frontier number **halves** from v3
to v4 luna-against-luna; search and the local arm fall by about the same
ratio.

## Stack PBE v4 — policy size (300-task eval cohort, 60 per witness-length bucket)

Next-op policies trained on v4 train, beam 64 × depth 48 with the #70 guard, two
seeds per arm; solve rate is the mean of two seeds over the CPU cells. CUDA
agrees on every task but one (10m seed 1, `negate_each`), which puts 10m at
6–10 at 0.033 there and its coverage at 0.180. Witness length is an upper
bound, not a certified distance.
[report](../reports/compute-optimal-mark28-20260903/README.md),
[compute-optimal-mark-28 (#101)](../../issues/101)

| witness length | 1m (655,836 params) | 2m (2,045,660) | 10m (6,288,412) | BFS 10,000 nodes, full split |
|---|---:|---:|---:|---:|
| 1–5 | 0.608 | 0.592 | 0.617 | 0.500 |
| 6–10 | 0.050 | 0.000 | 0.042 | 0.000 |
| 11–20 | 0.267 | 0.267 | 0.233 | 0.000 |
| 21–40 | 0.000 | 0.000 | 0.000 | 0.000 |
| 41+ | 0.000 | 0.017 | 0.017 | 0.000 |

Cost to clear the cohort, (coverage, total wall-clock): CPU 0.185 / 1,009 s,
0.175 / 1,374 s, 0.182 / 2,198 s; CUDA 0.185 / 887 s, 0.175 / 940 s,
0.180 / 1,074 s, for 1m / 2m / 10m. Per-execution AUC at 1–5 ops: 0.315 /
0.312 / 0.315. Every arm's validation CE bottoms at 0.925–0.976 at epoch 1 or 2
and rises after. Eleven of the cohort's 43 behaviours are ever solved; the
longest verified solves are 12 ops (the 44-op and 28-op solves the report quotes
are held-out-surviving shortcuts; `docs/STATUS.md`, Superseded, 2026-09-09).

## `deep-corpus-v1` — learned solver against search (96 held-out tasks, 12 per distance 1–8)

[report](../reports/materialized-rollout-20260829/README.md)

| solver | solved /96 | distance 5–8 /48 |
|---|---:|---:|
| aligned-pair rollout policy, 90,374 params, beam 64 | **60** | **12** (7/12 at d5, 5/12 at d6, 0 at d7–8) |
| BFS at the policy's mean d5–8 spend (6,238 executions) | — | 1 |
| `opt-125m` SFT, k=64 | 14 | 0 |
