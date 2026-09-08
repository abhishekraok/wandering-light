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

One luna solve and one BFS solve are the `aligned_matches` sampler shortcut,
so 14 and 5 compute the behaviour. BFS solves nothing at 6+ ops; qwen3
nothing at 16+; luna's wall is ~40 ops.

**v3 anchor, same protocol on the 40-behaviour predecessor:** luna 30/40
(75.0%) on its sample 40, `gpt-5.6-terra` 377/440 (85.7%) on the full split,
BFS at 100,000 nodes 11/40 (27.5%). The frontier number **halves** from v3
to v4 luna-against-luna; search and the local arm fall by about the same
ratio.

## `deep-corpus-v1` — learned solver against search (96 held-out tasks, 12 per distance 1–8)

[report](../reports/materialized-rollout-20260829/README.md)

| solver | solved /96 | distance 5–8 /48 |
|---|---:|---:|
| aligned-pair rollout policy, 90,374 params, beam 64 | **60** | **12** (7/12 at d5, 5/12 at d6, 0 at d7–8) |
| BFS at the policy's mean d5–8 spend (6,238 executions) | — | 1 |
| `opt-125m` SFT, k=64 | 14 | 0 |
