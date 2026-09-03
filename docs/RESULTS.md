# Results

**The numbers ledger.** One row per measurement, copied from the report
artifact it cites, no argument. `STATUS.md` says what the numbers mean;
issues say why; `metrics.md` says how they are measured. A new run adds a
row here as part of the work, with its JSON under `reports/`.

Conventions: solve counts are graded on **held-out** examples unless a row
says otherwise; "sample 40" is the seed-17 40-task eval sample the baseline
scripts draw by default; wall clock is one host (`UltraDesk`, RTX 4090 for
local models, CPU for BFS) and comparable only within a table.

## Stack PBE v3 — 40 behaviours, eval split 440 tasks

`stack_pbe_v3`, `mark-28`. BFS budgets are node expansions. [report](../reports/stack-pbe-baseline-20260829/README.md), [rerun-baselines-v4 (#85)](../../issues/85) for luna.

| solver | config | tasks | solved | memorised | wall (s) | artifact |
|---|---|---:|---:|---:|---:|---|
| BFS | 1,000 nodes | sample 40 | 0 (0.0%) | 1 | 30 | `bfs_eval_1k_sample40.json` |
| BFS | 10,000 nodes | sample 40 | 3 (7.5%) | 1 | 369 | `bfs_eval_10k_sample40.json` |
| BFS | 10,000 nodes | full 440 | 39 (8.9%) | 3 | 5,200 | `bfs_eval_10k_full.json` |
| BFS | 100,000 nodes | sample 40 | 11 (27.5%) | 1 | 3,326 | `bfs_eval_100k_sample40.json` |
| `gpt-5.6-terra` | effort high | sample 40 | 33 (82.5%) | 3 | 227 | `openai_terra_high_eval40.json` |
| `gpt-5.6-terra` | effort high | full 440 | **377 (85.7%)** | 12 | 2,231 | `openai_terra_high_eval_full.json` |
| `gpt-5.6-luna` | effort high | sample 40 | 30 (75.0%) | 4 | 622 | `stack-pbe-v4-baseline-20260901/luna_v3_sample40.json` |
| `qwen3:30b-a3b` | unbounded | sample 40 | 10 (25.0%) | 1 | 1,988 | `qwen3_30b_eval40.json` |
| `qwen3:30b-a3b` | 300 s deadline | sample 40 | 9 (22.5%) | 0 | 1,941 | `qwen3_30b_timeout_eval40.json` |
| `qwen3:30b-a3b` | 4,096-token cap | sample 40 | 2 (5.0%) | 0 | 844 | `qwen3_30b_capped_eval40.json` |
| `gpt-oss:20b` | default effort | 4-task sample | 1 | — | ~59 s/task | report only |
| `qwen3-coder:30b` | — | 4-task sample | 0 | — | ~4 s/task | report only |
| `qwen3:32b` (dense) | — | 4-task sample | did not finish | — | 5.1 tok/s | report only |

All `qwen3:30b-a3b` rows are ollama digest `ad815644918f`, the July-2025 thinking weights (`qwen3:30b-a3b-thinking-2507-q4_K_M` is the same blob).

**Search audit, exhaustive `mark-28` from each task's inputs, sample 40** ([report](../reports/stack-pbe-v3-20260828/README.md), [guard report](../reports/mark-28-guard-20260901/README.md)):

| depth | tasks with a fit | spurious (shown-only) fits | states/task unpruned | states/task, `close` guard | wall unpruned → guarded |
|---:|---:|---:|---:|---:|---|
| 4 | 1/40 | 0 | 41,917 | 9,742 (4.30x) | 78 s → 39 s |
| 5 | 7/40 | 0 | 1,001,797 | 110,266 (9.09x) | 1,851 s → 682 s |

Guard `body` alone: 1.30x / 1.71x / 2.30x at depths 4/5/6. Authored witnesses pruned: 0 of 705 (v3), 0 of 26,013 (v4).

## Stack PBE v4 — 220 behaviours, eval split 1,800 tasks

`stack_pbe_v4`, `mark-28`, witness median 43 ops. [report](../reports/stack-pbe-v4-baseline-20260901/README.md), [local-arm report](../reports/stack-pbe-v4-local-arm-20260902/README.md), [rerun-baselines-v4 (#85)](../../issues/85).

| solver | config | tasks | solved | memorised | wall (s) | notes | artifact |
|---|---|---:|---:|---:|---:|---|---|
| BFS | 1,000 nodes | full 1,800 | 80 (4.4%) | 8 | 2,116 | | `bfs_v4_1k_full.json` |
| BFS | 1,000 nodes, guard `both` | full 1,800 | 80 (4.4%) | 8 | 2,154 | identical verdicts | `bfs_v4_1k_full_guard.json` |
| BFS | 10,000 nodes | full 1,800 | 120 (6.7%) | 22 | 22,803 | 0/1,560 at 6+ ops | `bfs_v4_10k_full.json` |
| BFS | 10,000 nodes, guard `both` | full 1,800 | 120 (6.7%) | 24 | 19,558 | same 120 solves | `bfs_v4_10k_full_guard.json` |
| BFS | 100,000 nodes | sample 40 | 6 (15.0%) | 2 | 3,917 | 1 is the `aligned_matches` shortcut | `bfs_v4_100k_sample40.json` |
| BFS | 100,000 nodes, guard `both` | sample 40 | 6 (15.0%) | 2 | 3,624 | identical verdicts | `bfs_v4_100k_sample40_guard.json` |
| `gpt-5.6-luna` | effort high | sample 40 | **15 (37.5%)** | 3 | 2,286 | 1 is the shortcut → 14 real; 18/22 failures do not run | `luna_v4_sample40.json` |
| `qwen3:30b-a3b` | 300 s deadline | sample 40 | 6 (15.0%) | 0 | 3,381 | 0/26 at 16+ ops; 30/34 failures do not run | `qwen3_v4_sample40.json` |
| `gpt-oss:20b` | effort high, 32k ctx | 34 of 40 (capped) | 6 (17.6%) | 0 | 6,119 | 23/34 context-truncated, all empty | `gptoss20b_high_v4_sample40.json` |
| `gpt-oss:20b` | effort high, 131k ctx | sample 40 | 9 (22.5%) | 0 | 8,873 | 21/40 hit the 300 s deadline; 1 is the shortcut → 8 real | `gptoss20b_high_ctx131k_v4_sample40.json` |

Depth-5 states per task (`#70` probe, 3 tasks, seed 17): v4 roots **1,166,586** unpruned, **157,931** guarded (7.39x); v3 roots 980,688 / 105,549 (9.29x). `states-depth5-v4.json`.

Corpus scope numbers (v3 → v4): distinct input stacks 15,550 → 83,488; witness median 6.5 → 43 ops; witnesses with a branch — → 76%; median 8-op-window overlap between witnesses 79.8%, eval-with-train 76.8%.

## `deep-corpus-v1` — solvers on the held-out cohort (96 tasks, 12 per distance 1–8)

`wl-core-v1`, 118 functions. [materialized-rollout report](../reports/materialized-rollout-20260829/README.md), [fine-tune-llm (#50)](../../issues/50), [solve-step-by-step (#55)](../../issues/55).

| solver | solved /96 | d5–8 /48 | by distance d5/d6/d7/d8 | cost note |
|---|---:|---:|---|---|
| BFS | step function: 100% to depth cap, 0% past it | — | — | distance-8 ≈ 4.4M executions |
| `opt-125m`, 3-epoch SFT, k=64 | 14 | 0 | 0/12 each | cheaper than BFS at d3–4 |
| `SmolLM2-360M`, same protocol, k=64 | 4 | 0 | 0/12 each | 42.4% candidates unparseable (opt: 0.4%) |
| aligned-pair rollout policy, 90,374 params, beam 64 | **60** | **12** | 7 / 5 / 0 / 0 | 6,238 mean executions at d5–8; BFS at that spend 1/48; 10.74x executions to match; ~4.5x wall clock |

## `deep-corpus-v1` — probes ([#33](../../issues/33), reports `policy-probe-*`, `value-probe-20260825`)

| probe | lookup table | best encoder | note |
|---|---|---|---|
| policy, first action | rank 3.18 / top-1 42.1% | −10% rank, tie top-1 | type mask 9.50 of 118 at k=0, 18.41 at k=1 |
| policy, last action | rank 4.80 / top-1 22.3% | conv **3.43 / 42.1%** | mask 25.87 at the back |
| policy, inward k=1,2 | — | conv beats dict by 16–25% rank, 23–88% top-1 | both ends |
| policy, untruncated (1024 bytes) | — | back +1.06–1.46 rank, +14.6–17.2 top-1 vs 256-byte cut | 46.0% of tasks overran 256 bytes |
| policy, aligned example pairs (90,374 params) | — | **back 1.79 rank / 71.0% top-1 / 1.35 bits/step** | +0.25–0.61 rank over widened encoder at all six probes; bits saved over mask: front 1.29–1.79, back 3.08–3.25 |
| value, distance-to-go | MAE 1.232 | conv **0.978** | off by 1.66 / 2.08 / 2.51 at remaining 6/7/8; as orderer rank 3.35 (worse than policy lookup 3.18) |

## Substrates and bases

| measurement | value | source |
|---|---|---|
| Branching, first-order 18 | `b_eff` 11.14, 3.464 bits/step | #42 |
| Branching, `executable-26` | 13.38 / 3.742 bits / 7.3x growth | #42, #60 |
| Branching, `full-52` | 37.33 / 5.222 / 32.4x | #42, #60 |
| Branching, `mark-28` | 23.11 / 4.530 / 22.5x | #60 |
| Branching, `wl-core-v1` | 118 functions, 20.52 applicable, 4.292 bits/step; type mask ≈ 3.9 bits | #33 |
| Substrate comparison | new behaviour 5.87x/op vs 5.77x; 73% vs 59% of choice bits; stack pays median 2.00x length | #42 |
| Basis growth by MDL | 118 → 518 words, corpus −36.4%, depth-6 search 42,631x dearer | #41 |
| `mark-28` corpus probe, depth 5 | combinator share 6.7% → 7.0%, running code 12.4% both, generation 130 s → 1,364 s (10.5x) | [report](../reports/mark-28-corpus-20260828/README.md) |
| `mark-28` from data-only roots, depth 5, 4 roots | `executable-26`: 6,058 quotations, 0 executable, 33,468 combinator applications executing 0 ops; `mark-28`: 78,731 executable, 2,289 working | [report](../reports/stack-pbe-v3-20260828/README.md) |
| Undirected walks, `wl-core-v1` | 300 walks of 40 → 43–56 endpoints; 80-step walk within 6 of root 99%, modal distance 2 | #55 |
| Undirected walks, `executable-26` | 300/300 distinct endpoints; endpoint out-degree → 12.9; depth ~15 at 40 steps, ~29 at 80 | #60 (external implementation) |

## Cost

| measurement | value | source |
|---|---|---|
| Basis function share of a charged execution | ~15% (4.1 µs of 27 µs) | [report](../reports/inference-cost-20260830/README.md) |
| Policy scoring, beam 64, CUDA | flat 8.2–10.3 ms from 90K to 125M params | same |
| Policy scoring, beam 64, CPU | ~80x from 90K to 125M | same |
| CPU/CUDA crossover | ~1M–10M params | same |

## No result

| study | state |
|---|---|
| Scratch solver on `deep-corpus-v2` (10M, 125M) | trained (loss 0.755 / 0.686), weights lost, validation never sealed; no solve rate | [report](../reports/deep-corpus-v2-scratch-20260827-incomplete/README.md) |
| Compute-optimal policy size (#80) | in flight; rows land here when its evidence is on `main` | [#80](../../issues/80) |
