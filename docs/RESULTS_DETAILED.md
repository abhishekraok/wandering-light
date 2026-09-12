# Results, detailed

**Every measured number, one row per run.** The headline subset is in
[`RESULTS.md`](RESULTS.md); this file is the full ledger. Rows are copied from
the report artifact they cite, no argument. `STATUS.md` says what the numbers
mean; issues say why; `metrics.md` says how they are measured. A new run adds
a row here as part of the work, with its JSON under `reports/`, and to
`RESULTS.md` only if it is a headline.

Conventions: solve counts are graded on **held-out** examples unless a row
says otherwise; "sample 40" is the seed-17 40-task eval sample the baseline
scripts draw by default; wall clock is one host (`UltraDesk`, RTX 4090 for
local models, CPU for BFS) and comparable only within a table.

**What an artifact holds.** Each stack-PBE row names a JSON under its report
directory, written by `experiments/stack_pbe_baseline.py`: the run's config
(`backend`, `budget`, `guard`, `split`), the counts (`tasks`, `solved`,
`memorised`, `shown_fit`, `illegal`, the two rates), `by_behaviour` counts,
and `solved_programs`, the distinct solving programs with multiplicities.
Runs from `stack-pbe-v4-baseline-20260901` on, and the two bounded v3 qwen
runs, also carry `candidates`: one entry per task with the behaviour, the
authored witness, the program the solver returned, and its verdict
(`solved` / `memorised` / `failed`), so failed attempts are inspectable. The
older v3 BFS, terra and unbounded-qwen JSONs predate that field and keep only
the solved programs. No run stores the model's raw text or reasoning. The
`.txt` or `.log` beside each JSON is the console summary; v4 local-arm logs
also carry per-task timing and deadline hits.

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

`stack_pbe_v4`, `mark-28`, witness median 43 ops. [report](../reports/stack-pbe-v4-baseline-20260901/README.md), [local-arm report](../reports/stack-pbe-v4-local-arm-20260902/README.md), [codex-exec report](../reports/stack-pbe-v4-codex-exec-20260910/README.md), [rerun-baselines-v4 (#85)](../../issues/85), [codex-exec-frontier-effort-curve (#123)](../../issues/123). **The `codex exec` rows were run with the invoking user's `~/.codex` in context** (a generator bug, fixed after the runs); a post-fix replicate of one arm agrees 37/40, see the report's correction section.

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
| `gpt-5.6-luna` | effort high, via `codex exec` trimmed to the API protocol | sample 40 | 12 (30.0%) | 6 | 2,936 | anchor against the API row: paired agreement 33/40; 1 is the shortcut → 11 real; 17 do not run | `stack-pbe-v4-codex-exec-20260910/luna_high_v4_sample40.json` |
| `gpt-5.6-terra` | effort high, via `codex exec` | sample 40 | 16 (40.0%) | 4 | 1,344 | 1 is the shortcut → 15 real; 1/12 at 41+ ops | `…/terra_high_v4_sample40.json` |
| `gpt-6-astra` | effort low, via `codex exec` | sample 40 | 24 (60.0%) | 16 | 606 | shown-only 100%; 1 is the shortcut → 23 real; median 256 reasoning tok/task | `…/astra_low_v4_sample40.json` |
| `gpt-6-astra` | effort medium, via `codex exec` | sample 40 | 24 (60.0%) | 16 | 661 | shown-only 100%; 1 is the shortcut → 23 real; 278 reasoning tok/task | `…/astra_medium_v4_sample40.json` |
| `gpt-6-astra` | effort high, via `codex exec` | sample 40 | **23 (57.5%)** | 15 | 840 | shown-only 95%; 1 is the shortcut → 22 real; 396 reasoning tok/task, effort barely engaged | `…/astra_high_v4_sample40.json` |
| `gpt-6-astra` | effort low, via `codex exec`, 6 shards | full 1,800 | **995 (55.3%)** | 714 | 33,487 | shown-only 94.9%; 20 are the shortcut → 975 real; 214/840 at 41+ ops, `median_floor` (253) 17/40; hand-written 318/320 vs sol-written 677/1,480, matched only at 1–10 ops (279/280 vs 79/80) | `…/astra_low_v4_full_shard{0..5}.json`, `…/astra_low_v4_full_analysis.md` |
| `gpt-6-astra` | effort low, via `codex exec`, private home actually applied | sample 40 | 23 (57.5%) | 17 | 601 | replicate after the CODEX_HOME fix; paired agreement 37/40 with the row above, 480 fewer input tok/task | `…/astra_low_v4_sample40_fixedhome.json` |
| beam 64 × depth 48, 1m policy, seed 0 | trained on v4 train, LR 1.25e-4, cpu (8 threads) | sample 40 | 7 (17.5%) | 0 | 131 |  | `compute-optimal-mark28-20260903/timing/1m-s0-cpu.json` |
| beam 64 × depth 48, 1m policy, seed 0 | trained on v4 train, LR 1.25e-4, cuda | sample 40 | 7 (17.5%) | 0 | 120 | same program on both devices | `…/timing/1m-s0-cuda.json` |
| beam 64 × depth 48, 1m policy, seed 1 | trained on v4 train, LR 1.25e-4, cpu (8 threads) | sample 40 | 10 (25.0%) | 1 | 126 |  | `…/timing/1m-s1-cpu.json` |
| beam 64 × depth 48, 1m policy, seed 1 | trained on v4 train, LR 1.25e-4, cuda | sample 40 | 10 (25.0%) | 1 | 107 | same program on both devices | `…/timing/1m-s1-cuda.json` |
| beam 64 × depth 48, 2m policy, seed 0 | trained on v4 train, LR 6.25e-5, cpu (8 threads) | sample 40 | 7 (17.5%) | 0 | 177 |  | `…/timing/2m-s0-cpu.json` |
| beam 64 × depth 48, 2m policy, seed 0 | trained on v4 train, LR 6.25e-5, cuda | sample 40 | 7 (17.5%) | 0 | 123 | same program on both devices | `…/timing/2m-s0-cuda.json` |
| beam 64 × depth 48, 2m policy, seed 1 | trained on v4 train, LR 6.25e-5, cpu (8 threads) | sample 40 | 9 (22.5%) | 1 | 175 |  | `…/timing/2m-s1-cpu.json` |
| beam 64 × depth 48, 2m policy, seed 1 | trained on v4 train, LR 6.25e-5, cuda | sample 40 | 9 (22.5%) | 1 | 120 | same program on both devices | `…/timing/2m-s1-cuda.json` |
| beam 64 × depth 48, 10m policy, seed 0 | trained on v4 train, LR 3.1e-5, cpu (8 threads) | sample 40 | 6 (15.0%) | 0 | 303 |  | `…/timing/10m-s0-cpu.json` |
| beam 64 × depth 48, 10m policy, seed 0 | trained on v4 train, LR 3.1e-5, cuda | sample 40 | 6 (15.0%) | 0 | 146 | same program on both devices | `…/timing/10m-s0-cuda.json` |
| beam 64 × depth 48, 10m policy, seed 1 | trained on v4 train, LR 3.1e-5, cpu (8 threads) | sample 40 | 8 (20.0%) | 0 | 290 |  | `…/timing/10m-s1-cpu.json` |
| beam 64 × depth 48, 10m policy, seed 1 | trained on v4 train, LR 3.1e-5, cuda | sample 40 | 8 (20.0%) | 0 | 146 | same program on both devices | `…/timing/10m-s1-cuda.json` |

Depth-5 states per task (`#70` probe, 3 tasks, seed 17): v4 roots **1,166,586** unpruned, **157,931** guarded (7.39x); v3 roots 980,688 / 105,549 (9.29x). `states-depth5-v4.json`.

Corpus scope numbers (v3 → v4): distinct input stacks 15,550 → 83,488; witness median 6.5 → 43 ops; witnesses with a branch — → 76%; median 8-op-window overlap between witnesses 79.8%, eval-with-train 76.8%.

## World-behaviours probe (#118): `mark-28` witnesses for MBPP and LeetCode-easy

95 behaviours sampled from MBPP (50) and LeetCode-easy (45) with every argument and result an int or flat int list, ≤ 3 arguments, ≥ 10 tests (MBPP's 3 asserts extended by shape-mutated inputs run through the reference solution); programs authored by four Claude sessions in macro form, expanded to `mark-28`, re-verified on every test by `experiments/world_behaviours/report.py`. Lengths are expanded `mark-28` ops. [report](../reports/world-behaviours-probe-20260909/README.md), `report.json`, `behaviours.jsonl`, `programs_<k>.jsonl`; macros in `experiments/world_behaviours/macros*.txt`.

| source | verified on every test | median witness length | median macro-form length | unsolved, reason |
|---|---|---|---|---|
| MBPP | 49/50 (98%) | 115 | 13 | 1: output is CPython set-iteration order |
| LeetCode-easy | 42/45 (93%) | 99.5 | 23.5 | 3: fuel, tests enumerate up to 90,002 numbers against the 100,000-step cap |
| all | 91/95 (96%) | 102 (v4, both splits: 43) | 17 | 4 |

Length buckets, verified probe / v4 behaviours (train + eval, 220): 1–5 ops 6 / 15, 6–10 3 / 14, 11–20 8 / 34, 21–40 8 / 37, 41+ 66 / 120. Exact v4 witness matches 5 of 91; same skeleton 5. 129 macros defined; median expansion 4.6 ops per macro token. Six solves replay reference quirks rather than the description (listed in the report). `MAX_QUOTATION_DEPTH = 32` caps a self-applying loop at ~7 iterations; every author rebuilt ranges as recursion-free folds.

## world-authoring-500 (#127), wave 1: Codex-authored witnesses on the wider pool

100 behaviours from the pool the probe did not touch (36 MBPP, 34 CodeARC, 30 LeetCode medium/hard; `extract.py --take mbpp=36,codearc=34,leetcode=30 --leetcode-difficulty Medium,Hard --exclude <probe ids>`, seed 17), every source extended by the hypothesis-breaking sampler; authored by four `codex exec` agents (gpt-6-astra, reasoning effort high, one worktree each, ~25 behaviours mixed by source) in macro form, migrated to the canonical library (`migrate.py` + `aliases.txt` + `repairs.jsonl`), re-verified on every test by `report.py`. [report](../reports/world-authoring-20260910/README.md), `report.json`, `programs_canonical.jsonl`; raw author outputs, macro files, run accounting and Codex transcripts under `authors/`.

| source | verified on every test | median witness length | median macro-form length | unsolved, reason |
|---|---|---|---|---|
| MBPP | 35/36 (97%) | 143 | 16 | 1: output is CPython set-iteration order (`mbpp-769`) |
| CodeARC | 34/34 (100%) | 175.5 | 24.5 | 0 |
| LeetCode-medium | 21/25 (84%) | 147 | 37 | 4: fuel (DP / enumeration with n in the thousands) |
| LeetCode-hard | 2/5 (40%) | 1080 | 242 | 3: fuel |
| all | 92/100 (92%) | 164 (probe 102, v1 111, v4 43) | 24.5 | 8 |

Authoring cost: 372k tokens and 40 agent-minutes for 100 behaviours (86k–111k tokens and 7–14 min per agent of 24–26), 14 min wall on four agents; claimed and verified statuses agreed on all 100. Library: authors wrote 128 private macros, 16 went canonical (94 → 110; three authors independently wrote `isqrt`, two `palindrome?`, two `pack5`), 112 inlined. Canonical `gcd` was a self-applying loop refusing past ~7 Euclid steps (`gcd 832040 514229`) and is now a 32-step fold at ~7× the fuel (700 vs 105 per call); that broke one fuel-tight witness, repaired by Codex (22k tokens) with a coprime test exact on the problem's stated bound n ≤ 12. v1's 91 witnesses re-verify under the merged library. Exact v4 witness matches 1 of 92 (`mbpp-517`), shared skeletons 4. Length buckets, verified / v4 (220): 1–5 ops 3 / 15, 6–10 5 / 14, 11–20 3 / 34, 21–40 3 / 37, 41+ 78 / 120.

`stack_pbe_world_v2` = v1's 91 + these 92 = 183 tasks, 168 train / 15 eval by behaviour-id hash (10%), 1,098 shown and 5,034 held-out examples, 4,696 distinct input stacks; manifest pinned at `wandering_light/evals/data/stack_pbe_world_v2/`.

Later waves, same settings (four `codex exec` agents, gpt-6-astra high, ~25 behaviours each; verified after migration to the canonical library):

| wave | report | sources | verified | median ops / macro tokens | tokens, wall | macros canonical / written | unsolved |
|---|---|---|---|---|---|---|---|
| 1 (2026-09-10) | [w1](../reports/world-authoring-20260910/README.md) | 36 MBPP, 34 CodeARC, 30 LeetCode med/hard | 92/100 | 164 / 24.5 | 372k, 14 min | 16 / 128 (94 → 110) | 7 fuel (LeetCode), 1 set order |
| 2 (2026-09-11) | [w2](../reports/world-authoring-20260911-w2/README.md) | 50 MBPP, 50 CodeARC | 97/100 (MBPP 50/50, CodeARC 47/50) | 144 / 17 | 289k, 11 min | 7 / 90 (110 → 117) | 3 fuel (CodeARC); one identity repaired from an empty program |
| 3 (2026-09-11) | [w3](../reports/world-authoring-20260911-w3/README.md) | 50 MBPP, 50 CodeARC | 99/100 (MBPP 50/50, CodeARC 49/50) | 151 / 20 | 277k, 6 min | 8 / 106 (117 → 125) | 1 fuel (CodeARC) |
| 4 (2026-09-11) | [w4](../reports/world-authoring-20260911-w4/README.md) | 50 MBPP, 50 CodeARC | 100/100 | 126 / 18.5 | 432k, 7 min | 4 / 68 (125 → 129) | 0; one identity repaired from an empty program; 9 exact v4 witness matches |
| 5 (2026-09-11) | [w5](../reports/world-authoring-20260911-w5/README.md) | 50 MBPP, 50 CodeARC | 98/100 (MBPP 49/50, CodeARC 49/50) | 128 / 16.5 | 318k, 10 min | 4 / 59 (129 → 133) | 1 set order (MBPP), 1 fuel (CodeARC); 7 exact v4 witness matches |

Totals over the five waves: 486/500 verified (MBPP 234/236, CodeARC 229/234, LeetCode medium 21/25, hard 2/5), 1.69M tokens, 48 minutes wall, library 94 → 133. `stack_pbe_world_v3` = the probe's 91 + 486 = 577 tasks, 522 train / 55 eval by behaviour-id hash (10%), 3,462 shown and 13,788 held-out examples, 12,853 distinct input stacks, median witness 134 ops; every earlier report regenerated under the final library before the build (all counts unchanged). Manifest pinned at `wandering_light/evals/data/stack_pbe_world_v3/`.


## Library hygiene (#124): canonical macro library and hypothesis-breaking MBPP examples

The 91 probe witnesses rewritten from four author dialects (129 macros) to one canonical library (94 macros, `experiments/world_behaviours/macros.txt`, every macro with an executed test) by `migrate.py` + `aliases.txt` + `repairs.jsonl`, and re-verified on behaviours regenerated by the invariant-as-hypothesis sampler (`extract.py --ids-from`, same 95 ids, seed 17): 60% of generated candidates keep every property the given asserts share, the rest violate one in turn; any one output value is capped at 40% of the generated set. [report](../reports/world-behaviours-library-20260910/README.md), `report.json`, `programs_canonical.jsonl`, `repairs.jsonl`; author files and Codex repair transcripts under `authors/`.

| | before (2026-09-09 probe) | after |
|---|---|---|
| verified on every test | 91/95 | 91/95 (32 expansions changed; **11 witnesses repaired**, all MBPP: they matched the reference only on inputs satisfying the asserts' incidental properties) |
| median witness length, `mark-28` ops | 102 | 111 |
| `range 1000` fuel | 29,064 (recursive `range` refused past n = 7; author variants 5k–23k) | 4,766, valid to 65,536 |
| `dedup sort` on the two sorting behaviours (`mbpp-152`, `mbpp-54`) | 27/27 and 26/27 under strict inferred `distinct` (Codex review); 8/27 and 14/27 on the original examples | 24/27 and 21/27 |
| MBPP held-out: median share of the most common output | 0.23 | 0.10 |
| MBPP behaviours with one output above half the held-out set | 10 | 0 |
| violating examples per MBPP behaviour | 0 | median 9 (by property: nonneg 144, positive 107, distinct 69, min_len 31, bounded_by 19, sorted 8) |
| held-out examples, all 91 tasks | 2,254 | 2,332 |
| `base-digit-sum(128, 2)`, `band(65536, 65536)` | 0, 0 (7-digit / 16-bit helpers) | 1, 65536 (20 digits / 32 bits, contracts stated) |

Corpus republished at Hub revision `fa60b4292b273b6c24072bf895a71b031f318127` (91 tasks, all re-graded solved, 2,244 distinct input stacks). Repairs: two `codex exec` agents (gpt-6-astra, effort high), five witnesses each, ~29k tokens each, 10/10 verified; one more (`mbpp-4`, n outside [0, len]) after the re-review fix that let index relations to argument 0 be violated.

## Stack PBE v4 — compute-optimal policy size (#101), 300-task eval cohort

`stack_pbe_v4` eval, 60 tasks per witness-length bucket (seed-17 shuffle, first 60 per bucket; contains sample 40), beam 64 × depth 48, guard `both`, two seeds per arm, CPU (8 threads) and CUDA cells. Solve rates are graded on held-out examples; the solve-rate table is the CPU cells, and CUDA agrees on 3,599 of 3,600 task-cells (10m seed 1 misses one `negate_each` task on CUDA, giving 0.033 at 6–10 and coverage 0.180 there). AUC is normalised area under solve rate against log₂ budget: wall-clock on [0.01, 100] s; executions (one per op run on one example stack) on [16, 524,288], computed from every hit replayed under the corrected counter (0 program mismatches). [report](../reports/compute-optimal-mark28-20260903/README.md), `aggregate.json`, `timing/<arm>-s<seed>-<device>.json`.

**Solve rate at max budget, mean of two seeds (seed SD):**

| bucket | 1m | 2m | 10m |
|---|---:|---:|---:|
| 1–5 | 0.608 (0.075) | 0.592 (0.075) | 0.617 (0.033) |
| 6–10 | 0.050 (0.050) | 0.000 (0.000) | 0.042 (0.025) |
| 11–20 | 0.267 (0.000) | 0.267 (0.017) | 0.233 (0.050) |
| 21–40 | 0.000 | 0.000 | 0.000 |
| 41+ | 0.000 | 0.017 (0.017) | 0.017 (0.000) |

**Normalised AUC, mean of two seeds:**

| bucket | device | 1m wall | 2m wall | 10m wall | 1m exec | 2m exec | 10m exec |
|---|---|---:|---:|---:|---:|---:|---:|
| 1–5 | cpu | 0.547 | 0.521 | 0.507 | 0.315 | 0.312 | 0.315 |
| 1–5 | cuda | 0.545 | 0.532 | 0.545 | 0.315 | 0.312 | 0.315 |
| 6–10 | cpu | 0.036 | 0.000 | 0.025 | 0.015 | 0.000 | 0.012 |
| 6–10 | cuda | 0.036 | 0.000 | 0.023 | 0.015 | 0.000 | 0.009 |
| 11–20 | cpu | 0.157 | 0.149 | 0.120 | 0.056 | 0.056 | 0.047 |
| 11–20 | cuda | 0.157 | 0.160 | 0.134 | 0.056 | 0.056 | 0.047 |
| 21–40, 41+ | both | 0.000 | ≤ 0.006 | ≤ 0.007 | 0.000 | ≤ 0.001 | ≤ 0.002 |

**Cost to clear the cohort (coverage, wall-clock over the covered subset, total spent), mean of two seeds:**

| device | 1m | 2m | 10m |
|---|---|---|---|
| cpu | 0.185, 10 s, 1,009 s | 0.175, 18 s, 1,374 s | 0.182, 27 s, 2,198 s |
| cuda | 0.185, 11 s, 887 s | 0.175, 13 s, 940 s | 0.180, 15 s, 1,074 s |

**Training** (plain Adam, constant LR, batch 256, best-validation weights; validation is 1 in 10 train skeletons, 26,040 rows; 313,560 training rows). Weights are W&B model artifacts `abhishekraok-na/wandering-light-rollout-policy/mark28-policy-<arm>-s<seed>:v0` on each checkpoint's run (`archive.json`).

| arm | dims (embed, conv, hidden) | params | LR | seed 0 best val CE (epoch) | seed 1 | epochs run |
|---|---|---:|---:|---:|---:|---:|
| 1m | 128, 320, 640 | 655,836 | 1.25e-4 | 0.952 (1) | 0.945 (2) | 12 / 12 |
| 2m | 224, 576, 1152 | 2,045,660 | 6.25e-5 | 0.939 (1) | 0.976 (2) | 12 / 3 |
| 10m | 384, 1024, 2048 | 6,288,412 | 3.1e-5 | 0.925 (2) | 0.934 (2) | 3 / 3 |

LR per arm from a one-epoch probe (`lr_probe.json`), extended three halvings below the #80 grid; each arm's best is the bottom of its extended grid. 1m seed 0 over 12 epochs: train CE 1.09 → 0.10, validation 0.95 → 1.99. Two seeds where `metrics.md` asks three, as #80.

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
| Undirected walks, `executable-26` | 4 roots × 300 walks: 300/300 distinct endpoints at 40 and 80 steps; inside the depth-5 ball 100% at 4 steps, 87% at 6, 57% at 10, 17% at 20, 1% at 40, 0% at 80; stack depth 15.5 at 40, 29.5 at 80; endpoint out-degree 12.3–12.6; empty-root ball 9,503 (= #41's pin) | [report](../reports/walk-collapse-stack-20260908/README.md), #60 |
| Undirected walks, `mark-28`, no guard | same protocol: 300/300 distinct; inside the depth-5 ball 100% / 59–66% / 25–34% / 5–9% / 1–2% / 0–2% at 4 / 6 / 10 / 20 / 40 / 80 steps; stack depth 23.5 at 40, ~40 at 80; out-degree 24.0–24.9; **54–58% of endpoints have a `sym` on top** (inside an open `[`); empty-root ball 726,064 | [report](../reports/walk-collapse-stack-20260908/README.md), #60 |
| Undirected walks, `mark-28`, guard `both` | same protocol under the #97 guard: 300/300 distinct; inside the ball 100% / 64–73% / 33–34% / 5–9% / 0–3% / 0% at 4 / 6 / 10 / 20 / 40 / 80 steps; stack depth 8.3–9.7 at 40, 12.0–13.5 at 80; out-degree 14.6–14.7; 76–78% of endpoints have a list on top, ≤1% a `sym` | [report](../reports/walk-collapse-stack-20260908/README.md), #60 |

## Cost

| measurement | value | source |
|---|---|---|
| Basis function share of a charged execution | ~15% (4.1 µs of 27 µs) | [report](../reports/inference-cost-20260830/README.md) |
| Policy scoring, beam 64, CUDA | flat 8.2–10.3 ms from 90K to 125M params | same |
| Policy scoring, beam 64, CPU | ~80x from 90K to 125M | same |
| CPU/CUDA crossover | ~1M–10M params | same |
| Unsolved task, beam 64 × depth 48 on `mark-28` (2,957 expansions) | 1m / 2m / 10m: 4.0 / 5.5 / 8.9 s CPU (8 threads), 3.0 / 3.1 / 3.6 s CUDA | [report](../reports/compute-optimal-mark28-20260903/README.md) |

## Legacy typed-list line (2026-08-10)

The pre-pivot solver and proposer, before `deep-corpus-v1`. No report
directory; these rows were carried by the README (commits `d70d143`, `864ec0a`)
and moved here when it was rewritten. Basis `wl-core-pyhash-v1`, the alias the
checkpoint was trained against; a solve is an exact output match, so the rate
is an upper bound on "finds the shortest path". The checkpoint predates
`str_hash`/`set_hash`/`set_to_list` being made deterministic, so it was scored
against three definitions it never saw.

| measurement | value | source |
|---|---|---|
| `abhishekraok/induction-basicfns-opt125m-sft434k-rl-6k-with-lp` (OPT-125M, SFT then GRPO with a length penalty) on `random_inputs_500_shortest_v1`, 480 certified specs, budget 1 / 2 / 4 / 8 / 16 | 83.96 / 84.38 / 85.21 / 85.00 / 85.42 % | README at `864ec0a` |
| same checkpoint on `random_inputs_500`, 496 specs, budget 1 / 2 / 4 / 8 / 16 | 85.48 / 85.08 / 85.28 / 85.69 / 86.29 % | same |
| Best-of-K | 16x the compute adds 7 solves of 480, non-monotonically; residual failures are systematic | same |
| Solution length, certified eval | mean solve 1.83 functions against 1.84 ground truth; the 1.81-vs-3.00 gap on `random_inputs_500` was its random-walk labels, 62.7 % of which certification shortened | same |
| `abhishekraok/proposer-basicfns-opt125m-sft2k` scored against that solver | parse 0.96, solver success 0.15, 31 % of groups with non-zero reward std (69 % give GRPO no gradient); measured before the palette fix, never re-measured | README at `d70d143` |

## No result

| study | state |
|---|---|
| Scratch solver on `deep-corpus-v2` (10M, 125M) | trained (loss 0.755 / 0.686), weights lost, validation never sealed; no solve rate | [report](../reports/deep-corpus-v2-scratch-20260827-incomplete/README.md) |
| Compute-optimal policy size (#80) | merged (#100, `05fe25d`) with its report; rows not yet copied here | [report](../reports/compute-optimal-20260903/README.md), [#80](../../issues/80) |
