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

`stack_pbe_v4`, `mark-28`, witness median 43 ops. [report](../reports/stack-pbe-v4-baseline-20260901/README.md), [local-arm report](../reports/stack-pbe-v4-local-arm-20260902/README.md), [codex-exec report](../reports/stack-pbe-v4-codex-exec-20260910/README.md), [claude-exec report](../reports/stack-pbe-v4-claude-exec-20260913/README.md), [rerun-baselines-v4 (#85)](../../issues/85), [codex-exec-frontier-effort-curve (#123)](../../issues/123), [opus-effort-curve (#137)](../../issues/137). **The `codex exec` rows were run with the invoking user's `~/.codex` in context** (a generator bug, fixed after the runs); a post-fix replicate of one arm agrees 37/40, see the report's correction section. A paired, per-example regrade of the luna API row against the astra high row, with per-op traces of every candidate, is in [run-comparison](../reports/stack-pbe-v4-run-comparison-20260913/README.md).

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
| `claude-opus-5` | effort low, via `claude -p` trimmed to the API protocol | sample 40 | 17 (42.5%) | 6 | 1,424 | shown-only 57.5%; 1 is the shortcut → 16 real; 13 do not run; median 526 thinking tok/task; 0 tool calls in 40 | `stack-pbe-v4-claude-exec-20260913/opus_low_v4_sample40.json` |
| `claude-opus-5` | effort high, via `claude -p` | sample 40 | 20 (50.0%) | 12 | 2,220 | shown-only 80%; 1 is the shortcut → 19 real; 5 do not run; 1,400 thinking tok/task, above low on 38/40 paired tasks | `…/opus_high_v4_sample40.json` |
| `claude-opus-5` | effort xhigh, via `claude -p` | sample 40 | **25 (62.5%)** | 13 | 3,568 | shown-only 95%; 1 is the shortcut → 24 real; 1 does not run; 1,962 thinking tok/task (max 119,941 on `median_floor`, memorised); agrees with astra high on 38/40, astra-only 0, at 12.5× its output tokens | `…/opus_xhigh_v4_sample40.json` |
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

## Stack PBE v4 + world v3 — world-authored data under the #101 recipe (#134), 300-task v4 cohort

The #101 `1m` next-op policy (655,836 params, Adam 1.25e-4, batch 256, 6 epochs, CUDA, seeds 0 / 1 / 2) trained on `stack_pbe_v4` train alone (v4 only) and on v4 train plus `stack_pbe_world_v3` train less three behaviours whose witness equals a v4 eval witness (v4 + world). Both select on the same v4 holdout, 26,040 rows (`--validation-corpus stack_pbe_v4`); fit rows are 313,560 against 429,078, so 1,225 against 1,677 steps per epoch. Search is #101's 300-task cohort, beam 64 × depth 48, guard `both`. Weights were saved at every epoch (each seed's snapshot at its best epoch bitwise equal to its best-validation checkpoint) and every epoch was searched. A verified solve passes held-out grading and is a catalogue witness or agrees with the Python reference on 10,000 draws in each of three modes (the behaviour's sampler; random values in −20..20, lengths 0–10; random values in −2..2, lengths 0–10), counted where the catalogue witness computes the reference, at least 500 per mode. Contained: the eval witness occurs as whole ops inside a world fit program and inside no v4 fit program. [report](../reports/world-plus-v4-policy-20260912/README.md), `paired.json`, `epochs.json`, `containment.json`, `verified_solves_epochs.json`, `ce.json`.

**Best validation CE on the v4 holdout (epoch):**

| seed | v4 only | v4 + world | difference |
|---|---:|---:|---:|
| 0 | 0.9519 (1) | 0.8434 (2) | −0.1085 |
| 1 | 0.9454 (2) | 0.8689 (4) | −0.0765 |
| 2 | 0.9650 (1) | 0.8606 (3) | −0.1044 |
| mean ± sd | 0.9541 ± 0.0100 | 0.8576 ± 0.0130 | −0.0965 ± 0.0174 |

A stack-blind op 4-gram gains 0.088 nats on the same holdout when world fit programs join its training text (1.280 to 1.192).

**Verified solves by epoch (verified / graded; validation CE; `*` the best-validation checkpoint):**

| epoch | v4-only s0 | v4-only s1 | v4-only s2 | v4+world s0 | v4+world s1 | v4+world s2 |
|---|---|---|---|---|---|---|
| 1 | 48/48* (0.952) | 56/56 (0.965) | 46/48* (0.965) | 53/53 (0.898) | 43/43 (0.952) | 57/59 (0.920) |
| 2 | 55/55 (0.956) | 63/63* (0.945) | 53/54 (0.997) | 52/53* (0.843) | 64/66 (0.900) | 82/82 (0.880) |
| 3 | 84/85 (1.094) | 77/77 (1.061) | 80/80 (1.099) | 89/89 (1.001) | 89/89 (0.876) | 102/103* (0.861) |
| 4 | 94/94 (1.210) | 74/74 (1.187) | 88/89 (1.234) | 85/86 (0.983) | 104/105* (0.869) | 94/97 (0.872) |
| 5 | 83/84 (1.362) | 77/80 (1.257) | 90/92 (1.286) | 104/109 (0.935) | 109/111 (0.879) | 90/91 (0.901) |
| 6 | 86/91 (1.509) | 72/72 (1.358) | 75/76 (1.412) | 75/77 (1.012) | 70/70 (0.931) | 80/82 (0.936) |

**Matched epoch, matched steps and each seed's peak:**

Matched epoch, v4 + world minus v4 only:

| epoch | s0 | s1 | s2 | mean |
|---|---|---|---|---|
| 1 | +5 | -13 | +11 | +1.0 |
| 2 | -3 | +1 | +29 | +9.0 |
| 3 | +5 | +12 | +22 | +13.0 |
| 4 | -9 | +30 | +6 | +9.0 |
| 5 | +21 | +32 | +0 | +17.7 |
| 6 | -11 | -2 | +5 | -2.7 |

Matched steps (nearest v4-only epoch within half an epoch; gap is v4-only minus v4 + world steps):

| v4+world epoch | v4-only epoch | step gap | s0 | s1 | s2 | mean |
|---|---|---|---|---|---|---|
| 1 | 1 | -452 | +5 | -13 | +11 | +1.0 |
| 2 | 3 | +321 | -32 | -13 | +2 | -14.3 |
| 3 | 4 | -131 | -5 | +15 | +14 | +8.0 |
| 4 | 5 | -583 | +2 | +27 | +4 | +11.0 |

Each seed's best searched epoch, chosen on this cohort (an upper bound for both arms), and the mean over epochs:

| summary | s0 | s1 | s2 | mean |
|---|---|---|---|---|
| v4-only peak (epoch) | 94 (4) | 77 (3) | 90 (5) | 87.0 |
| v4+world peak (epoch) | 104 (5) | 109 (5) | 102 (3) | 105.0 |
| difference | +10 | +32 | +12 | +18.0 |
| v4-only mean over epochs 1-6 | 75.0 | 69.8 | 72.0 | 72.3 |
| v4+world mean over epochs 1-6 | 76.3 | 79.8 | 84.2 | 80.1 |
| difference of means | +1.3 | +10.0 | +12.2 | +7.8 |

**By containment (5 of 43 cohort behaviours contained):**

| contained behaviour | witness ops | world fit programs containing it |
|---|---|---|
| `drop_last` | 12 | 1 |
| `increment_each` | 5 | 44 |
| `keep_below` | 6 | 156 |
| `n_minus_each` | 7 | 2 |
| `negate_each` | 6 | 3 |

Summed over seeds; the last column is the other 38 behaviours' difference per seed:

| epoch | contained v4-only | contained v4+world | difference | other v4-only | other v4+world | difference | other, per seed |
|---|---|---|---|---|---|---|---|
| 1 | 26 | 38 | +12 | 124 | 115 | -9 | +0, -9, +0 (mean -3.0) |
| 2 | 26 | 58 | +32 | 145 | 140 | -5 | -10, -4, +9 (mean -1.7) |
| 3 | 59 | 113 | +54 | 182 | 167 | -15 | -5, -16, +6 (mean -5.0) |
| 4 | 65 | 91 | +26 | 191 | 192 | +1 | -8, +5, +4 (mean +0.3) |
| 5 | 60 | 110 | +50 | 190 | 193 | +3 | -2, +1, +4 (mean +1.0) |
| 6 | 51 | 60 | +9 | 182 | 165 | -17 | -11, -7, +1 (mean -5.7) |
| each seed's peak | 67 | 125 | +58 | 194 | 190 | -4 | -5, +0, +1 (mean -1.3) |

Over epochs 1–6 `keep_below` is +186 of the contained +183 and the cohort's +141, and every v4 + world `keep_below` solve is the exact witness `[ lt? ] swap cons filter`; the other four contained behaviours net −3 and the other 38 −42 (`keep_negative` −36). 29 of the 43 cohort behaviours are never solved by either arm and 4 always are; all three 6–10 bucket behaviours are contained.

**Longest verified solves:** both arms verify `shared_value_occurrences`' 31-op catalogue witness (v4 only on every seed from epoch 3), `count_in_range`'s 40-op witness (v4 only: seed 0 epochs 4–5, seed 2 epoch 6) and a 32-op program for the 54-op `farther_choice_from_target` (v4 only: seed 0 epochs 3 and 6, seed 1 epoch 3). At the best-validation checkpoints only v4 + world reached the 21–40 bucket.

**Shortcuts** (pass held-out grading, fail the reference in every mode, including the task's own sampler): `all_positive` 17 and 28 ops and `bounds_cover_list_flag` 44 and 46 ops in both arms; `all_positive` 19 ops (v4 + world seed 1 epoch 5), `aligned_matches` 32 ops (v4 only seed 2 epoch 5) and `keep_negative` 17 ops (v4 + world seed 0 epoch 5) too; 36 distinct solved programs, 29 verified, none undecided.

**World v3 eval, 44 clean behaviours, best-validation checkpoints, same search:** 1 / 0 / 0 solved in both arms, on different tasks; 4 of the 44 witnesses fit within depth 48 (median 149.5 ops). v4-only seed 0 at depth 192: 1 / 44. Next-op CE there: 3.16 ± 0.61 v4 only, 1.02 ± 0.05 v4 + world, 0.91 for an op 4-gram fit on world fit programs.

## Stack PBE v4 + world v3 — dropping the 156 world programs that contain `keep_below`'s witness (#136), 300-task v4 cohort

The #134 v4 + world arm retrained with the 156 world fit behaviours whose program contains `[ lt? ] swap cons filter` added to the drop list (`ablated`, seeds 0 / 1 / 2, 358,086 fit rows) and with 156 world train behaviours dropped at random instead (`random`, seed 0, 389,527 fit rows; the draw keeps 114 of the 156 carriers). Everything else as #134: 1m, 6 epochs, snapshots and search at every epoch, validation pinned to the v4 holdout, beam 64 × depth 48, verification at 10,000 draws per mode. #134's arms are its epoch study, re-verified in the same file. [report](../reports/keep-below-ablation-20260913/README.md), `ablation.json`, `verified_solves_ablation.json`.

**Best validation CE on the v4 holdout (epoch), and the difference from v4 + world on the same seed:**

| seed | v4 only | v4 + world | ablated | random |
|---|---:|---:|---:|---:|
| 0 | 0.9519 (1) | 0.8434 (2) | 0.8716 (2), +0.0282 | 0.9135 (2), +0.0701 |
| 1 | 0.9454 (2) | 0.8689 (4) | 0.8657 (2), −0.0032 | |
| 2 | 0.9650 (1) | 0.8606 (3) | 0.8756 (2), +0.0151 | |
| mean ± sd | 0.9541 ± 0.0100 | 0.8576 ± 0.0130 | 0.8710 ± 0.0050 | |

**Verified solves by epoch (all / `keep_below`, of 300 and 19 tasks):**

| epoch | v4-only s0 | s1 | s2 | v4+world s0 | s1 | s2 | ablated s0 | s1 | s2 | random s0 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 48/0 | 56/0 | 46/0 | 53/7 | 43/0 | 57/11 | 48/0 | 55/0 | 47/0 | 50/3 |
| 2 | 55/0 | 63/0 | 53/0 | 52/7 | 64/15 | 82/15 | 57/0 | 64/0 | 65/0 | 67/17 |
| 3 | 84/0 | 77/0 | 80/0 | 89/11 | 89/19 | 102/16 | 58/0 | 73/1 | 82/0 | 77/16 |
| 4 | 94/0 | 74/0 | 88/0 | 85/7 | 104/8 | 94/16 | 80/0 | 90/1 | 78/1 | 86/17 |
| 5 | 83/0 | 77/0 | 90/0 | 104/8 | 109/19 | 90/8 | 75/0 | 100/2 | 83/1 | 102/19 |
| 6 | 86/0 | 72/0 | 75/0 | 75/1 | 70/7 | 80/11 | 61/0 | 94/1 | 78/0 | 108/15 |

Summed over the 18 cells of each three-seed arm: `keep_below` 0 / 186 / 7 (v4 only / v4 + world / ablated; all 7 the exact witness), the other 42 behaviours 1,301 / 1,256 / 1,281. Per-seed peaks 94 / 77 / 90 (87.0), 104 / 109 / 102 (105.0), 80 / 100 / 83 (87.7); means over epochs 72.3, 80.1, 71.6. Random seed 0: 87 `keep_below` solves over 6 cells, peak 108 at epoch 6, mean 81.7.

**Matched epoch, ablated minus v4 + world (all verified solves; with `keep_below` removed):**

| epoch | s0 | s1 | s2 | mean | other s0 | s1 | s2 | mean |
|---|---|---|---|---|---|---|---|---|
| 1 | −5 | +12 | −10 | −1.0 | +2 | +12 | +1 | +5.0 |
| 2 | +5 | +0 | −17 | −4.0 | +12 | +15 | −2 | +8.3 |
| 3 | −31 | −16 | −20 | −22.3 | −20 | +2 | −4 | −7.3 |
| 4 | −5 | −14 | −16 | −11.7 | +2 | −7 | −1 | −2.0 |
| 5 | −29 | −9 | −7 | −15.0 | −21 | +8 | +0 | −4.3 |
| 6 | −14 | +24 | −2 | +2.7 | −13 | +30 | +9 | +8.7 |

**Behaviours:** 17 of the 43 are verified by some arm across #134 and this run, 26 by none; the ablated arm verifies 17 (v4 only 13, v4 + world 14, random 15), three of them by no #134 arm: `bounds_cover_list_flag`, `n_minus_each`, `sum_negative`.

**Longest verified solve:** a 46-op `bounds_cover_list_flag` program, `[ [ drop lt? ] [ drop drop drop 0 ] [ [ [ drop ] dip swap lt? ] [ drop drop drop 0 ] [ drop drop drop 1 ] ifte ] ifte ] swap cons swap cons map uncons [ mul ] fold`, against a 92-op catalogue witness; 0 failures in 10,000 sampler, 9,083 perturbed and 9,088 dense draws. Ablated seed 0 epochs 5 and 6 (one task each), random seed 0 epoch 4 (4 of the behaviour's 8 tasks) and 6 (2). It shares its first 42 ops (through `map uncons`) with the two `bounds_cover_list_flag` shortcuts #134 recorded (44 and 46 ops) and differs in the reduction. No #134 cell found it.

**Shortcuts:** 45 distinct solved programs over the six run dirs, 35 genuine, 10 shortcuts, 0 inconclusive; new here `aligned_matches` 26 and 28 ops and `drop_last` 29 ops.

## Macro compression of v4 and world-v3 witnesses under the authoring library (#144)

Static: the 133 macros of `experiments/world_behaviours/macros.txt` expanded to `mark-28` primitives; compression is the fewest tokens over ops ∪ macros expanding to each distinct witness (dynamic programming, exact); branching at 8 prefix states per witness, an action legal iff every shown example accepts it under plain `step_raw` (no #70 guard); cost estimate is highways (#96) step 3's `d · log2 b_eff`, mean over witnesses, in bits. Subset: the 35 macros that hit a v4 train witness, selected on train. [report](../reports/macro-compression-20260914/README.md), `tables.md`, `compression.json`.

| corpus | witnesses | median ops | full library: median tokens, ops saved, witnesses hit, macros used, b_eff, bits raw → library | 35-macro subset: median tokens, ops saved, b_eff, bits raw → library |
|---|---:|---:|---|---|
| v4 train | 175 | 44 | 31, 0.314, 0.891, 35 / 133, 24.8 → 137.1, 224.9 → 236.2 | 31, 0.314, 24.7 → 54.0, 224.4 → 191.5 |
| v4 eval | 45 | 40 | 30, 0.343, 0.778, 26 / 133, 24.4 → 133.2, 228.8 → 230.1 | 30, 0.343, 24.9 → 54.4, 230.1 → 188.1 |
| v4 cohort (#101) | 43 | 39 | 30, 0.344, 0.767, 25 / 133, 24.7 → 135.7, 228.0 → 229.0 | 30, 0.344, 24.5 → 53.3, 227.3 → 185.4 |
| world-v3 train | 482 | 145 | 21, 0.825, 0.977, 126 / 133, 24.7 → 137.5, 1181.1 → 317.4 | 83.5, 0.452, 24.6 → 53.8, 1180.1 → 805.0 |
| world-v3 eval | 44 | 149.5 | 28, 0.813, 1.000, 80 / 133, 25.5 → 142.3, 1050.5 → 300.2 | 90, 0.411, 25.1 → 55.1, 1045.1 → 766.1 |

Cohort by witness-length bucket, bits raw → full library → subset: 1–5 (6 witnesses) 22.4 → 28.3 → 22.9; 6–10 (3) 29.3 → 40.1 → 32.5; 11–20 (6) 71.7 → 85.0 → 68.8; 21–40 (9) 154.7 → 174.7 → 141.5; 41+ (19) 408.3 → 393.4 → 318.4. Macros hitting v4, by ops saved on train: `pack3` 300, `abs` 276, `max2` 253, `unpack3` 252, `pack2` 152, `reverse` 140, `second` 108, `unpack2` 104, `take` 98, `drop-n` 98; no semantic macro (`range`, `digits`, `gcd`, `sort`, `zip`) occurs in any v4 witness. On world-v3 train `range` alone saves 13,081 ops; the DP parse equals the authors' macro form at the median (21) and is never longer.

## Where a staged witness's deficit sits: recurring cycles against the glue between them (#148)

Staged arm of #155, seeds 0 / 1 / 2, epoch 3, replayed along the staged witness on 137 treated train behaviours x 4 tasks, CPU, no search. Cycle = adjacent top-level quotations immediately followed by a combinator; library = top 35 cycles by ops saved, chosen on the 123 fit behaviours; glue = every other top-level step. Deficit = `log p(top legal op) - log p(authored op)`. `tail` = a cycle's summed deficit less its largest step, `upper` = the whole sum. [report](../reports/cycle-deficit-20260920/README.md), [mine-v4-library (#148)](../../issues/148)

| | share of steps: library / other cycle / glue | share of deficit: library / other cycle / glue | nats per step: library / other cycle / glue | compression in steps | compression in deficit, tail / upper | behaviours inside ln 64: raw / minus tail / minus upper |
|---|---|---|---|---:|---|---|
| fit, 123 behaviours | 51.7% / 20.0% / 28.3% | 15.1% / 18.2% / 66.7% | 0.07 / 0.23 / 0.59 | 1.83x | 1.01x / 1.18x | 37 / 38 / 44 |
| held out, 14 behaviours | 57.2% / 21.6% / 21.2% | 17.2% / 22.0% / 60.7% | 0.18 / 0.62 / 1.75 | 1.98x | 1.02x / 1.21x | 1 / 1 / 2 |

Cycle recurrence behind the library (`experiments/cycle_recurrence.py`, the 143 staged witnesses): 595 cycles, 147 distinct, 65 in two or more behaviours covering 59.8% of staged ops; top-35 length compression 1.82x in sample, 1.72x held out 5-fold by behaviour (1.75x without the 6 contrived rows).

## Training on staged witnesses: four arms and the per-cycle diagnostic (#155)

The #101 1m policy, seeds 0 / 1 / 2, epoch 3 fixed in advance, 313,560 task-balanced rows per epoch in every arm, cohort 300 eval tasks / 43 behaviours, beam 64 x depth 48, guard `both`, CUDA. Arms differ only in the program each of 123 treated fit behaviours is replayed along. Verified = 10,000-draw `genuine` or equal to a catalogue witness. A task is abandoned after 10 s of wall clock (beam-state-size-unbounded (#157)): 0 / 15 / 2 tasks in the staged seeds, none elsewhere. [arms report](../reports/staged-arms-20260920/README.md), [cycles report](../reports/staged-cycles-20260920/README.md), [staged-witness-reauthoring (#155)](../../issues/155)

| arm | graded tasks /300 | verified tasks | verified behaviours /43 | union over seeds | by a program found nowhere in the arm's train text | only by a contained program |
|---|---|---|---|---:|---|---|
| fused | 86.0 (88 / 81 / 89) | 83.7 | 12.33 (13 / 11 / 13) | 15 | 8.33 | 4.00 |
| re-fused | 90.3 (86 / 90 / 95) | 87.0 | 12.67 (12 / 13 / 13) | 15 | 7.33 | 5.33 |
| staged | 139.7 (143 / 128 / 148) | 137.7 | 15.67 (16 / 16 / 15) | 16 | 4.33 | 11.33 |
| both | 130.3 (121 / 147 / 123) | 128.0 | 14.00 (14 / 15 / 13) | 15 | 1.00 | 13.00 |

Verified behaviours by bucket (fused / re-fused / staged / both): 1-5 6.00 / 4.67 / 6.00 / 6.00; 6-10 1.00 / 1.33 / 3.00 / 3.00; 11-20 3.67 / 4.00 / 5.00 / 4.67; 21-40 1.33 / 2.33 / 1.67 / 0.33; 41+ 0.33 / 0.33 / 0.00 / 0.00. Staged against the fused union: gains `keep_below`, `n_minus_each`, `largest_inside_interval`; loses `shared_value_occurrences`, `farther_choice_from_target`.

Per-cycle diagnostic, oracle sub-goal (next closed state as the search target), validation-skeleton holdout, 14 behaviours x 5 tasks, beam 64, depth = cycle length + 4, mean of 3 seeds:

| witness | transitions | 2+-op transitions solved | 13+ ops (3-4 distinct cycles) | whole witnesses walked | 2+-op cycles whose text is in the fit witnesses, solved | cycles whose text is new, solved |
|---|---:|---|---|---|---|---|
| fused | 500 | 0.445 | 0.308 | 0.152 | 45 of 110, 0.86 | 65 rows / 13 distinct, 0.16 |
| staged | 750 | 0.914 | 1.000 | 0.795 | 275 of 305, 0.99 | 30 rows / 6 distinct, 0.20 |

## Staged and re-fused witnesses for the v4 train behaviours (#155)

153 train behaviours of 12+ fused ops rewritten by Claude Opus 5 authors under three rules (no quotation over 8 atoms, no write run over 12 steps, at most 4 top-level non-write steps between a quotation's `]` and the combinator that runs it); every program and its re-fused twin verified against the Python reference and all 40 tasks' pairs. Staged = within the rules and (2+ more post-execution closed states, or the fused witness broke a rule). Side file `experiments/staged_witnesses/v4.jsonl`. [report](../reports/staged-full-20260920/README.md), [pilot](../reports/staged-pilot-20260920/README.md), [staged-witness-reauthoring (#155)](../../issues/155)

| verdict | n | | over the 143 staged | fused | staged | re-fused |
|---|---:|---|---|---:|---:|---:|
| staged | 143 | | total ops | 7,470 | 6,425 | 6,790 |
| not staged (4 sorting / selection, 4 that need the extremum of every prefix) | 8 | | ops written under a mark | 0.841 | 0.704 | 0.754 |
| fused already within rules | 2 | | closed states / ops | 0.144 | 0.275 | 0.225 |
| already short (< 12 ops), not attempted | 22 | | executions, median | 1 | 4 | 2 |
| | | | largest quotation, atoms, median | 29 | 7 | 16 |
| | | | longest write run, median | 36 | 11 | 21 |

Length ratio staged / fused, median: 12-20 ops 1.67 (24 of 26 staged), 21-40 1.05 (26 of 28), 41+ 0.74 (93 of 99); all staged 0.88, 86 of 143 shorter. 88% of fused training rows (one per witness op) get a new witness; 14 of the validation-skeleton holdout's 16 behaviours staged. Twin executes fewer times than staged for 136 of 143, 6 identical, median 1.04x the staged length. Flags: 6 contrived, 7 sampler preconditions, 1 override.

## Witness trajectory shape of v4 and world-v3 (#150)

Static: every distinct witness replayed on its first shown input, each step labelled W (under an open mark, or the `[` opening one), D (top-level op) or X (top-level combinator). Cells are p10 / p50 / p90 over witnesses (interpolated); a closed state has no open mark and no parked quotation. [report](../reports/witness-trajectory-shape-20260919/README.md), [witness-trajectory-shape (#150)](../../issues/150)

| corpus | witnesses | W share of all ops | W share | executions X | exactly one X | largest body / length | fuel inside combinators | input prefix intact until | closed states | peak size / max(in, out) | X in last decile |
|---|---:|---:|---|---|---:|---|---|---|---|---|---:|
| v4 | 220 | 0.844 | 0.60 / 0.83 / 0.93 | 1 / 1 / 3 (mean 1.57) | 127 | 0.21 / 0.60 / 0.71 | 0.25 / 0.57 / 0.76 | 0.00 / 0.71 / 0.95 | 1 / 5 / 13 | 1.75 / 5.14 / 14.28 | 0.60 |
| v4, 41+ ops | 120 | | 0.78 / 0.84 / 0.93 | 1 / 1 / 3 | 63 | 0.37 / 0.65 / 0.72 | 0.33 / 0.62 / 0.78 | 0.00 / 0.53 / 0.95 | 1 / 11 / 16 | 4.49 / 7.64 / 17.05 | |
| world-v3 | 526 | 0.885 | 0.40 / 0.83 / 0.99 | 1 / 2 / 8 (mean 3.63) | 155 | 0.07 / 0.42 / 0.87 | 0.09 / 0.75 / 0.91 | 0.00 / 0.51 / 1.00 | 1 / 14 / 66 | 3.00 / 16.54 / 175.92 | 0.36 |
## Solved-program cache, search width, and the `ln W` budget on the v4 cohort (#151, #99)

The #101 300-task `stack_pbe_v4` cohort (43 distinct behaviours), depth 48, guard `both`, checkpoint `witness-s0-e4` of the 1m v4-only arm, one host (`UltraDesk`, 8 CPU threads). The cells span commits `dd2a3bbc`, `8cd3c165` and `e21c967b`; `wandering_light/` is byte-identical across all three, so the solver is one solver. **timed** is the sum of the per-task timers, the quantity `docs/metrics.md` defines; `cell_seconds` is never quoted here. Every cell ran with a faster `beam_solve` — structural `==` in place of `value_key` for the hit test and the visited set, a reason-free `step_all_quiet` (together **1.43x** on the cohort clock, `0635935`), and the cyclic collector disabled for the search (`910cfa6`, unquantified) — so none compares with a recorded pre-speedup wall-clock. Equivalence is pinned by `tests/test_stack_beam_equivalence.py`, which runs the pre-speedup loop against the shipped one.

**Behaviours are verified**, meaning the union of draw-tested `genuine` verdicts and `equals_catalogue_witness` ones, which is the convention `step3_verify_solves.partition` documents. Of the 27 distinct solved programs, **18 were draw-tested** (10,000 draws in each of the sampler / perturbed / dense modes, 500-tested floor) and **9 equal a catalogue witness and were not drawn against at all** — those inherit the corpus's own witness verification, a weaker standard that #142 counts apart. The graded column is kept beside it.

| cell | tasks /300 | graded behaviours /43 | verified /43 | timed (s) | executions | cache hits (of which solved) |
|---|---:|---:|---:|---:|---:|---:|
| cache off, beam 64 (published protocol) | 94 | 12 | 12 | 195 | 71,700,695 | — |
| cache on, beam 64 | 117 | 13 | 12 | 162 | 58,736,228 | 109 (105) |
| cache off, beam 128 | 106 | 13 | 12 | 376 | 134,396,899 | — |
| cache on, beam 128 | 121 | 14 | 12 | 321 | 113,665,637 | 113 (108) |
| cache off, beam 256 | 118 | 16 | 14 | 750 | 251,743,031 | — |
| cache on, beam 256 | 134 | 16 | 14 | 662 | 209,416,601 | 123 (118) |
| cache off, beam 512 | 126 | 17 | 16 | 1369 | 471,107,580 | — |
| cache on, beam 512 | 160 | 18 | 16 | 965 | 334,439,543 | 150 (143) |

By witness-length bucket (1-5 / 6-10 / 11-20 / 21-40 / 41+, each of 60), cache **on**: beam 64 gives 59, 21, 25, 11, 1 and beam 512 gives 60, 38, 43, 12, 7. Cache **off**: beam 64 gives 58, 13, 13, 10, 0 and beam 512 gives 60, 24, 28, 10, 4. No bucket loses tasks to a wider beam in either arm. **Width alone** is 13 -> 24 in the 6-10 bucket and 0 -> 4 in 41+; the larger 13 -> 38 and 0 -> 7 deltas are width *and* cache together, and must not be quoted as a width effect. Behaviours added by width over the beam-64 baseline: `bounds_cover_list_flag`, `farther_choice_from_target`, `keep_below`, `sum_negative`; none lost. Verification moved two in opposite directions: `aligned_matches` is a confirmed shortcut (5 ops, [#141](../../issues/141)), `bounds_cover_list_flag` a genuine 46-op program distinct from the 46-op shortcut it also finds.

Cache over 100 task orders, simulated exactly (the cache never changes what a search returns, only whether a task is searched): 115.9 ± 3.0 tasks [min–max 107–122], 0.04 regressions per order, and shortest-first / longest-first / insertion-order agree within noise. The `disjoint` diagnostic (replay only programs found on a *different* behaviour; needs behaviour labels, so never a solver) adds 0 tasks over cache-off: cross-behaviour transfer is exactly zero.

Floored path score (`max(log p, -tau)` in the ranking key only): tau inf / 3.0 / 1.5 / 0.7 → 94 / 47 / 39 / 28 tasks. `tau=inf` reproduces the baseline's 94 tasks and 71,700,695 executions exactly.

`ln W` budget law, `L x deficit < ln W` over the 43 behaviours: 83.7% accuracy, 5 true positives, 0 false positives, 7 false negatives, 31 true negatives. Shortening the 31 unsolved behaviours need to fit beam 64: min 1.9x, median 6.7x, max 103x.

Self-shortening: harvest over 1140 stratified train searches returned 936 programs, 80 shorter than the authored witness, **42 behaviour-equivalent** on every shown input of every train task of that behaviour; 8 of 159 train behaviours relabelled, 312,880 rows against the control's 313,560. Per-seed **verified** behaviour deltas against the matched `witness` control (all 18 cells draw-tested): epoch 3 (0, −1, −2), epoch 4 (−1, +2, +1), epoch 5 (0, 0, +1) — **summing to 0 over the nine cells**. On raw graded counts the same nine sum to +6, and on graded-minus-leaky-names to +2; the arm is a null only once shortcuts are excluded, which is why the verified column is the one quoted.

## The pinned best system on the v4 cohort: width, the library and the state cap (#158, #157, #99)

`solve.BEST` = `staged-s0-e3` (the staged arm of staged-witness-reauthoring (#155), seed 0, epoch 3; Hub `abhishekraok/wandering-light-stack-policy` revision `e5a0b1391c5268c3ae04a49e6862f19a30779bb2`), run by `uv run python -m wandering_light.solve` through `experiments/best_system_cells.py`. The #101 300-task / 43-behaviour cohort in cohort order, one host (`UltraDesk`, CPU, 8 threads), **one process per cell** with `run_body`'s memo cleared per cohort, guard `both`, depth 48, state cap 16,384 atoms. Each cell is `BEST` with **one** setting changed. `timed` is the sum of the per-task clocks, which charge library replay, search and policy inference; grading runs after the last clock stops. **Verified** is the union of 10,000-draw `genuine` verdicts and catalogue-witness-equal programs. **search-solved** is how many of the solved tasks were solved by a search rather than replayed from the library, and is the number of independent attempts the verified column actually rests on. [report](../reports/best-system-20260922/README.md), `cells.json`, `verified.json`

| cell | beam | library | tasks /300 | search-solved | graded behaviours /43 | **verified /43** | timed (s) | executions |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| b64 (BEST) | 64 | on | 180 | 19 | 18 | **16** | 154 | 34,785,672 |
| b128 | 128 | on | 181 | 17 | 18 | **16** | 330 | 65,233,317 |
| b256 | 256 | on | 183 | 19 | 20 | **16** | 706 | 120,034,579 |
| b512 | 512 | on | 190 | 20 | 22 | **17** | 1,397 | 214,205,105 |
| nocap-b64 | 64 | on | 180 | 19 | 18 | **16** | 153 | 34,785,672 |
| nolib-b64 | 64 | off | 143 | 143 | 18 | **16** | 209 | 49,078,770 |
| nolib-b128 | 128 | off | 158 | 158 | 17 | **16** | 399 | 82,869,294 |
| nolib-b256 | 256 | off | 168 | 168 | 19 | **18** | 823 | 147,323,518 |
| nolib-b512 | 512 | off | 177 | 177 | 21 | **17** | 1,661 | 268,820,179 |

- **The library wins the objective, and the searches it skips cost verified behaviours.** At beam 64 it is 180 tasks against 143 in 154 s against 209 s, with 29% fewer executions, so it improves the `docs/metrics.md` pair on both terms; `nolib-b64`'s 143 tasks match the task count of #155's own cell for this checkpoint (the task sets were not compared). At beam 256 it takes verified behaviours **18 down to 16**, and the cause is search count rather than the library: `b256` searched 131 of 300 tasks, and scoring `nolib-b256` on only those 131 gives 16 and the identical set, while on every task both arms searched they return the same program at the same depth. Both losses turn on cohort order: `all_positive`'s 22-op shortcut is found at cohort position 134 and replayed to 154, whose own search finds the 23-op genuine program, and `prefix_sums`'s 16-op shortcut at 232 is replayed to 233, whose own search finds an 18-op genuine program. A replay can only duplicate a program some search already found, so it can never add a verified behaviour and can remove one; in verified **tasks** the library wins at every width, 176 / 178 / 171 / 178 against 139 / 155 / 165 / 168.
- **Width replaces genuine programs with shortcuts.** Library-off, `all_positive`'s 9 cohort tasks are solved by its **genuine** program on 6 / 3 / 1 / 0 of them across beam 64 / 128 / 256 / 512 while the 22-op shortcut appears on 0 / 0 / 1 / 2, and memorised rows (shown-fit, held-out-fail) rise 17 / 30 / 39 / 43 library-off and 12 / 20 / 28 / 33 library-on. The beam takes the first fitting successor it meets while expanding a layer (`stack_beam.py:334`), before the layer is scored at all (`:345`), so the order is depth, then parent beam position, then op index; widening is not a superset operation, because the visited set is shared across depths (`:364`) and `nondecreasing_indicator` is *lost* at beam 128 in both arms. Verified behaviours are therefore not monotone -- library-off 16 / 16 / **18** / 17, library-on 16 / 16 / 16 / 17 -- but read that as an existence proof: verified **tasks** over the same cells are monotone (139 / 155 / 165 / 168), and beam 256's 18 rests on three behaviours verified by one task each.
- **Width buys graded behaviours, not verified ones.** Beam 64 to 512 is 8x the width and 9.1x the clock for +1 verified behaviour in each arm, while graded goes 18 to 22 and tasks 180 to 190. Width adds four graded behaviours library-on and loses none; **two of the four are shortcuts** (`aligned_matches`, `stable_distinct_shared_values`) and the other two (`pair_floor_averages`, `prefix_sums`) are the +1 net verified. Library-off it adds three, one a shortcut.
- **The #157 state cap is free.** All 300 rows identical with and without it on program, executions, fuel, expansions, depth and source; the clock moves 153.4 s to 154.2 s (0.46%). The cap is 132x the largest state along any v4 witness (124 atoms) and 6x the largest along any world-v3 witness (2,642). It is not a speed fix: `cartesian_pair_sums` is the slowest task in every cell and still costs 33 s at beam 512.
- **Cost to clear is a cost of failure.** 96% of the clock goes to tasks never solved (0.967 at b64, 0.961 at b512; 86.7-89.2% library-off), and the median library-on task costs 1 ms against 596 ms library-off at the same width.

Caveats: one checkpoint, one policy seed, one host, one cohort and one cohort order. `beam_solve` is deterministic given a checkpoint and a task, so a repeat run reproduces a cell exactly and these cells carry no estimate of the variance that governs a 2-behaviour swing. Verified counts are behaviours of the cohort's 43, not of v4's 220. Nothing here compares staged against fused under matched conditions.

## Epoch selection in verified behaviours, 1m v4-only arm (#154)

18 (seed, epoch) cells re-running the #101 1m v4-only arm (reproduces #134 bitwise: seed 0 epoch 4 verifies 94/94 tasks), all 30 distinct solved programs draw-tested. Cells are graded/verified behaviours of 43.

| epoch | s0 | s1 | s2 | mean verified |
|---|---|---|---|---:|
| 1 | 6/6 | 7/7 | 6/5 | 6.00 |
| 2 | 8/8 | 7/7 | 7/6 | 7.00 |
| 3 | 12/11 | 11/11 | 11/11 | 11.00 |
| 4 | 12/12 | 10/10 | 11/10 | 10.67 |
| 5 | 12/11 | 11/10 | 12/10 | 10.33 |
| 6 | 13/11 | 10/10 | 12/11 | 10.67 |

Selection rules, mean verified behaviours over three seeds: validation CE 6.00, fixed epoch 5 10.33, best fixed epoch (3) 11.00, oracle 11.33. Validation CE per epoch, seed 0: 0.952 / 0.956 / 1.094 / 1.210 / 1.362 / 1.509.

## Hindsight sub-target relabelling on v4 (#152)

Three arms of the #101 1m recipe (LR 1.25e-4, batch 256, 6 epochs, seeds 0/1/2, CUDA) with identical row counts (313,560 fit / 26,040 validation), identical labels, optimiser and seeds; only each row's target differs. Validation is the control arm's final-target rows in all three. Behaviours of 43 / tasks of 300, mean over seeds 0/1/2, cache off, **graded** (these cells were not draw-tested, so the absolute counts are upper bounds; the arms share a corpus and budget, so the comparison between them stands). The final-output shares are the realised draw at `ROW_SEED = 20260919` over the 6,360 fit tasks, not an expectation.

| arm | rows whose target is the final output | e3 | e4 | best validation CE (s0) | train CE at e6 (s0) |
|---|---:|---:|---:|---:|---:|
| `witness` (control) | 100% | 11.3 / 80.7 | 11.0 / 85.7 | 0.952 | 0.149 |
| `hindsight` (mark-free states ahead) | 20.6% | 9.3 / 60.3 | 10.3 / 77.7 | 0.984 | 0.187 |
| `hindsight-all` (every state ahead) | 8.3% | 9.0 / 61.3 | 9.3 / 72.0 | 1.008 | 0.233 |

`hindsight` uses the loose "no open mark" rule, so its draw includes parked quotations; under #150's tightened rule (no open mark **and** no parked quotation) v4 has a median 5 closed states per witness, 14.0% of ops. Witness-shape covariates over the 43 cohort behaviours, from #150's `shape.json`: Spearman(length, longest uninterrupted write run) **+0.954**, (length, closed states) +0.712, (length, executions) +0.645, (length, largest body share) +0.397. Restricted to 11–40 ops (n=15, 5 solved at beam 64), every staging covariate is null against solves: longest write run rho −0.197 (permutation p=0.49), closed states −0.156 (p=0.62), largest body share −0.033 (p=0.95), executions +0.000 (p=1.00). That band holds most of the variation but not all of it — 1–5 ops is 6/6 solved, 6–10 is 1/3, 41+ is 0/19 — and n=15 with 5 positives cannot support a verdict either way.

## Fused-policy transfer to staged witness text (#155, #99)

`witness-s0-e4` (1m v4-only) scored along the 18 staged rewrites of the #155 pilot and along their fused twins, same v4 train task, same inputs and final target. **Out of distribution by construction**: the policy trained only on write-then-run text, so this bounds transfer and does not estimate the deficit a staged-trained policy would show. Tolerance `ln 64` = 4.159 nats over the whole witness. [report](../reports/staged-transfer-20260920/README.md), driver `experiments/staged_transfer.py`

| arm | n | median length | median per-step deficit | median `L x d` | within `ln W` |
|---|---:|---:|---:|---:|---:|
| fused, train split (policy trained on these) | 40 | 44.0 | 0.075 | 3.2 | 23/40 |
| the 18 pilot fused witnesses (also train) | 18 | 43.5 | 0.086 | 3.5 | 11/18 |
| fused, eval split — unseen content, familiar style | 40 | 44.0 | 0.535 | 24.6 | 4/40 |
| **the 18 staged rewrites** — unseen content and style | 18 | 39.5 | **3.392** | **162.1** | **0/18** |

Staged is slightly *shorter* at the median, so the 39x deficit is not a length effect. Unseen content in the familiar style costs about 7x; unseen style about 45x.

Per-step deficit by step kind, fused → staged: `body` (under an open mark) 0.107 → **5.302** (400 → 298 steps), `open` 0.091 → 0.966 (133 → 131), `close` 0.025 → 2.557 (133 → 131), `run` 0.106 → 2.617 (28 → 92), `direct` 0.356 → 2.764 (125 → 151). The registered diagnostic — cheap bodies plus expensive boundaries would be weak evidence that short bodies are cheaper steps — **returns a clean negative**: body tokens are the most expensive staged kind, and one alone exceeds the whole budget. `open` stays cheap, so the policy will start a quotation anywhere; what it cannot predict is a short body's contents.

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
