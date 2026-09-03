# Compute-optimal policy size on `mark-28` / `stack_pbe_v4`

Issue: [compute-optimal-mark-28 (#101)](../../../../issues/101). Source `4fd2eb9`
(`claude/issue-101-compute-optimal-mark28`), clean. Campaign 2026-09-03 08:44–15:01 PDT
on UltraDesk (RTX 4090; CPU cells at 8 torch threads, executor single-threaded).

Question: the compute-optimal-policy-size (#80) question on the stack machine. Which
rollout-policy size is compute-optimal, as normalized AUC of solve rate against log₂
budget — wall-clock seconds primary, op applications as the device-free diagnostic —
per **witness-length bucket**, paired over two training seeds, CPU and CUDA cells.
v4 has no certified distance; witness length is an upper bound.

## Result

**The size axis is flat, and the corpus is why.** Every arm solves the same
behaviours at the same rate; per op application the three are indistinguishable;
on wall-clock the smallest wins on CPU and ties on CUDA. This is the opposite of
`wl-core-v1`, where the 10M arm searched better per execution at every distance. The
difference is not the model: it is that v4's 313,560 training rows are **175
programs**, one per behaviour, on fresh inputs — so behaviour cloning peaks at epoch 1–2
and every arm is compared at the same under-trained point (validation CE 0.925–0.976).

### Solve rate at max budget, mean of two seeds (60 tasks per bucket; CPU and CUDA agree)

| witness length | 1m | 2m | 10m | BFS 10k nodes on the full split (#85) |
|---|---|---|---|---|
| 1–5 | 0.608 | 0.592 | 0.617 | 0.500 |
| 6–10 | 0.050 | 0.000 | 0.042 | 0.000 |
| 11–20 | 0.267 | 0.267 | 0.233 | 0.000 |
| 21–40 | 0.000 | 0.000 | 0.000 | 0.000 |
| 41+ | 0.000 | 0.017 | 0.017 | 0.000 |

### Normalized AUC, mean of two seeds (seed SD), paired difference against 1m

Wall-clock range [0.01, 100] s; op-application range [28, 100,000]; both fixed before any cell ran.

| bucket | device | 1m wall | 2m wall | 10m wall | 1m ops | 2m ops | 10m ops |
|---|---|---|---|---|---|---|---|
| 1–5 | cpu | 0.547 (0.053) | 0.521 Δ−0.026 | 0.507 Δ−0.040 | 0.356 (0.021) | 0.355 Δ−0.001 | 0.359 Δ+0.002 |
| 1–5 | cuda | 0.545 (0.054) | 0.532 Δ−0.013 | 0.545 Δ−0.000 | 0.356 | 0.355 | 0.359 |
| 11–20 | cpu | 0.157 (0.000) | 0.149 Δ−0.008 | 0.120 Δ−0.037 | 0.058 (0.000) | 0.057 Δ−0.001 | 0.049 Δ−0.010 |
| 11–20 | cuda | 0.157 (0.000) | 0.160 Δ+0.003 | 0.134 Δ−0.023 | 0.058 | 0.057 | 0.049 |
| 6–10 | both | 0.036 (0.036) | 0.000 | 0.025 | 0.017 | 0.000 | 0.012 |
| 21–40, 41+ | both | 0.000 | ≤ 0.006 | ≤ 0.007 | 0.000 | ≤ 0.001 | ≤ 0.001 |

Full tables with SDs for every cell: `tables.md`; per-seed values: `aggregate.json`.

### Cost to clear the 300-task cohort — the define-cost-to-clear (#95) pair, mean of two seeds

| device | | 1m | 2m | 10m |
|---|---|---|---|---|
| cpu | coverage | 0.185 | 0.175 | 0.182 |
| cpu | wall-clock over covered subset | 10 s | 18 s | 27 s |
| cpu | total spent | 1,009 s | 1,374 s | 2,198 s |
| cuda | coverage | 0.185 | 0.175 | 0.180 |
| cuda | wall-clock over covered subset | 11 s | 13 s | 15 s |
| cuda | total spent | 887 s | 940 s | 1,074 s |

Coverage does not move with size; the bill does. An unsolved task costs 4.0 / 5.5 / 8.9 s
on CPU and 3.0 / 3.1 / 3.6 s on CUDA for 1m / 2m / 10m: the executor dominates the 1m arm
on either device, and only the 10m arm's inference is large enough for the device to matter.

### Like-for-like against the v4 baselines

The cohort contains the seed-17 40-task sample every #85 arm was scored on.

| solver | solved / 40 |
|---|---|
| `gpt-5.6-luna`, effort high | 15 |
| **beam 64 × depth 48, 1m policy** | **7, 10** (seeds 0, 1) |
| **beam 64 × depth 48, 2m / 10m** | 7, 9 / 6, 8 |
| BFS, 100,000 nodes | 6 |
| `qwen3:30b-a3b`, 300 s | 6 |

The learned policy sits with a 100k-node search and the local model, at a third of
the search's node budget, and at half the frontier model.

## What the beam finds

Eleven of the cohort's 43 behaviours are ever solved. Five are solved by every arm and
seed at every task: `push_back` (12/12), `reverse` (9/9), `rotate_left` (10/10),
`min_pair` (7/7, a 12-op `[ over lt? ] [ swap ] [ ] ifte drop`) and most of `drop_last`
(12-op). Two are seed-dependent regardless of size: `keep_zeros` (0 or 8–10 of 11) and
`negate_each` (0–6 of 21). Everything at 21–40 ops and 17 of the 18 behaviours at 41+
are never touched.

The found program is usually **shorter than the authored witness**: 144 solves at depth 2
against 4-op witnesses, 120 at depth 3 against 5-op, 84 at depth 11 against 12-op. The
long shot is `bounds_cover_list_flag`, witness **92 ops**, solved by both 10m seeds and
2m seed 1 as a **44–46-op program** — a 64-beam carrying a program eight times past the
5-op wall BFS hits on this corpus. 10m also found a 28-op `all_positive` and a 22-op
`capped_sum`.

Memorisations (fit the shown examples, fail the held-out) run 5–15 per 300-task cell;
the shown-only number would overstate coverage by 2–5 points.

## Training

| arm | dims | params | LR | seed 0 best val CE (epoch) | seed 1 |
|---|---|---|---|---|---|
| 1m | (128, 320, 640) | 655,836 | 1.25e-4 | 0.952 (1) | 0.945 (2) |
| 2m | (224, 576, 1152) | 2,045,660 | 6.25e-5 | 0.939 (1) | 0.976 (2) |
| 10m | (384, 1024, 2048) | 6,288,412 | 3.1e-5 | 0.925 (2) | 0.934 (2) |

Plain Adam, constant LR, batch 256, best-validation weights restored; validation is
1 in 10 train skeletons by sha256 (26,040 rows); the eval split was touched only by
the timed cells. LR per arm from a one-epoch probe (`lr_probe.json`, W&B runs
tagged `lr-probe`): every arm's best was the bottom of the #80 grid, so the probe was
made to halve until an interior point wins (`913c69d`); after three halvings each arm
still sat at the edge, with the last halving worth 0.01 nats. The chosen LRs are
∝ 1/width and ~16× below `wl-core-v1`'s.

**Overfitting from epoch 2.** `1m` seed 0 over 12 epochs: train CE 1.09 → 0.10,
validation 0.95 → 1.99. The trainer restores epoch 1, so the checkpoint is sound, and
the remaining runs were cut to 3 epochs (Abhishek, mid-campaign; data order is seeded,
so early epochs are identical). The 12-epoch runs are 1m ×2 and 2m seed 0.

Arms: 90k dropped (44% of witness-replay states overflow its 96-byte item cap, so its
encoder rather than its size would bind); 40m stays dropped from #80.

## Caveats

- 60 tasks per bucket; seed spreads reach 0.075 on solve rate at 1–5. Quote the paired differences.
- Witness length is an upper bound, not a certified distance, and the beam's own solutions show the authored witnesses at 4, 5 and 12 ops are not shortest.
- Depth cap 48 and beam 64 bound the op-application axis at 82,796; the 100,000 range never binds, as on #80.
- CPU/CUDA parity: solved and program agree on 3,599 of 3,600 task-cells; 7 differ by a few op applications on float tie-breaks.
- `load.log` samples load average once a minute; the CPU cells' own 8 threads dominate it, so it shows the box was not otherwise idle-to-busy in a detectable way rather than proving isolation.

## Protocol and files

`experiments/compute_optimal_mark28.py campaign --run-dir …` (relaunched twice: `--epochs 3` from 2m seed 1 on). Search: `wandering_light/stack_beam.py`, beam 64, depth 48, `quotation_guard.blocked(…, "both")`, product state over shown examples, held-out grading by `stack_pbe_eval.grade`. Policy: `wandering_light/stack_policy.py`, one item per shown example, op/value tokens, state cap 160 keeping the top.

`aggregate.json`, `tables.md`, `lr_probe.json`, `timing/<arm>-s<seed>-<device>.json` (every task row with program, termination and costs), `policies/*/` (trainer state, manifest with provenance and W&B URL; weights stay on UltraDesk under `checkpoints/mark28-co-20260903/policies/`), `runner.log`, `load.log`.
