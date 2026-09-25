# Results

**The headline numbers.** Every row is copied from an artifact under
`reports/`; the full ledger, every run and every benchmark, is
[`RESULTS_DETAILED.md`](RESULTS_DETAILED.md). `STATUS.md` says what these
mean; issues say why.

## Certified coverage — M3, synthetic stack pairs, one seed

Certified coverage, as registered, did not move over three cycles. Walker success rose in every branch over the frozen start policy: adaptive +0.039, control +0.061 and solved +0.062, as paired panel differences on one seed. The readout measures how well the refit P head is calibrated, not capability. A same-seed replicate moves it by 0.05.
[STATUS row](STATUS.md) · [novelty-reduction (#171)](../../issues/171) · [numbers](RESULTS_DETAILED.md#certified-coverage-m3-an-adaptive-attempt-selector-against-a-u-ranked-control-171)

## Stack PBE v4 — 220 behaviours, eval split 1,800 tasks

Held-out grading, one sample per task. "Sample 40" is the seed-17 eval sample;
it is the easy end of the split (median witness 26 ops against 40).
[report](../reports/stack-pbe-v4-baseline-20260901/README.md)

| solver | config | tasks | solved | wall (s) |
|---|---|---:|---:|---:|
| `gpt-6-astra` (via `codex exec`) | effort low | full 1,800 | **995 (55.3%)** | 33,487 (6 shards) |
| `gpt-6-astra` (via `codex exec`) | effort low / medium / high | sample 40 | **24 / 24 / 23 (60.0% / 60.0% / 57.5%)** | 606 / 661 / 840 |
| `claude-opus-5` (via `claude -p`) | effort low / high / xhigh | sample 40 | **17 / 20 / 25 (42.5% / 50.0% / 62.5%)** | 1,424 / 2,220 / 3,568 |
| `gpt-5.6-luna` | effort high | sample 40 | **15 (37.5%)** | 2,286 |
| `gpt-5.6-terra` (via `codex exec`) | effort high | sample 40 | 16 (40.0%) | 1,344 |
| BFS | 100,000 nodes | sample 40 | 6 (15.0%) | 3,917 |
| `qwen3:30b-a3b` (local, 4090) | 300 s deadline | sample 40 | 6 (15.0%) | 3,381 |
| BFS | 10,000 nodes | full 1,800 | 120 (6.7%) | 22,803 |
| beam 64 × depth 48, 1m next-op policy trained on v4 train | seeds 0 / 1, CPU | sample 40 | 7 (17.5%) / **10 (25.0%)** | 131 / 126 |

One solve per frontier arm and one BFS solve are the `aligned_matches`
sampler shortcut, so 24 (Opus xhigh), 23 (astra), 15 (terra), 14 (luna) and 5 (BFS) compute
the behaviour. BFS solves nothing at 6+ ops; qwen3 nothing at 16+. On the sample every
frontier arm solves 0 or 1 of 12 tasks past 40 ops, but on the full split
astra solves 214 of 840 past 40 ops (`median_floor`, 253-op witness, 17/40):
the sample's wall was the sample's. Astra memorises 714 of 1,800 (shown-only
94.9%), concentrated on flag-valued behaviours. Astra fits
the shown examples on 38–40 of 40 tasks and 15–16 of those are memorisations,
so shown-only grading would say 100%; its effort setting moved median
reasoning only 256 → 396 tokens per task, which is why the curve is flat.
Luna re-run through `codex exec` scores 12/40 against the API row's 15,
agreeing on 33 of 40 tasks (one-sample noise).
[codex-exec report](../reports/stack-pbe-v4-codex-exec-20260910/README.md)
Claude Opus 5 does spend its effort knob (median thinking 526 → 1,400 →
1,962 tokens per task, high above low on 38 of 40 paired tasks) and the curve
rises 17 → 20 → 25; every gain is a 12–49-op behaviour, the twelve tasks past
40 ops stay at 0–1 solved, and at xhigh Opus solves the same tasks as astra
(agree 38/40 with astra high, none astra-only) at 12.5× the output tokens.
[claude-exec report](../reports/stack-pbe-v4-claude-exec-20260913/README.md)

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
and rises after. Eleven of the cohort's 43 behaviours are ever solved at these
checkpoints; the longest verified solve is 10m seed 1's 22-op `capped_sum`, which
world-plus-v4-policy (#134) later verified, and the 44-op and 28-op solves the report
quotes are held-out-surviving shortcuts. Every arm here was selected at epoch 1 or 2;
the 1m arm retrained under the same recipe and searched at epochs 3–6 verifies 31-
and 40-op catalogue witnesses and up to 94 of 300 tasks, so at least the 1m rates
understate what its training reaches (`docs/STATUS.md`, Superseded).

## Stack PBE v4 — adding world-authored training data (300-task eval cohort)

The 1m policy above, trained on v4 train alone and with `stack_pbe_world_v3` train
added (world behaviours whose program or skeleton equals a v4 eval witness dropped,
model selection on the same v4 holdout), three seeds, searched at every epoch 1–6.
Verified solves of 300, mean of three seeds.
[report](../reports/world-plus-v4-policy-20260912/README.md),
[world-plus-v4-policy (#134)](../../issues/134)

| | v4 only | v4 + world | v4 + world, 156 `keep_below` carriers dropped |
|---|---:|---:|---:|
| holdout CE at the best-validation epoch | 0.954 | 0.858 | 0.871 |
| verified solves at the best-validation epoch (v4 only 1–2, v4 + world 2–4, dropped 2) | 52.3 | 86.0 | 62.0 |
| verified solves, mean over epochs 1–6 | 72.3 | 80.1 | 71.6 |
| verified solves at each seed's best epoch (chosen on the cohort) | 87.0 | 105.0 | 87.7 |

The search difference is one behaviour, `keep_below`, whose 6-op witness occurs
inside 156 world training programs and no v4 training program: summed over seeds and
epochs 1–6 it gains 186 solves against the cohort's net 141. 29 of the 43 cohort
behaviours are never solved by either arm. Dropping the 156 carrier programs
([keep-below-ablation (#136)](../../issues/136), third column) removes the search
gain (`keep_below` 7 solves over 18 cells, the other 42 behaviours level with v4 only)
and keeps 86% of the loss gain; dropping 156 world programs at random (one seed)
keeps 35%.

## `deep-corpus-v1` — learned solver against search (96 held-out tasks, 12 per distance 1–8)

[report](../reports/materialized-rollout-20260829/README.md)

| solver | solved /96 | distance 5–8 /48 |
|---|---:|---:|
| aligned-pair rollout policy, 90,374 params, beam 64 | **60** | **12** (7/12 at d5, 5/12 at d6, 0 at d7–8) |
| BFS at the policy's mean d5–8 spend (6,238 executions) | — | 1 |
| `opt-125m` SFT, k=64 | 14 | 0 |
