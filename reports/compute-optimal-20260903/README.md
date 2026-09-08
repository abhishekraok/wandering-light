# Compute-optimal policy size on `deep_corpus_v2` (three arms, two seeds)

Issue: compute-optimal-policy-size (#80). Source `e8989d2` (`claude/issue-80-three-arm`). Protocol `sha256:8d23d97a81da4eba…`, aggregate `sha256:cd392bd1d8cdc456…`, campaign completed 2026-09-03 03:58 on UltraDesk (RTX 4090, 8 pinned cores).

Question: which rollout-policy size is compute-optimal, measured as normalized AUC of solve rate against log2 budget (wall-clock seconds, or basis-function executions) on 96 validation tasks, 12 per certified distance 1–8, beam 64, depth 8, execution budget 100k. Two training seeds per arm, paired.

## Result

Per execution, the bigger policy searches better at every distance. Whether that buys wall-clock depends on the device's inference cost: on the 4090 the 10M arm wins from distance 4 up; on CPU its inference is ~5× slower and the 90K arm wins at every distance except 6 and 7.

### Wall-clock AUC (normalized), mean of two seeds

| distance | 90k cpu | 1m cpu | 10m cpu | 90k cuda | 1m cuda | 10m cuda |
|---|---|---|---|---|---|---|
| 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 | 1.000 | 0.990 | 0.878 | 0.984 | 0.987 | 0.989 |
| 3 | 0.873 | 0.788 | 0.607 | 0.865 | 0.857 | 0.843 |
| 4 | 0.585 | 0.549 | 0.481 | 0.582 | 0.589 | 0.664 |
| 5 | 0.536 | 0.477 | 0.401 | 0.534 | 0.501 | 0.554 |
| 6 | 0.272 | 0.336 | 0.271 | 0.274 | 0.372 | 0.398 |
| 7 | 0.071 | 0.110 | 0.138 | 0.074 | 0.125 | 0.215 |
| 8 | 0.065 | 0.053 | 0.056 | 0.068 | 0.066 | 0.093 |
| **mean** | **0.550** | **0.538** | **0.479** | **0.548** | **0.562** | **0.594** |

### Paired differences, 10m minus 90k, wall-clock AUC (mean over seeds, seed SD)

| distance | cpu | cuda |
|---|---|---|
| 1 | -0.000 (0.000) | -0.000 (0.000) |
| 2 | -0.122 (0.004) | +0.004 (0.002) |
| 3 | -0.266 (0.003) | -0.021 (0.002) |
| 4 | -0.104 (0.046) | +0.082 (0.033) |
| 5 | -0.135 (0.108) | +0.019 (0.106) |
| 6 | -0.001 (0.017) | +0.124 (0.018) |
| 7 | +0.066 (0.007) | +0.141 (0.001) |
| 8 | -0.008 (0.004) | +0.025 (0.029) |

### Execution AUC (device-free) and solve rate at max budget, mean of two seeds

| distance | 90k exec AUC | 1m | 10m | 90k solve | 1m | 10m |
|---|---|---|---|---|---|---|
| 1 | 0.992 | 0.997 | 0.997 | 1.00 | 1.00 | 1.00 |
| 2 | 0.799 | 0.801 | 0.801 | 1.00 | 1.00 | 1.00 |
| 3 | 0.593 | 0.594 | 0.593 | 1.00 | 1.00 | 1.00 |
| 4 | 0.313 | 0.328 | 0.357 | 0.79 | 0.83 | 0.92 |
| 5 | 0.257 | 0.246 | 0.271 | 0.79 | 0.75 | 0.83 |
| 6 | 0.133 | 0.171 | 0.193 | 0.46 | 0.58 | 0.67 |
| 7 | 0.033 | 0.056 | 0.101 | 0.12 | 0.21 | 0.38 |
| 8 | 0.031 | 0.031 | 0.042 | 0.12 | 0.12 | 0.17 |
| **mean** | **0.394** | **0.403** | **0.420** | | | |

### In-situ H at the root state (bits per step, masked optimal-set, state-weighted), headline only where certified coverage ≥ 0.8

| certified distance | 90k | 1m | 10m |
|---|---|---|---|
| 1 | 0.55 ± 0.03 | 0.35 ± 0.18 | 0.16 ± 0.01 |
| 2 | 1.20 ± 0.08 | 1.15 ± 0.24 | 1.00 ± 0.05 |
| 3 | 1.28 ± 0.03 | 1.08 ± 0.24 | 0.99 ± 0.28 |
| 4 | 2.40 ± 0.38 | 2.07 ± 0.50 | 2.57 ± 0.03 |
| 5 | — (coverage 0.67) | — (coverage 0.67) | — (coverage 0.67) |
| 6 | — (coverage 0.00) | — (coverage 0.00) | — (coverage 0.00) |
| 7 | — (coverage 0.00) | — (coverage 0.00) | — (coverage 0.00) |
| 8 | — (coverage 0.00) | — (coverage 0.00) | — (coverage 0.00) |

Coverage beyond the root collapses (0.4–0.6 at rollout depth 2, ≤0.3 deeper, ≤0.1 for certified distance ≥ 5): most beam-visited states are off every shortest path and the bounded oracle (depth 6, 20k states) cannot reach the target from them. Certified-only means for those groups are in `aggregate.json.gz` and are not comparable across arms. The durable fix is persistent-certified-distance-store (#94); every sampled visit with its masked scores is saved under the run root's `annotation/` directory for retroactive labelling.

## Training

| arm | params | LR | seed 0 best val CE (epoch) | seed 1 | source |
|---|---|---|---|---|---|
| 90k | 90,374 | 2e-3 | 1.148 (11) | 1.138 (12) | `eb3ca27`, Hub `…-issue-80-90k-s{0,1}` |
| 1m | 724,022 | 2e-3 | 1.007 (12) | 1.005 (12) | `eb3ca27`, Hub `…-issue-80-1m-s{0,1}` |
| 10m | 6,504,310 | 5e-4 | 0.937 (7) | 0.932 (6) | `db90b06`, Hub `…-issue-80-10m-lr5e-4-s{0,1}` |

Plain Adam, constant LR, 12 epochs, best-validation weights restored, 1,073,797 discovery prefixes. LR per arm from a one-epoch probe (see #80): the fixed 2e-3 recipe left 10m at 1.138, level with 90k. 40M was dropped: 10m already overfits after epoch 7 on this corpus, and a 4× model at 4× inference cost cannot win on the wall-clock axis.

## Caveats

- 12 tasks per distance; single-cell seed spreads reach 0.16. Quote the paired differences with SDs.
- The 100k execution budget never binds; beam 64 × depth 8 terminates first (≈6.8k executions at distance 8). The budget axis is effectively the beam schedule.
- CPU/CUDA parity: success, termination, and solutions match on all 96×6 tasks; 9 tasks differ by 1–5 executions (float tie-breaks late in the search).
- The 90k and 1m checkpoints were trained at `eb3ca27` and the 10m at `db90b06`; the protocol records those source mismatches rather than refusing them.

## Files

`protocol.json` (frozen protocol), `timing-matrix.json` (sealed), `aggregate.json.gz` (full aggregate), `oracle-labels.json.gz` (shared oracle cache: certified optimal-action sets by state/target), `campaign.log`. Raw per-cell timing rows and sampled visits live in the run root on UltraDesk: `wandering_light/.worktrees/issue-80-refreeze/checkpoints/compute-optimal-policy-size-20260902-per-arm-lr-two-seed/`.
