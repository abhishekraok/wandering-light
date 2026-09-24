# K diagnostic (short-segment-campaign #168)

`experiments/segment_k_diagnostic.py` runs the segment-reset beam with these settings:
- width 64;
- 16 roots;
- 12 segments;
- state cap 16,384;
- GPU policy.

The witness's closed state is forced back in as a root at every cut. The diagnostic records where the witness's next closed state ranks among every merged closed state. Each run uses about 40 programs per set, every k-th program.

- **`segment-k-20260924/`**: segment depth 12, registered rank only. The two capped `yardstick-167` runs crashed partway (a float32 serialisation bug, fixed in 6bd7036) and are partial.
- **`segment-k-20260924-v2/`**: depth per arm (capped 8, staged 16). It adds analysis ranks:
  - `local_rank`: score gained in this segment only;
  - `per_op_rank`: mean log-probability per op;
  - `fresh_rank`: never closed or expanded before.

  It also describes the top competitor: its segment length, its token edit distance from the root, and its program.

## Share of segments with the witness's state in the top 16

| policy | programs | depth | segments | not produced | cumulative (registered) | local | per op | fresh |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BEST (staged/current) | staged train | 16 | 179 | 0.23 | 0.12 | 0.28 | 0.08 | 0.11 |
| capped/difference s0 | capped train | 8 | 268 | 0.13 | 0.19 | 0.46 | 0.19 | 0.20 |
| BEST | yardstick-168 | 16 | 230 | 0.80 | 0.02 | 0.06 | 0.01 | 0.02 |
| capped/difference s0 | yardstick-168 | 8 | 230 | 0.26 | 0.03 | 0.27 | 0.03 | 0.03 |

The depth-12 runs, registered column only:

| policy | programs | top 16 |
|---|---|---:|
| capped/machine | capped train | 0.19 |
| capped/difference | capped train | 0.19 |
| fused/current | fused train | 0.13 |
| BEST | yardstick-167 | 0.02 |

On capped/difference's own training programs, the first segment reaches 0.62 in the top 16 under every ranking. Later segments reach 0.12 cumulative and 0.43 local.
