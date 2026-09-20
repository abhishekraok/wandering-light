# solved-program-cache (#151) and hindsight-subtargets (#152)

The #101 300-task `stack_pbe_v4` cohort, beam 64 x depth 48, guard `both`, one
host (`UltraDesk`, RTX 4090, 8 CPU threads), one commit. Every cell here was
timed with the faster `beam_solve`, so none of them compares with a recorded
pre-speedup wall-clock; the arms compare with each other. **timed** is the sum
of the per-task timers, which is the quantity `docs/metrics.md` defines and the
quantity the recorded cells aggregate; **elapsed** is the whole cell and is kept
beside it. Grading runs after the timers are sealed, because it warms the
executor's process-global `run_body` cache that later searches read.

The cohort is **43 distinct behaviours behind 300 tasks**, so tasks and
behaviours are reported side by side. `behaviours*` excludes
`all_positive` and `bounds_cover_list_flag`, whose samplers `docs/STATUS.md` records as leaking.

## The cache

| cell | tasks /300 | behaviours /43 | behaviours* | timed (s) | elapsed (s) | executions | served from cache |
|---|---:|---:|---:|---:|---:|---:|---:|
| `off-s0-e4` | 94 | 12 | 12 | 195 | 197 | 71,700,695 | 0 |
| `off-s0-e4-order1` (order 1) | 94 | 12 | 12 | 209 | 211 | 71,700,695 | 0 |
| `on-s0-e4` | 117 | 13 | 12 | 162 | 163 | 58,736,228 | 109 |
| `on-s0-e4-order1` (order 1) | 115 | 14 | 12 | 165 | 166 | 59,666,010 | 107 |
| `on-s0-e4-order2` (order 2) | 116 | 14 | 12 | 173 | 175 | 59,587,435 | 109 |
| `disjoint-s0-e4` — diagnostic, cross-behaviour reuse only | 94 | 12 | 12 | 202 | 204 | 71,912,694 | 0 |

Per witness-length bucket:

| cell | 1-5 | 6-10 | 11-20 | 21-40 | 41+ |
|---|---:|---:|---:|---:|---:|
| `off-s0-e4` | 58/60 | 13/60 | 13/60 | 10/60 | 0/60 |
| `on-s0-e4` | 59/60 | 21/60 | 25/60 | 11/60 | 1/60 |
| `disjoint-s0-e4` | 58/60 | 13/60 | 13/60 | 10/60 | 0/60 |

## Over 100 task orders, simulated exactly

The cache never changes what a search returns, only whether a task is searched,
so one cache-off cell plus a fits matrix determines every order. `cohort()` is
bucket-major and witness-length bucket is not visible to a solver that sees only
shown examples, so bucket-major is not an order the system could choose.

| | tasks /300 | behaviours /43 | searches | regressions |
|---|---:|---:|---:|---:|
| cache off | 94 | 12 | 300 | — |
| cache on, random order | 115.8 ± 3.0 [107-122] | 13.7 [12-14] | 192 | 0.0 (max 1) |
| cache on, bucket-major (privileged) | 117 | 13 | 191 | 0 |

104 of 108 cache hits survive held-out grading. The replay rule does not matter: insertion-order 115.8, longest-first 115.8, shortest-first 115.8.

## Training arms at a matched epoch (#152), cache off

Identical row counts, labels, optimiser and seeds; only the target each
training row carries differs.

Behaviours are reported beside tasks and are the more stable of the two:
within one arm the seed spread on tasks is several times the spread on
behaviours, so a task difference of under ~15 says nothing.

| arm | seeds | tasks /300, mean ± sd | behaviours /43, mean ± sd | task range |
|---|---:|---:|---:|---|
| `witness` | 4 | 87.8 ± 8.2 | 11.2 ± 0.8 | 74-94 |
| `hindsight` | 3 | 77.7 ± 10.3 | 10.3 ± 0.5 | 68-92 |
| `hindsight-all` | 3 | 72.0 ± 7.1 | 9.3 ± 0.5 | 62-78 |

## Arm x epoch (#152), cache off — behaviours of 43, then tasks of 300

A single-epoch comparison inherits the confound #134 was burned by: solves
track the epoch as much as the data. Each cell is the mean over the seeds run.

| arm | e1 | e2 | e3 | e4 | e5 | e6 |
|---|---:|---:|---:|---:|---:|---:|
| `witness` | **6.3** / 51 | **7.3** / 57 | **11.3** / 81 | **11.0** / 86 | **11.7** / 85 | **11.7** / 80 |
| `hindsight` | **6.0** / 50 | **7.3** / 56 | **9.3** / 60 | **10.3** / 78 | **10.0** / 79 | **10.7** / 76 |
| `hindsight-all` | **6.0** / 48 | **6.3** / 56 | **9.0** / 61 | **9.3** / 72 | **9.7** / 79 | **10.3** / 77 |
| `shortened` | — | — | **11.0** / 67 | **13.0** / 81 | **12.0** / 77 | — |

## Search shape, on top of the cache

| cell | tasks /300 | behaviours /43 | wall (s) | executions |
|---|---:|---:|---:|---:|
| `on-s0-e4` | 117 | 13 | 162 | 58,736,228 |
| `on-s0-e4-b128` | 121 | 14 | 321 | 113,665,637 |
| `on-s0-e4-b256` | 134 | 16 | 662 | 209,416,601 |
| `on-s0-e4-b512` | 160 | 18 | 965 | 334,439,543 |
| `on-s0-e4-d20` | 97 | 9 | 68 | 26,487,453 |

