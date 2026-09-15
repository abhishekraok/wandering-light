## Full library

### Compression of distinct witnesses under the 133-macro library (fewest tokens over ops U macros)

| corpus | tasks | witnesses | median length | median compressed | ops saved | share of ops saved | share of ops inside a macro | witnesses with a hit | macros used |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v4 train | 7000 | 175 | 44 | 31 | 2668 / 8490 | 0.314 | 0.377 | 0.891 | 35 / 133 |
| v4 eval | 1800 | 45 | 40 | 30 | 766 / 2233 | 0.343 | 0.405 | 0.778 | 26 / 133 |
| v4 cohort (#101) | 300 | 43 | 39 | 30 | 729 / 2119 | 0.344 | 0.407 | 0.767 | 25 / 133 |
| world-v3 train | 522 | 482 | 145 | 21 | 101524 / 123058 | 0.825 | 0.871 | 0.977 | 126 / 133 |
| world-v3 eval | 44 | 44 | 149.5 | 28 | 8045 / 9892 | 0.813 | 0.860 | 1.000 | 80 / 133 |

### Effective branching at witness-prefix states (8 per witness, every shown example must accept; no #70 guard) and the #96 step-3 estimate, mean over witnesses of d · log2 b

| corpus | states | b_eff primitives | b_eff primitives U macros | log2 cost raw | log2 cost library |
|---|---:|---:|---:|---:|---:|
| v4 train | 1340 | 24.8 | 137.1 | 224.9 | 236.2 |
| v4 eval | 336 | 24.4 | 133.2 | 228.8 | 230.1 |
| v4 cohort (#101) | 320 | 24.7 | 135.7 | 228.0 | 229.0 |
| world-v3 train | 3741 | 24.7 | 137.5 | 1181.1 | 317.4 |
| world-v3 eval | 352 | 25.5 | 142.3 | 1050.5 | 300.2 |

### By witness-length bucket

| corpus | bucket | witnesses | median length | median compressed | share of ops saved | witnesses with a hit | log2 cost raw | log2 cost library |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v4 train | 1-5 | 9 | 3 | 1 | 0.429 | 0.556 | 14.4 | 12.6 |
| v4 train | 6-10 | 11 | 6 | 5 | 0.333 | 0.727 | 30.3 | 31.0 |
| v4 train | 11-20 | 28 | 15 | 7 | 0.422 | 0.679 | 69.0 | 61.1 |
| v4 train | 21-40 | 28 | 31.5 | 20 | 0.329 | 0.964 | 139.9 | 143.8 |
| v4 train | 41+ | 99 | 64 | 42 | 0.306 | 0.980 | 333.7 | 355.0 |
| v4 eval | 1-5 | 6 | 5 | 4.5 | 0.172 | 0.333 | 22.3 | 28.2 |
| v4 eval | 6-10 | 3 | 6 | 6 | 0.105 | 0.333 | 29.2 | 40.0 |
| v4 eval | 11-20 | 6 | 15.5 | 15.5 | 0.226 | 0.500 | 71.5 | 84.7 |
| v4 eval | 21-40 | 9 | 39 | 22 | 0.262 | 0.889 | 154.2 | 174.1 |
| v4 eval | 41+ | 21 | 76 | 45 | 0.368 | 1.000 | 393.2 | 380.4 |
| v4 cohort (#101) | 1-5 | 6 | 5 | 4.5 | 0.172 | 0.333 | 22.4 | 28.3 |
| v4 cohort (#101) | 6-10 | 3 | 6 | 6 | 0.105 | 0.333 | 29.3 | 40.1 |
| v4 cohort (#101) | 11-20 | 6 | 15.5 | 15.5 | 0.226 | 0.500 | 71.7 | 85.0 |
| v4 cohort (#101) | 21-40 | 9 | 39 | 22 | 0.262 | 0.889 | 154.7 | 174.7 |
| v4 cohort (#101) | 41+ | 19 | 78 | 45 | 0.371 | 1.000 | 408.3 | 393.4 |
| world-v3 train | 1-5 | 21 | 2 | 2 | 0.369 | 0.524 | 14.3 | 13.9 |
| world-v3 train | 6-10 | 16 | 7.5 | 5 | 0.368 | 0.938 | 33.8 | 32.9 |
| world-v3 train | 11-20 | 33 | 15 | 5 | 0.614 | 1.000 | 69.8 | 41.3 |
| world-v3 train | 21-40 | 33 | 28 | 12 | 0.569 | 1.000 | 134.6 | 89.1 |
| world-v3 train | 41+ | 379 | 198 | 27 | 0.829 | 1.000 | 1482.1 | 390.1 |
| world-v3 eval | 6-10 | 1 | 8 | 6 | 0.250 | 1.000 | 37.4 | 42.9 |
| world-v3 eval | 21-40 | 3 | 24 | 7 | 0.680 | 1.000 | 116.8 | 57.2 |
| world-v3 eval | 41+ | 40 | 166 | 29.5 | 0.815 | 1.000 | 1145.9 | 324.9 |

### Macros by ops saved (top 15)

| corpus | macro: ops saved |
|---|---|
| v4 train | `pack3` 300, `abs` 276, `max2` 253, `unpack3` 252, `pack2` 152, `reverse` 140, `second` 108, `unpack2` 104, `take` 98, `drop-n` 98, `len` 90, `min` 90, `head` 84, `min2` 77, `max` 75 |
| v4 eval | `abs` 120, `nth` 104, `pack3` 90, `unpack3` 54, `reverse` 48, `max` 45, `min` 45, `pack2` 36, `head` 32, `unpack2` 28, `second` 24, `len` 24, `max2` 22, `not` 20, `2` 14 |
| v4 cohort (#101) | `abs` 108, `nth` 104, `pack3` 90, `unpack3` 54, `reverse` 48, `max` 45, `pack2` 36, `head` 32, `min` 30, `unpack2` 28, `len` 24, `second` 24, `max2` 22, `not` 20, `gt?` 13 |
| world-v3 train | `range` 13081, `range1` 5076, `nth` 4160, `take` 3920, `pow` 3689, `set-at` 3503, `sxor` 2724, `decimal-digits` 2522, `digits` 2490, `zip` 2376, `at` 2343, `drop-n` 2107, `pow2` 1812, `sort` 1792, `isqrt` 1767 |
| world-v3 eval | `range` 1339, `sxor` 681, `range1` 432, `gcd` 424, `pow` 357, `isqrt` 279, `adjacent` 236, `prime?` 234, `nth` 208, `enumerate` 179, `digit-sum` 176, `10` 160, `bxor` 155, `bool>int` 140, `abs` 132 |

### World witnesses: the DP parse against the authors' own macro form

| corpus | median authored tokens | median DP tokens | DP shorter | DP longer |
|---|---:|---:|---:|---:|
| world-v3 train | 21 | 21 | 64 | 0 |
| world-v3 eval | 28.5 | 28 | 9 | 0 |

## Subset: the 35 macros that hit a v4 train witness (selected on train; v4 eval and the cohort are held out)

`-1`, `2`, `abs`, `bool>int`, `count`, `dec`, `drop-n`, `flatten`, `gt?`, `head`, `inc`, `last`, `len`, `max`, `max2`, `min`, `min2`, `mod`, `neg`, `not`, `nth`, `pack2`, `pack3`, `pick2`, `product`, `reverse`, `rot`, `second`, `sq`, `sum`, `tail`, `take`, `third`, `unpack2`, `unpack3`

### Compression of distinct witnesses under the 35-macro library (fewest tokens over ops U macros)

| corpus | tasks | witnesses | median length | median compressed | ops saved | share of ops saved | share of ops inside a macro | witnesses with a hit | macros used |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v4 train | 7000 | 175 | 44 | 31 | 2668 / 8490 | 0.314 | 0.377 | 0.891 | 35 / 35 |
| v4 eval | 1800 | 45 | 40 | 30 | 765 / 2233 | 0.343 | 0.405 | 0.778 | 25 / 35 |
| v4 cohort (#101) | 300 | 43 | 39 | 30 | 729 / 2119 | 0.344 | 0.407 | 0.767 | 25 / 35 |
| world-v3 train | 522 | 482 | 145 | 83.5 | 55572 / 123058 | 0.452 | 0.541 | 0.977 | 35 / 35 |
| world-v3 eval | 44 | 44 | 149.5 | 90 | 4065 / 9892 | 0.411 | 0.508 | 1.000 | 35 / 35 |

### Effective branching at witness-prefix states (8 per witness, every shown example must accept; no #70 guard) and the #96 step-3 estimate, mean over witnesses of d · log2 b

| corpus | states | b_eff primitives | b_eff primitives U macros | log2 cost raw | log2 cost library |
|---|---:|---:|---:|---:|---:|
| v4 train | 1340 | 24.7 | 54.0 | 224.4 | 191.5 |
| v4 eval | 336 | 24.9 | 54.4 | 230.1 | 188.1 |
| v4 cohort (#101) | 320 | 24.5 | 53.3 | 227.3 | 185.4 |
| world-v3 train | 3741 | 24.6 | 53.8 | 1180.1 | 805.0 |
| world-v3 eval | 352 | 25.1 | 55.1 | 1045.1 | 766.1 |

### By witness-length bucket

| corpus | bucket | witnesses | median length | median compressed | share of ops saved | witnesses with a hit | log2 cost raw | log2 cost library |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v4 train | 1-5 | 9 | 3 | 1 | 0.429 | 0.556 | 14.4 | 10.2 |
| v4 train | 6-10 | 11 | 6 | 5 | 0.333 | 0.727 | 30.3 | 25.1 |
| v4 train | 11-20 | 28 | 15 | 7 | 0.422 | 0.679 | 68.9 | 49.5 |
| v4 train | 21-40 | 28 | 31.5 | 20 | 0.329 | 0.964 | 139.6 | 116.6 |
| v4 train | 41+ | 99 | 64 | 42 | 0.306 | 0.980 | 333.0 | 287.8 |
| v4 eval | 1-5 | 6 | 5 | 4.5 | 0.172 | 0.333 | 22.4 | 23.1 |
| v4 eval | 6-10 | 3 | 6 | 6 | 0.105 | 0.333 | 29.4 | 32.7 |
| v4 eval | 11-20 | 6 | 15.5 | 15.5 | 0.226 | 0.500 | 71.9 | 69.2 |
| v4 eval | 21-40 | 9 | 39 | 22 | 0.262 | 0.889 | 155.1 | 142.2 |
| v4 eval | 41+ | 21 | 76 | 45 | 0.367 | 1.000 | 395.5 | 311.1 |
| v4 cohort (#101) | 1-5 | 6 | 5 | 4.5 | 0.172 | 0.333 | 22.3 | 22.9 |
| v4 cohort (#101) | 6-10 | 3 | 6 | 6 | 0.105 | 0.333 | 29.2 | 32.5 |
| v4 cohort (#101) | 11-20 | 6 | 15.5 | 15.5 | 0.226 | 0.500 | 71.5 | 68.8 |
| v4 cohort (#101) | 21-40 | 9 | 39 | 22 | 0.262 | 0.889 | 154.3 | 141.5 |
| v4 cohort (#101) | 41+ | 19 | 78 | 45 | 0.371 | 1.000 | 407.1 | 318.4 |
| world-v3 train | 1-5 | 21 | 2 | 2 | 0.369 | 0.524 | 14.3 | 11.2 |
| world-v3 train | 6-10 | 16 | 7.5 | 5 | 0.325 | 0.938 | 33.8 | 28.4 |
| world-v3 train | 11-20 | 33 | 15 | 7 | 0.520 | 1.000 | 69.8 | 41.6 |
| world-v3 train | 21-40 | 33 | 28 | 13 | 0.507 | 1.000 | 134.5 | 82.4 |
| world-v3 train | 41+ | 379 | 198 | 110 | 0.451 | 1.000 | 1480.8 | 1011.1 |
| world-v3 eval | 6-10 | 1 | 8 | 6 | 0.250 | 1.000 | 37.2 | 34.7 |
| world-v3 eval | 21-40 | 3 | 24 | 7 | 0.640 | 1.000 | 116.2 | 52.1 |
| world-v3 eval | 41+ | 40 | 166 | 106 | 0.409 | 1.000 | 1140.0 | 837.9 |

### Macros by ops saved (top 15)

| corpus | macro: ops saved |
|---|---|
| v4 train | `pack3` 300, `abs` 276, `max2` 253, `unpack3` 252, `pack2` 152, `reverse` 140, `second` 108, `unpack2` 104, `take` 98, `drop-n` 98, `len` 90, `min` 90, `head` 84, `min2` 77, `max` 75 |
| v4 eval | `abs` 120, `nth` 104, `pack3` 90, `unpack3` 54, `reverse` 48, `max` 45, `min` 45, `pack2` 36, `head` 32, `unpack2` 28, `second` 24, `len` 24, `max2` 22, `not` 20, `2` 16 |
| v4 cohort (#101) | `abs` 108, `nth` 104, `pack3` 90, `unpack3` 54, `reverse` 48, `max` 45, `pack2` 36, `head` 32, `min` 30, `unpack2` 28, `len` 24, `second` 24, `max2` 22, `not` 20, `gt?` 13 |
| world-v3 train | `nth` 6552, `take` 5488, `unpack2` 4868, `pack2` 4836, `drop-n` 4802, `2` 3270, `pick2` 2560, `rot` 2420, `not` 2270, `last` 2085, `mod` 1744, `bool>int` 1560, `len` 1506, `max2` 1221, `inc` 1064 |
| world-v3 eval | `pack2` 436, `unpack2` 428, `2` 356, `nth` 260, `drop-n` 245, `abs` 240, `last` 210, `bool>int` 200, `not` 200, `mod` 188, `rot` 184, `pick2` 160, `len` 126, `take` 98, `inc` 91 |

### World witnesses: the DP parse against the authors' own macro form

| corpus | median authored tokens | median DP tokens | DP shorter | DP longer |
|---|---:|---:|---:|---:|
| world-v3 train | 21 | 83.5 | 2 | 381 |
| world-v3 eval | 28.5 | 90 | 0 | 37 |
