## BEST staged-s0-e3, world-v3 eval (44 tasks)

44 tasks, 2071 layers, 142,353 ranked successors.

| verdict | count | share of ranked |
|---|---:|---:|
| retained | 130,344 | 91.6% |
| duplicate | 10,263 | 7.2% |
| guard:close | 1,744 | 1.2% |
| guard:type | 2 | 0.0% |

| duplicate class | open | closed |
|---|---:|---:|
| cycle | 3,139 | 2,510 |
| same-layer | 859 | 294 |
| earlier-layer | 1,198 | 2,263 |

- closed cross-path duplicates per closed retained state: **18.9%** (2,557 / 13,500), in 44 of 44 tasks
- cross-path share of duplicates: 45.0%; the rest are cycles
- distinct closed states a second route reached, per closed retained state: 9.0% (1,221)
- closed share of retained states: 10.4%
- effective width: 98.3% of the beam filled per layer
- below the cut (never checked by the search): 2,574,320 successors, 179,509 a guard would refuse, 186,792 already seen (7.8%; 85,882 closed); among the ranked the guards allowed, 7.3%; over all allowed successors 7.8%

Where the second route meets the first (closed cross-path duplicates):

| program length at meeting | count | | second route longer by | count |
|---:|---:|---|---:|---:|
| 1 | 2 | | 0 | 294 |
| 2 | 261 | | 1 | 374 |
| 3 | 172 | | 2 | 157 |
| 4 | 62 | | 3 | 101 |
| 5 | 63 | | 4 | 48 |
| 6 | 44 | | 5 | 107 |
| 7 | 37 | | 6 | 99 |
| 8 | 58 | | 7 | 82 |
| 9 | 54 | | 8 | 74 |
| 10 | 99 | | 9 | 116 |
| 11 | 81 | | 10 | 78 |
| 12 | 120 | | 11 | 81 |
| 13 | 94 | | 12 | 93 |
| 14 | 78 | | 13 | 43 |
| 15 | 63 | | 14 | 67 |
| 16 | 67 | | 15 | 42 |
| 17 | 46 | | 16 | 57 |
| 18 | 47 | | 17 | 30 |
| 19 | 40 | | 18 | 28 |
| 20 | 39 | | 19 | 30 |
| 21 | 54 | | 20 | 36 |
| 22 | 41 | | 21 | 21 |
| 23 | 38 | | 22 | 29 |
| 24 | 47 | | 23 | 14 |
| 25 | 35 | | 24 | 17 |
| 26 | 37 | | 25 | 25 |
| 27 | 48 | | 26 | 41 |
| 28 | 62 | | 27 | 28 |
| 29 | 55 | | 28 | 21 |
| 30 | 37 | | 29 | 10 |
| 31 | 24 | | 30 | 8 |
| 32 | 29 | | 31 | 12 |
| 33 | 26 | | 32 | 13 |
| 34 | 31 | | 33 | 21 |
| 35 | 31 | | 34 | 17 |
| 36 | 28 | | 35 | 14 |
| 37 | 18 | | 36 | 17 |
| 38 | 31 | | 37 | 16 |
| 39 | 28 | | 38 | 17 |
| 40 | 42 | | 39 | 10 |
| 41 | 34 | | 40 | 25 |
| 42 | 33 | | 41 | 15 |
| 43 | 38 | | 42 | 18 |
| 44 | 28 | | 43 | 13 |
| 45 | 18 | | 44 | 17 |
| 46 | 32 | | 45 | 15 |
| 47 | 28 | | 46 | 28 |
| 48 | 77 | | 47 | 38 |
