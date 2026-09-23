## world arm 1m-s0-e5, #101 cohort (300 tasks, 43 behaviours)

300 tasks, 9633 layers, 654,121 ranked successors.

| verdict | count | share of ranked |
|---|---:|---:|
| retained | 601,473 | 92.0% |
| duplicate | 17,956 | 2.7% |
| guard:close | 34,519 | 5.3% |
| guard:type | 173 | 0.0% |

| duplicate class | open | closed |
|---|---:|---:|
| cycle | 3,125 | 9,172 |
| same-layer | 214 | 1,172 |
| earlier-layer | 318 | 3,955 |

- closed cross-path duplicates per closed retained state: **7.2%** (5,127 / 71,061), in 288 of 300 tasks
- cross-path share of duplicates: 31.5%; the rest are cycles
- distinct closed states a second route reached, per closed retained state: 5.8% (4,128)
- closed share of retained states: 11.8%
- effective width: 97.6% of the beam filled per layer
- below the cut (never checked by the search): 13,956,002 successors, 1,847,857 a guard would refuse, 338,135 already seen (2.8%; 255,602 closed); duplicate share of all allowed successors 2.8%

Where the second route meets the first (closed cross-path duplicates):

| program length at meeting | count | | second route longer by | count |
|---:|---:|---|---:|---:|
| 1 | 1 | | 0 | 1,172 |
| 2 | 1,502 | | 1 | 1,601 |
| 3 | 585 | | 2 | 502 |
| 4 | 180 | | 3 | 166 |
| 5 | 143 | | 4 | 41 |
| 6 | 42 | | 5 | 338 |
| 7 | 70 | | 6 | 43 |
| 8 | 113 | | 7 | 56 |
| 9 | 67 | | 8 | 37 |
| 10 | 131 | | 9 | 15 |
| 11 | 63 | | 10 | 68 |
| 12 | 94 | | 11 | 45 |
| 13 | 25 | | 12 | 42 |
| 14 | 30 | | 13 | 105 |
| 15 | 37 | | 14 | 19 |
| 16 | 62 | | 15 | 39 |
| 17 | 71 | | 16 | 36 |
| 18 | 72 | | 17 | 72 |
| 19 | 38 | | 18 | 75 |
| 20 | 99 | | 19 | 27 |
| 21 | 73 | | 20 | 27 |
| 22 | 46 | | 21 | 22 |
| 23 | 35 | | 22 | 15 |
| 24 | 44 | | 23 | 33 |
| 25 | 44 | | 24 | 34 |
| 26 | 24 | | 25 | 7 |
| 27 | 38 | | 26 | 46 |
| 28 | 59 | | 27 | 25 |
| 29 | 79 | | 28 | 3 |
| 30 | 52 | | 29 | 44 |
| 31 | 90 | | 30 | 17 |
| 32 | 36 | | 31 | 20 |
| 33 | 42 | | 32 | 9 |
| 34 | 30 | | 33 | 8 |
| 35 | 20 | | 34 | 10 |
| 36 | 18 | | 35 | 5 |
| 37 | 15 | | 36 | 28 |
| 38 | 23 | | 37 | 12 |
| 39 | 18 | | 38 | 14 |
| 40 | 15 | | 39 | 6 |
| 41 | 34 | | 40 | 13 |
| 42 | 40 | | 41 | 15 |
| 43 | 44 | | 42 | 19 |
| 44 | 73 | | 43 | 6 |
| 45 | 58 | | 44 | 37 |
| 46 | 146 | | 45 | 13 |
| 47 | 132 | | 46 | 102 |
| 48 | 374 | | 47 | 38 |
