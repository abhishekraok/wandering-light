## world arm 1m-s0-e5, world-v3 eval (44 tasks)

44 tasks, 2071 layers, 141,449 ranked successors.

| verdict | count | share of ranked |
|---|---:|---:|
| retained | 130,344 | 92.1% |
| duplicate | 5,689 | 4.0% |
| guard:close | 5,399 | 3.8% |
| guard:type | 17 | 0.0% |

| duplicate class | open | closed |
|---|---:|---:|
| cycle | 1,440 | 2,447 |
| same-layer | 67 | 428 |
| earlier-layer | 198 | 1,109 |

- closed cross-path duplicates per closed retained state: **8.7%** (1,537 / 17,578), in 44 of 44 tasks
- cross-path share of duplicates: 31.7%; the rest are cycles
- distinct closed states a second route reached, per closed retained state: 6.3% (1,101)
- closed share of retained states: 13.5%
- effective width: 98.3% of the beam filled per layer

Where the second route meets the first (closed cross-path duplicates):

| program length at meeting | count | | second route longer by | count |
|---:|---:|---|---:|---:|
| 1 | 2 | | 0 | 428 |
| 2 | 322 | | 1 | 230 |
| 3 | 101 | | 2 | 254 |
| 4 | 48 | | 3 | 36 |
| 5 | 64 | | 4 | 38 |
| 6 | 50 | | 5 | 60 |
| 7 | 20 | | 6 | 17 |
| 8 | 29 | | 7 | 27 |
| 9 | 34 | | 8 | 22 |
| 10 | 31 | | 9 | 27 |
| 11 | 44 | | 10 | 61 |
| 12 | 42 | | 11 | 20 |
| 13 | 33 | | 12 | 20 |
| 14 | 34 | | 13 | 52 |
| 15 | 22 | | 14 | 15 |
| 16 | 23 | | 15 | 20 |
| 17 | 17 | | 16 | 15 |
| 18 | 33 | | 17 | 36 |
| 19 | 30 | | 18 | 14 |
| 20 | 21 | | 19 | 5 |
| 21 | 28 | | 20 | 7 |
| 22 | 26 | | 22 | 3 |
| 23 | 12 | | 23 | 6 |
| 24 | 12 | | 24 | 20 |
| 25 | 22 | | 25 | 1 |
| 26 | 10 | | 26 | 16 |
| 27 | 20 | | 27 | 9 |
| 28 | 17 | | 28 | 1 |
| 29 | 22 | | 29 | 7 |
| 30 | 10 | | 30 | 3 |
| 31 | 24 | | 31 | 9 |
| 32 | 9 | | 33 | 8 |
| 33 | 22 | | 34 | 1 |
| 34 | 10 | | 35 | 1 |
| 35 | 26 | | 36 | 6 |
| 36 | 11 | | 38 | 4 |
| 37 | 16 | | 39 | 2 |
| 38 | 14 | | 40 | 4 |
| 39 | 16 | | 41 | 2 |
| 40 | 4 | | 42 | 3 |
| 41 | 23 | | 43 | 8 |
| 42 | 17 | | 44 | 7 |
| 43 | 14 | | 45 | 1 |
| 44 | 27 | | 46 | 10 |
| 45 | 16 | | 47 | 1 |
| 46 | 26 | | | |
| 47 | 19 | | | |
| 48 | 64 | | | |
