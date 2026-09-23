## BEST staged-s0-e3, #101 cohort (300 tasks, 43 behaviours)

300 tasks, 8271 layers, 555,164 ranked successors.

| verdict | count | share of ranked |
|---|---:|---:|
| retained | 514,305 | 92.6% |
| duplicate | 31,338 | 5.6% |
| guard:close | 9,510 | 1.7% |
| guard:size | 1 | 0.0% |
| guard:type | 10 | 0.0% |

| duplicate class | open | closed |
|---|---:|---:|
| cycle | 5,208 | 14,141 |
| same-layer | 739 | 1,793 |
| earlier-layer | 1,645 | 7,812 |

- closed cross-path duplicates per closed retained state: **7.6%** (9,605 / 126,344), in 288 of 300 tasks
- cross-path share of duplicates: 38.3%; the rest are cycles
- distinct closed states a second route reached, per closed retained state: 5.4% (6,837)
- closed share of retained states: 24.6%
- effective width: 97.2% of the beam filled per layer
- below the cut (never checked by the search): 10,522,835 successors, 997,079 a guard would refuse, 600,092 already seen (6.3%; 483,394 closed); among the ranked the guards allowed, 5.7%; over all allowed successors 6.3%

Where the second route meets the first (closed cross-path duplicates):

| program length at meeting | count | | second route longer by | count |
|---:|---:|---|---:|---:|
| 1 | 1 | | 0 | 1,793 |
| 2 | 1,122 | | 1 | 1,931 |
| 3 | 679 | | 2 | 870 |
| 4 | 277 | | 3 | 488 |
| 5 | 231 | | 4 | 166 |
| 6 | 164 | | 5 | 636 |
| 7 | 170 | | 6 | 477 |
| 8 | 236 | | 7 | 236 |
| 9 | 128 | | 8 | 307 |
| 10 | 227 | | 9 | 188 |
| 11 | 282 | | 10 | 388 |
| 12 | 292 | | 11 | 207 |
| 13 | 223 | | 12 | 163 |
| 14 | 308 | | 13 | 95 |
| 15 | 184 | | 14 | 190 |
| 16 | 301 | | 15 | 124 |
| 17 | 210 | | 16 | 168 |
| 18 | 197 | | 17 | 60 |
| 19 | 196 | | 18 | 85 |
| 20 | 218 | | 19 | 74 |
| 21 | 175 | | 20 | 78 |
| 22 | 228 | | 21 | 47 |
| 23 | 178 | | 22 | 34 |
| 24 | 125 | | 23 | 41 |
| 25 | 161 | | 24 | 53 |
| 26 | 178 | | 25 | 44 |
| 27 | 181 | | 26 | 68 |
| 28 | 188 | | 27 | 39 |
| 29 | 153 | | 28 | 25 |
| 30 | 159 | | 29 | 43 |
| 31 | 126 | | 30 | 43 |
| 32 | 126 | | 31 | 44 |
| 33 | 116 | | 32 | 36 |
| 34 | 116 | | 33 | 18 |
| 35 | 117 | | 34 | 25 |
| 36 | 100 | | 35 | 24 |
| 37 | 121 | | 36 | 18 |
| 38 | 109 | | 37 | 14 |
| 39 | 114 | | 38 | 22 |
| 40 | 134 | | 39 | 27 |
| 41 | 119 | | 40 | 15 |
| 42 | 116 | | 41 | 15 |
| 43 | 122 | | 42 | 21 |
| 44 | 130 | | 43 | 17 |
| 45 | 85 | | 44 | 16 |
| 46 | 116 | | 45 | 15 |
| 47 | 129 | | 46 | 23 |
| 48 | 237 | | 47 | 94 |
