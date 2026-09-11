## Findings (written with the 2026-09-11 wave 5; the tables below regenerate)

- **98 of 100 verified** (49/50 MBPP, 49/50 CodeARC); the misses are one
  CPython set-iteration order (`mbpp-193`, the third of its kind) and one
  fuel exhaustion (a prime sieve to large bounds). Over the probe and five
  waves MBPP stands at 283/286 and CodeARC at 229/234.
- **Cost:** 318k tokens, 24 agent-minutes, 10 minutes wall.
- **Library: 4 macros per 100** (129 → 133): `enumerate` (three programs),
  `highest-pow2` (two), `choose2` (also written in wave 2), and `index-of`,
  the concept two authors across waves wrote as private helpers with
  different behaviour on a missing element. A task-specific pair helper used
  by two MBPP programs (`first-even-odd`) stayed inlined: two uses of one
  problem family is not a concept. Growth per 100 over the five waves: 16, 7,
  8, 4, 4.
- **Overlap with v4:** 7 exact witness matches, all scalar one-liners.
- **The sampler-passing pool is nearly spent:** this wave tried 96 MBPP
  candidates to keep 50 and 69 CodeARC to keep 50. About 96 MBPP and 31
  CodeARC behaviours remain eligible.
