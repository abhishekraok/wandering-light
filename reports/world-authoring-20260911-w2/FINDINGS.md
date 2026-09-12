## Findings (written with the 2026-09-11 wave 2; the tables below regenerate)

- **MBPP and CodeARC verify at 97%** (50/50 MBPP, 47/50 CodeARC; 96 as
  authored plus one identity behaviour whose author submitted the empty
  program, recorded as a two-op repair). The three misses are fuel: a
  breadth-first factorisation, a digit-window search and Langton's ant with a
  10,000-step preperiod. Yield on these two sources now stands at 96% over
  135 behaviours across the probe, wave 1 and wave 2.
- **Cheaper than wave 1.** 289k tokens and 23 agent-minutes for 100
  behaviours (59k–86k tokens per agent), 11 minutes wall; the shorter MBPP and
  CodeARC programs (macro-form median 15 and 21 tokens) need fewer
  verifier rounds than LeetCode-medium did.
- **The library gained 7 macros per 100 behaviours, down from 16.** Authors
  wrote 90 private macros; `popcount` was written by two authors, `prime?`,
  `bisect-right` and `next-palindrome` were each used by two programs, and
  `lcm`, `set-at` and a sign-correct `sxor` had each been written by an
  author in wave 1 as well, so all seven went canonical (110 → 117). The rest
  are one-behaviour step functions, inlined. No library defect this time.
- **Overlap with v4 stays negligible**: no exact witness match, one shared
  skeleton.
