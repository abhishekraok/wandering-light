## Findings (written with the 2026-09-11 wave 3; the tables below regenerate)

- **99 of 100 verified** (50/50 MBPP, 49/50 CodeARC); the miss is fuel
  (divisor sums of factor pairs up to a limit of 100). Claimed and verified
  statuses agreed on all 100. Over the probe and three waves, MBPP stands at
  184/186 and CodeARC at 130/134.
- **Cost keeps falling:** 277k tokens, 24 agent-minutes, 6 minutes wall for
  100 behaviours (61k–77k tokens per agent; no agent above 6 minutes).
- **Library: 8 macros per 100** (117 → 125). Three authors across waves had
  written `factorial`; `binomial` was used by three programs and
  `primes-below` by two; `slice-index` (Python's negative index, clamped),
  `head-or-neg1` and the sort-by-key trio (`key-gt?`, `insert-by-head`,
  `sort-by-head`) had each been written by an author in an earlier wave as
  well. Two authors wrote the same inclusive-range helper (`over sub inc range
  swap [ add ] swap cons map`), which differs from the canonical
  `range-between` on negative bounds; it stays inlined for now and is the
  candidate replacement when the library is next re-expanded for a corpus
  build. Aliasing it to `range-between` cost two witnesses until re-checked,
  which is what the migration re-verification is for.
- **Overlap with v4**: two exact witness matches (`codearc-22`, `mbpp-972`,
  both scalar arithmetic) and four shared skeletons.
