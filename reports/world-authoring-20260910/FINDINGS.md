## Findings (written with the 2026-09-10 wave 1; the tables below regenerate)

- **Yield holds on MBPP and CodeARC, drops on LeetCode-medium and collapses on
  hard.** 92 of 100 verified as authored (35/36 MBPP, 34/34 CodeARC, 21/25
  LeetCode-medium, 2/5 hard), against the probe's 96% on MBPP + LeetCode-easy.
  Every LeetCode miss is fuel: dynamic programmes and enumerations whose tests
  reach n in the thousands against the 100,000-step cap. The one MBPP miss is
  CPython set-iteration order again (`mbpp-769`, the same class as `mbpp-579`).
  LeetCode is also exhausted: 40 medium/hard problems fit the domain in total,
  30 are here. The remaining pool is MBPP (~296) and CodeARC (~231).
- **Codex authors at roughly 4 minutes and 3.7k tokens per behaviour.** Four
  `codex exec` agents (gpt-6-astra, high effort), 24–26 behaviours each, ran
  7–14 minutes wall, 86k–111k tokens each, 372k in total; the whole wave took
  14 minutes on four agents. Claimed and verified statuses agreed on all 100.
- **The library grows about 16 macros per 100 behaviours and still catches
  library defects.** Authors wrote 128 private macros; 16 were reusable
  (three authors wrote `isqrt`, two `palindrome?`, two `pack5`; also
  `bitlen`/`log2`/`pow2`, `second`…`fifth`, `flatten`, `from-digits`,
  `decimal-digits`, `inversions`, `32`) and went canonical with tests, 112
  were one-behaviour step functions and were inlined. Two authors wrote their
  own `gcd` because the canonical one was a self-applying loop that died past
  ~7 Euclid steps (`gcd 832040 514229` refused); it is now a 32-step fold, at
  ~7× the fuel (700 vs 105), which broke one fuel-tight witness
  (`leetcode-number-of-self-divisible-permutations`, 14/14 → 12/14) until a
  Codex repair (22k tokens, `repairs.jsonl`) replaced `gcd` with a
  divisibility-based coprime test exact for values ≤ 12, the problem's stated
  bound (worst case 64k fuel). Macro form median stayed at 24 tokens after migration
  (16 in the authors' dialects); expanded median 163 ops, up from the
  probe's 111, because medium problems carry more state.
- **Overlap with v4 stays negligible**: one exact witness match (`mbpp-517`,
  a scalar identity) and three shared skeletons in 91.
