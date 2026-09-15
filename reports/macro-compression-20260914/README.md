# macro-compression-check (#144): do the authoring macros compress v4 witnesses?

Static, no search, no training. `experiments/macro_compression.py --out reports/macro-compression-20260914`,
about 8 minutes per pass on 16 CPU cores. Tables in [`tables.md`](tables.md), per-witness rows in
`compression.json`.

**Library.** The 133 macros of `experiments/world_behaviours/macros.txt`, each expanded to `mark-28`
primitives (2 to 917 ops; median 16). **Compression** of a witness is the fewest tokens over
ops ∪ macros that expand to it, by dynamic programming over positions, so it is exact for this
library, not greedy. **Branching** is measured at 8 prefix states per witness (seed 0), an action
legal iff every shown example accepts it under plain `step_raw` (no #70 guard, so the primitive
figure of ~25 is above what the guarded beam sees). The cost estimate is highways (#96) step 3's
closed form, `d · log2 b_eff`, mean over witnesses.

**Cells.** Distinct witnesses of `stack_pbe_v4` train (175), eval (45), the #101 cohort (43 of the
eval 45) and `stack_pbe_world_v3` train (482) and eval (44). Two passes: the full library, and the
35 macros that hit at least one v4 train witness (selected on train; v4 eval and the cohort are
held out from the selection).

## Findings (2026-09-14)

- **The library compresses v4 by a third, with 35 of its 133 macros, all plumbing.** 31 to 34% of
  ops saved, 77 to 89% of witnesses touched, median cohort witness 39 → 30 tokens. Every macro
  that hits v4 is stack plumbing or a one-liner: `pack2/3`, `unpack2/3`, `abs`, `max2`, `min2`,
  `reverse`, `head`, `nth`, `len`, `second`, `take`, `drop-n`. None of the semantic macros
  (`range`, `digits`, `gcd`, `sort`, `zip`, `primes-below`) occurs in any v4 witness. This is what
  highways (#96) predicted: "stack-shuffling idioms and accumulator plumbing, not semantic
  abstractions".
- **As an action set the full library is a wash on v4 and a loss where anything is solved.**
  Branching rises 25 → 137, and the estimate on the cohort is 228.0 → 229.0 bits: 1–5 ops
  22.4 → 28.3, 6–10 29.3 → 40.1, 11–20 71.7 → 85.0, 21–40 154.7 → 174.7, only 41+ falls,
  408.3 → 393.4. The compression (×0.77 depth) is cancelled by the branching (×1.53 in log2).
- **The 35-macro subset keeps all the compression and most of the branching.** On held-out v4 eval
  it saves 765 of 2,233 ops against the full library's 766; branching is 25 → 54, and the cohort
  estimate falls 227.3 → 185.4. Per bucket: 1–5 22.3 → 22.9, 6–10 29.2 → 32.5, 11–20 71.5 → 68.8,
  21–40 154.3 → 141.5, 41+ 407.1 → 318.4. The short buckets, the only ones any solver clears,
  still get no better; the gain is on witnesses that stay 22 to 45 tokens deep.
- **Not a highways library for v4.** #96's gate is a majority of validation witnesses certified
  shortest in library space at depth ≤ 5. This library takes the cohort's median from 39 to 30
  tokens. Mining from v4 witnesses (#96 step 2) is still the path; the 35 plumbing macros are the
  floor a mined library must beat, and they transfer: selected on v4 train, they lose nothing on
  v4 eval and compress world-v3 by 45%.
- **World witnesses are macro boilerplate in primitive form.** Median 145 ops, mean 255, 379 of
  482 train witnesses in the 41+ bucket; the full library takes them to a median of 21 tokens,
  which is exactly the authors' macro form (the DP parse is never longer than the authored
  program and shorter for 64 train and 9 eval witnesses). `range` alone accounts for 13,081 of
  the 101,524 ops saved on train. This is the text world-plus-v4-policy (#134) trained on, and it
  explains why its search gain was one contained chunk: the world corpus is a few dozen
  expansions repeated, not v4-shaped programs. Under the v4 subset world compresses 45%, so
  the plumbing is shared across corpora; the other 55% is world-only vocabulary.
