# distinct-programs (#174): more distinct programs, same eval coverage

**Question.** Do more distinct training programs raise verified v4 eval behaviours, and if so by composition (novel programs) or retrieval (contained ones)?

**Data.** 532 public behaviours (BIG-bench list_functions, DreamCoder, LambdaBeam, MBPP/CodeARC leftovers) → Python references by `gpt-6-sol` xhigh → signature audit against v4 (exact duplicates, ≥0.8 agreement with any of the 43 eval-cohort behaviours, new-vs-new duplicates) → **401 kept** → `mark-28` witnesses by `gpt-6-sol` xhigh under the #155 staged brief plus the owner's caution against contrived stages → **395 verified** on every test and none with an eval skeleton: 96 staged, 78 plain, 221 rule_break (mostly zip / take-k cap limits). 40 tasks each, 15,800 tasks. `tasks.jsonl` is gitignored (BIG-bench canary).

**Arms.** #101 1m policy, seeds 0/1/2, epoch 3, cohort 300 tasks / 43 behaviours, beam 64 × depth 48, 10 s task cap, verified = 10,000-draw `genuine` or equals a catalogue witness.
- `staged`: #155's policies, re-searched (reproduces 143/128/148 graded).
- `staged_new`: staged + new tasks, rows scaled with task count (1,092,530 rows/epoch, 3.5×): exposure-matched, **primary**.
- `staged_new_fixed`: staged + new tasks at 313,560 rows/epoch: compute-matched.

**Registered falsifier (both must hold for `staged_new`).** (1) verified mean ≥ staged + 2, every new seed above every staged seed; (2) novel mean ≥ staged novel + 3, every new seed ≥ 7.

| arm | graded tasks | verified tasks | verified behaviours /43 | union | novel | only contained |
|---|---|---|---|---:|---|---|
| staged | 139.7 (143 / 128 / 148) | 137.7 | 15.67 (16 / 16 / 15) | 16 | 4.33 (4 / 6 / 3) | 11.33 |
| staged_new | 115.7 (116 / 113 / 118) | 112.3 | 15.67 (16 / 15 / 16) | 17 | 5.00 (3 / 7 / 5) | 10.67 |
| staged_new_fixed | 142.0 (126 / 140 / 160) | 136.0 | 14.67 (14 / 15 / 15) | 16 | 2.00 (1 / 3 / 2) | 12.67 |

**Verdict: not met on either part.** Verified behaviours are flat (15.67 vs 15.67), novel moves 4.33 → 5.00 within seed spread (3–7). Doubling the distinct-program count (220 → 615 behaviours) buys nothing at this recipe. The exposure-matched arm also graded ~24 fewer tasks, mostly in the 6–10 op bucket (49.7 → 32.3), so v4-specific fit was diluted without a compositional return. Novel behaviours, union over seeds: staged adds nothing the new arms lack except `keep_zeros`; `staged_new` adds `drop_last`, `increment_each`, `keep_negative`, `min_pair`.

**Leak (independent Opus review).** The audit pooled agreement over each new behaviour's own tests, which dilutes it. On 200 fresh draws from the eval samplers, `lb-082` matches `prefix_sums` on 0.87 of inputs, and the first stage of `lb-093`'s witness *is* a `prefix_sums` program. `prefix_sums` is verified in `staged_new` s0 and `staged_new_fixed` s2 (never in staged) by exactly that stage, so it counts as contained, not novel. Its uncontained one-token variant is a 10,000-draw shortcut. `dc-keep-mod-k-k5` ≡ keep_zeros and `lf-c250` ≈ reverse also slipped through but change nothing (every arm solves them). **Without prefix_sums: staged_new 15.33 (15 / 15 / 16), staged_new_fixed 14.33**, novel unchanged. The leak only strengthens the null. A future audit should use the eval samplers' inputs and check stage outputs.

**Opus 5.5 witnesses (same 401 behaviours, effort medium, blind to Sol's).** 399 verified, 2 unsolved: 207 staged / 91 plain / 101 rule_break, against Sol's 96 / 78 / 221. Opus stages twice as often under the same brief. Not trained: after this null it would test authorship, not distinctness. Programs in the `opus-dp` worktree, `opus_check.out` here.

Files: `arms.png`, `graded.md`, `verified.md`, `verified_solves.json`, `containment.json`; pipeline in `experiments/distinct_programs/`.
