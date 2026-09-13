# world-plus-v4-policy (#134): does adding the world corpus beat a matched v4-only baseline?

Step 3 of training-data-sources (#118). The #101 next-op policy, arm `1m`
(655,836 parameters), trained twice under one recipe: arm A on `stack_pbe_v4`
alone, arm B on `stack_pbe_v4` plus `stack_pbe_world_v3`. Seeds 0, 1 and 2, six
epochs, Adam at 1.25e-4, batch 256, CUDA. Both arms choose their best epoch on the
**same** v4 holdout, 26,040 rows, and are searched on #101's 300-task v4 cohort and
on world-v3's eval split with beam 64 × depth 48.

**Status, 2026-09-13: the solve result is on hold.** An adversarial audit
recomputed every number below and found that solve counts track the epoch the
best-validation checkpoint was taken at, while no v4-only checkpoint was searched
past epoch 2. An epoch-resolved study is running. Until it lands, the solve
findings describe best-checkpoint numbers, not a data effect.
<!-- pending: epoch study -->

## Findings (written 2026-09-13; the tables below regenerate)

- **World data lowers next-op cross-entropy on v4's own holdout, on every seed.**
  0.9541 ± 0.0100 for v4-only against 0.8576 ± 0.0130 for v4+world; the paired
  delta is −0.0965 ± 0.0174 (−0.1085, −0.0765, −0.1044), about ten times the
  baseline's seed spread, on identical holdout rows. It is broad but not uniform:
  13 of the 16 holdout behaviours are lower on the seed mean and 10 on every seed,
  `ordered_pair` and `magnitude_ordered_pair` are higher on every seed, and no
  behaviour supplies more than 18% of the gap.
- **That number needs three caveats.** It is the model-selection metric itself, a
  minimum over six epochs on only 16 behaviours, so it flatters both arms. No
  world training behaviour shares a program or skeleton with a holdout behaviour,
  but `keep_above`'s whole 7-op witness occurs inside 37 world fit programs and no
  v4 fit program, and it supplies 14.1% of the gap from 1.1% of the rows. And an
  op 4-gram that never sees a stack gains 0.088 nats on the same holdout when world
  programs join its training text (1.280 to 1.192), about the size of the policy's
  gain, so much of it is consistent with better op-sequence statistics. The
  policies still beat every n-gram there by more than 0.3 nats, and rotating target
  stacks across tasks raises both arms' CE, so both use the examples.
- **Step count does not explain the CE gain, and a regularisation reading is not
  supported.** Arm A's validation CE rises monotonically after its best epoch on
  every seed, to 1.36–1.51 by epoch 6, and arm B is below arm A's best at 3, 5 and
  6 of its 6 epochs. Arm B's training CE is higher than arm A's in all 18
  epoch-seed pairs, but that is composition: at its best checkpoints arm B scores
  world fit rows at 2.5–3.5 times its v4 fit rows, and on v4 fit rows alone it fits
  *more* tightly than arm A (0.335, 0.192, 0.270 against 0.503, 0.294, 0.503) while
  still scoring lower on the holdout. The learning rate is the lowest value #101's
  probe tried, so a lower rate for arm A was never tested.
- **Verified solves on the v4 cohort are 48, 63, 46 for v4-only and 52, 104, 102
  for v4+world, and they are confounded with the selected epoch.** Checkpoints taken
  at epoch 1 solve 48 in each of three distinct runs, and at epoch 2 they solve
  51–63, including arm B seed 0 at 53. Arm B's 104 and 102 come from checkpoints
  taken at epochs 4 and 3, and no v4-only checkpoint was ever searched past epoch 2.
  A verified solve passes held-out grading and computes the behaviour: its program
  equals a catalogue witness, which the corpus checked against the Python reference
  on draws from the behaviour's own sampler, or it agrees with the reference on
  2,000 sampler draws and on the in-domain part of 2,000 perturbed draws. The only
  solved programs that fail are the two already on record in `STATUS.md`
  (Superseded, 2026-09-09): `all_positive` at 28 ops in both arms, and
  `bounds_cover_list_flag` at 44 ops in arm B seed 1.
- **Where the best-checkpoint gain sits.** `keep_below` adds 7, 8 and 16 solves,
  `negate_each` 1, 14 and 17, `keep_negative` 0, 9 and 9, `keep_zeros` 1, 1 and 11,
  `increment_each` −1, 9 and 0, and `shared_value_occurrences` 1, 5 and 5, while
  `drop_last` loses 5, 6 and 2 of its 10 tasks. Splitting the 43 cohort behaviours
  by whether their witness occurs inside world fit programs but in no v4 fit
  program gives +2, +25, +31 for those 5 and +2, +16, +25 for the other 38, so
  containment does not explain the gain; seed 0 is +2 in both groups. A bootstrap
  over behaviours gives +33.6 tasks per seed, 95% interval [+4.0, +70.0], treating
  the arm as the only difference, which the epoch confound says it is not.
- **Arm B's solves past 20 ops.** It solves 1, 6 and 5 tasks in the 21–40 bucket,
  where arm A and #101's arms at all three sizes solved none. That is 12 solves over
  three seeds but 6 distinct tasks from two behaviours, and one program does 11 of
  them: the 31-op catalogue witness of `shared_value_occurrences`, on 5 of its 6
  tasks. The other is `count_in_range`'s 40-op witness on 1 of its 5 tasks, in seed
  1 only. No training program computes either behaviour, but both witnesses are
  covered by fragments of training programs. `shared_value_occurrences`'s ops 1–24
  are v4 train's `disjoint_lists_flag` and its ops 10–31 occur in 13 world fit
  programs, so world data supplies the join. `count_in_range`'s are covered by v4
  alone, a 16-op prefix from `count_outside_interval` and a 23-op suffix from
  `sum_inside_interval`, with no more than 10 of its ops in any world program. The
  same epoch caveat applies.
- **World eval: the CE gap measures familiarity with world op sequences, and
  neither arm can search it.** On the 44 clean world eval behaviours arm A scores
  3.16 ± 0.61, at or worse than uniform over 28 ops (3.33; seed 1 is 3.86), and arm
  B 1.02 ± 0.05. An op 4-gram fit on world training programs, which never sees a
  stack, scores 0.91 there, and the audit's better-smoothed n-grams reached
  0.77–0.86, so the number does not show arm B inferring unseen behaviours. Solve
  rate at depth 48 is 1, 0, 0 of 44 in both arms, on different tasks: arm A's is
  `mbpp-672`, v4 train's `max_pair` witness written twice, and arm B's is
  `mbpp-436`, v4 eval's `keep_negative` followed by `uncons swap drop`. Only 4 of
  the 44 witnesses fit within 48 ops (median 149.5). Arm A's one depth-192 cell
  solved the same 3 of 55 as at depth 48; arm B's was stopped to free the GPU.
- **World-v3's eval split had 11 contaminated behaviours, and on solves they
  favoured neither arm.** All 11 share a skeleton with a world training behaviour,
  because v3 split by behaviour-id hash rather than by skeleton; 9 share the program
  too, and 3 also match v4 train (`mbpp-62` and `mbpp-924` by program, `mbpp-458`
  by skeleton). Both arms solved 2, 2 and 3 of them. On CE they flattered arm B
  more: dirty minus clean is −0.43, −0.52, −0.45 for arm B against −0.21, −0.39,
  −0.21 for arm A. They are removed from `stack_pbe_world_v3`
  <!-- pending: revision -->, with the train split byte-identical. "Clean" means no
  exact program or skeleton match, not no partial overlap.
- **Scope of any claim.** World supplies 26.9% of arm B's fit rows but 74.8% of its
  fit behaviours (473 of 632) and 409 skeletons that v4 fit lacks, and 49.4% of
  world fit rows come from the 54 behaviours longer than 500 ops, so world adds both
  diversity and long witnesses and this design cannot separate the two. The
  tokeniser keeps the top 160 tokens of each stack, truncating 42.7% of world fit
  example stacks (37.6% on world holdout, 24.6% on clean world eval) and none of
  v4's witness replays; the cap is part of each checkpoint's config, so raising it
  would break comparability with #101. Overlap with v4 eval was removed by exact
  program and skeleton identity; 65 world training tasks still share at least one
  literal input-output pair with v4 eval (155 pairs, 128 with `min_pair`), harmless
  as functions but proof that literal matching cannot detect overlap. Three paired
  seeds.
- **The baseline reproduces #101.** Arm A seeds 0 and 1 reach #101's recorded best
  validation CE (0.9519, 0.9454), and their v4 timing cells match #101's recorded
  cells row by row: task order, solved set and every program. The checkpoints carry
  no W&B run link; provenance is in each `training_manifest.json`.

## Design

| arm | training data | fit rows | validation rows |
|---|---|---|---|
| A | `stack_pbe_v4` train | 313,560 | 26,040 (v4 holdout) |
| B | `stack_pbe_v4` train + `stack_pbe_world_v3` train, three behaviours dropped | 429,078 | 26,040, the same rows |

**Held equal:** architecture and tokeniser, optimizer and learning rate (#101's
probe, `lr_probe.json`), batch size, epochs, seeds, the model-selection rows, the
300-task v4 cohort (verified identical to #101's), and the search budget.

**Dropped from arm B** (`drop_from_world_train.txt`): `codearc-128`, `mbpp-404` and
`mbpp-970`, whose witness is byte-identical to v4 eval's `min_pair`. Skeleton
identity against v4 eval finds the same three and no others. No world task repeats
a v4 eval task's shown examples. World's own 46 holdout tasks enter neither arm's
fit rows nor its model selection.

**Validation is pinned** with `--validation-corpus stack_pbe_v4`. Without it arm
B's validation set would be 37,352 rows, 30.3% world, and the two arms' best
validation CE would be different quantities.

## Tables

- [`paired.md`](paired.md), regenerated by `experiments/step3_report.py`: holdout
  CE, verified and graded solves, world eval, cross-entropy on six row sets.
- `epochs.md`, regenerated by `experiments/step3_epochs_report.py`: solves at
  matched epochs. <!-- pending: epoch study -->

## Files

- `v4only/`, `mixed/`: `policies/1m-s{0,1,2}/` (config, trainer state and training
  manifest; weights stay under `checkpoints/`), `timing/*.json` (every search cell,
  replayed with zero program mismatches, plus arm A's one depth-192 world cell),
  `runner.log`.
- `paired.md`, `paired.json`: the paired tables.
- `verified_solves.json`: the verdict and counterexamples for every solved program.
- `ce.json`: next-op cross-entropy for all six checkpoints on six row sets, and the
  leakage audit.
- `contaminated_world_eval.txt`, `drop_from_world_train.txt`, `lr_probe.json`.
- `runner/`: `run.sh`, `post.sh`, `structural.sh` and the combined log.

## Reproduce

```bash
bash reports/world-plus-v4-policy-20260912/runner/run.sh
bash reports/world-plus-v4-policy-20260912/runner/post.sh
uv run python experiments/step3_verify_solves.py \
  --run-dir checkpoints/step3-20260912/v4only --run-dir checkpoints/step3-20260912/mixed \
  --out reports/world-plus-v4-policy-20260912/verified_solves.json
uv run python experiments/step3_report.py \
  --baseline checkpoints/step3-20260912/v4only --mixed checkpoints/step3-20260912/mixed \
  --world-exclude reports/world-plus-v4-policy-20260912/contaminated_world_eval.txt \
  --verified reports/world-plus-v4-policy-20260912/verified_solves.json \
  --ce reports/world-plus-v4-policy-20260912/ce.json \
  --out reports/world-plus-v4-policy-20260912
```

The runners expect `lr_probe.json` and the drop list under
`checkpoints/step3-20260912/{v4only,mixed}/`, as `run.sh` was launched.
