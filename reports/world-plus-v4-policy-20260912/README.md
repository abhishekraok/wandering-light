# world-plus-v4-policy (#134): does adding the world corpus beat a matched v4-only baseline?

Step 3 of training-data-sources (#118). The #101 next-op policy, arm `1m`
(655,836 parameters), trained twice under one recipe: arm A on `stack_pbe_v4`
alone, arm B on `stack_pbe_v4` plus `stack_pbe_world_v3`. Seeds 0, 1 and 2, six
epochs, Adam at 1.25e-4, batch 256, CUDA. Both arms choose their best epoch on the
**same** v4 holdout, 26,040 rows, and are searched on #101's 300-task v4 cohort and
on world-v3's eval split with beam 64 × depth 48.

Both arms were also retrained with weights saved at every epoch (each seed's snapshot
at its best epoch is bitwise equal to the original best checkpoint) and searched on
the v4 cohort at epochs 1–6.

**Answer, 2026-09-13.** World data lowers next-op cross-entropy on v4's holdout on
every seed. Its measured search gain is one behaviour: `keep_below`, whose 6-op
witness occurs inside 156 world fit programs and no v4 fit program, gains 186
verified solves summed over seeds and epochs 1–6, more than the cohort's net 141,
and v4+world solves it only with that exact witness. The other 42 behaviours net
−45. That shows no transfer beyond a contained witness without ruling it out: 29 of
the 43 cohort behaviours are never solved by either arm and 4 always are. The
best-checkpoint comparison that first suggested a broad gain (verified 48, 63, 46
against 52, 104, 102 in [`paired.md`](paired.md)) compared checkpoints taken at
different epochs.

## Findings (2026-09-13; the tables below regenerate)

- **Solves track the epoch more than the data.** Validation CE selects epoch 1 or 2
  for every v4-only seed, but its verified solves peak later, at 94, 77 and 90
  (epochs 4, 3 and 5), by which point its validation CE has risen to 1.06–1.29. For
  v4+world, CE selection lands on the search peak for seed 2 (102), 5 short for seed
  1 (104 against 109) and far short for seed 0 (52 against 104). Searched at every
  epoch, v4+world minus v4-only is +1.0, +9.0, +13.0, +9.0, +17.7 and −2.7 verified
  solves per seed at epochs 1–6, and single seeds range from −13 to +32. Arm B takes
  1.37 times arm A's steps per epoch; pairing its epochs 1–4 with the v4-only epoch
  nearest in steps (1, 3, 4 and 5) gives +1.0, −14.3, +8.0 and +11.0, and its epochs
  5–6 have no v4-only epoch within half an epoch. So the sign depends on the
  comparison: epochs 2–5 favour v4+world, epoch 6 does not, and the step-matched
  pairs average +1.4. Each seed's best epoch, chosen on the eval cohort itself and so
  an upper bound for both arms, gives 87.0 against 105.0; the mean over the six
  epochs gives 72.3 against 80.1 ([`epochs.md`](epochs.md)).
- **The gain is one behaviour whose witness world programs contain and v4 programs
  do not.** Five cohort behaviours have an eval witness that occurs, as whole ops,
  inside world fit programs and no v4 fit program: `keep_below` (6 ops, inside 156
  world fit programs), `increment_each` (5 ops, 44), `negate_each` (6, 3),
  `n_minus_each` (7, 2) and `drop_last` (12, 1). Summed over seeds they gain +12,
  +32, +54, +26, +50 and +9 verified solves at epochs 1–6, while the other 38
  behaviours go −9, −5, −15, +1, +3 and −17 ([`containment.md`](containment.md)).
  Almost all of it is `keep_below`: +186 over the six epochs, never solved by v4-only,
  and every v4+world solve is the exact witness `[ lt? ] swap cons filter`. The
  other four contained behaviours net −3 (`negate_each` +25, `increment_each` −31,
  `drop_last` +3, `n_minus_each` never solved). On `negate_each` v4+world switches
  to the world-contained `[ 0 swap sub ] map` (125 of 126 solves), where v4-only
  mostly used `[ dup 1 1 add mul sub ] map`. v4 fit programs contain `swap cons
  filter` in 12 behaviours, 9 of them as `[ eq? ] swap cons filter`, but none
  contains `lt? ] swap cons filter`. The other 38's −42 is mostly `keep_negative`
  (−36), whose witness v4 and world fit programs both contain.
- **The long solves do not need world data.** From epoch 3 on, all three v4-only
  seeds verify `shared_value_occurrences`' 31-op catalogue witness, and v4-only also
  verifies `count_in_range`'s 40-op witness and a 32-op program for
  `farther_choice_from_target`, whose witness is 54 ops. v4+world solves the same
  three. At the best-validation checkpoints only v4+world reached the 21–40 bucket:
  11 of its 12 such solves come from seeds taken at epochs 3 and 4, and no v4-only
  checkpoint was taken past epoch 2. #101's recorded 12-op ceiling was not right even
  for its own checkpoints: its 10m seed-1 epoch-2 checkpoint verifies a 22-op
  `capped_sum` (`docs/STATUS.md`, Superseded).
- **What verified means.** A verified solve passes held-out grading and is checked
  against the behaviour: its program equals a catalogue witness, or it agrees with
  the Python reference on 10,000 draws in each of three modes (the behaviour's own
  sampler; random values in −20..20 with lengths 0–10; random values in −2..2 with
  lengths 0–10), each counted only where the catalogue witness computes the
  reference, with at least 500 per mode. The 36 searched cells solved 36 distinct
  programs: 29 verify, 7 are shortcuts that also fail on their behaviour's own
  sampler, and none is undecided
  ([`verified_solves_epochs.json`](verified_solves_epochs.json)). A pass is a finite
  test: a program wrong on 0.1% of inputs passes 10,000 draws in a mode with
  probability about 4.5e-5.
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
  epoch-seed pairs, but that is composition: at its best checkpoints arm B scores world fit rows at 2.5–3.5 times its v4 fit rows
  (length-weighted 3,000-row samples), and on all v4 fit rows it fits *more* tightly
  than arm A (0.388, 0.230, 0.298 against 0.548, 0.337, 0.555) while still scoring
  lower on the holdout. The learning rate is the lowest value #101's
  probe tried, so a lower rate for arm A was never tested.
- **World eval: the CE gap measures familiarity with world op sequences, and
  neither arm can search it.** On the 44 clean world eval behaviours arm A scores
  3.16 ± 0.61, at or worse than uniform over 28 ops (3.33; seed 1 is 3.86), and arm
  B 1.02 ± 0.05. An op 4-gram fit on world training programs, which never sees a
  stack, scores 0.91 there, and the audit's better-smoothed n-grams reached
  0.77–0.86, so the number does not show arm B inferring unseen behaviours. At the best-validation checkpoints (world eval was not searched per epoch), solve
  rate at depth 48 is 1, 0, 0 of 44 in both arms, on different tasks: arm A's is
  `mbpp-672`, v4 train's `max_pair` witness written twice, and arm B's is
  `mbpp-436`, v4 eval's `keep_negative` followed by `uncons swap drop`. Only 4 of
  the 44 witnesses fit within 48 ops (median 149.5). Arm A's one depth-192 cell
  solved the same 3 of 55 as at depth 48; arm B's was stopped to free the GPU.
- **World-v3's eval split had 11 contaminated behaviours, and on solves they
  favoured neither arm.** All 11 share a skeleton with a world training behaviour,
  because v3 split by behaviour-id hash rather than by skeleton; 9 share the program
  too, and 3 also match v4 train (`mbpp-62` and `mbpp-924` by program, `mbpp-458`
  by skeleton). Both arms' best-validation checkpoints solved 2, 2 and 3 of them. On CE they flattered arm B
  more: dirty minus clean is −0.43, −0.52, −0.45 for arm B against −0.21, −0.39,
  −0.21 for arm A. They are removed from `stack_pbe_world_v3`
  at Hub revision `3c05f57`, with the train split byte-identical. "Clean" means no
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
  as functions but proof that literal matching cannot detect overlap. The cohort has little room to show transfer: 29 of its 43 behaviours are never
  solved by either arm and 4 always are, and all three behaviours in the 6–10
  witness-length bucket are contained, so containment is confounded with bucket and
  headroom. The containment split was chosen after a best-checkpoint split pointed the
  other way. One model size (655,836 parameters) fits 632 behaviours in arm B against
  159 in arm A, arm B's learning rate was never probed, and CE on the cohort's own
  behaviours was not measured. Three paired seeds.
- **The baseline reproduces #101.** Arm A seeds 0 and 1 reach #101's recorded best
  validation CE (0.9519, 0.9454), their best-checkpoint v4 timing cells match #101's recorded cells row by row (task
  order, solved set and every program), and their weights are byte-identical to
  #101's W&B artifacts `mark28-policy-1m-s0:v0` and `-s1:v0` (sha256, checked
  2026-09-13). The checkpoints carry
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
  CE, verified and graded solves at the best-validation checkpoints, world eval,
  cross-entropy on six row sets.
- [`epochs.md`](epochs.md), regenerated by `experiments/step3_epochs_report.py`:
  solves at every epoch, at matched epochs and matched steps, and each seed's peak.
- [`containment.md`](containment.md), regenerated by
  `experiments/step3_containment.py`: the same solves split by whether world fit
  programs contain the eval witness.

## Files

- `v4only/`, `mixed/`: `policies/1m-s{0,1,2}/` (config, trainer state and training
  manifest; weights stay under `checkpoints/`), `timing/*.json` (every search cell,
  replayed with zero program mismatches, plus arm A's one depth-192 world cell),
  `runner.log`.
- `paired.md`, `paired.json`: the paired tables.
- `verified_solves_epochs.json`: the verdict and counterexamples for every program
  any cell solved, best checkpoints and epoch snapshots.
- `epochs.md`, `epochs.json`, `containment.md`, `containment.json`: the epoch
  study's tables.
- `epochs/v4only/`, `epochs/mixed/`: the epoch study's timing cells and per-epoch
  trainer states; weights stay under `checkpoints/`.
- `ce.json`: next-op cross-entropy for all six checkpoints on six row sets, and the
  leakage audit.
- `contaminated_world_eval.txt`, `drop_from_world_train.txt`, `lr_probe.json`.
- `runner/`: `run.sh`, `post.sh`, `structural.sh`, the epoch study's `study.sh`
  and `study_e56.sh`, and their logs.

## Reproduce

```bash
bash reports/world-plus-v4-policy-20260912/runner/run.sh
bash reports/world-plus-v4-policy-20260912/runner/post.sh
bash reports/world-plus-v4-policy-20260912/runner/study.sh      # epochs 1-4
bash reports/world-plus-v4-policy-20260912/runner/study_e56.sh  # epochs 5-6, verify, epochs.md
uv run python experiments/step3_report.py \
  --baseline checkpoints/step3-20260912/v4only --mixed checkpoints/step3-20260912/mixed \
  --world-exclude reports/world-plus-v4-policy-20260912/contaminated_world_eval.txt \
  --verified reports/world-plus-v4-policy-20260912/verified_solves_epochs.json \
  --ce reports/world-plus-v4-policy-20260912/ce.json \
  --out reports/world-plus-v4-policy-20260912
uv run python experiments/step3_containment.py \
  --baseline checkpoints/step3-20260913-epochs/v4only --mixed checkpoints/step3-20260913-epochs/mixed \
  --baseline-best checkpoints/step3-20260912/v4only --mixed-best checkpoints/step3-20260912/mixed \
  --verified reports/world-plus-v4-policy-20260912/verified_solves_epochs.json \
  --drop reports/world-plus-v4-policy-20260912/drop_from_world_train.txt \
  --out reports/world-plus-v4-policy-20260912
```

`run.sh` expects `lr_probe.json` and the drop list under
`checkpoints/step3-20260912/{v4only,mixed}/`; `study.sh` and `study_e56.sh` expect
`lr_probe.json` under `checkpoints/step3-20260913-epochs/{v4only,mixed}/` and the drop
list at `mixed/drop.txt` there (copies in `epochs/`).
