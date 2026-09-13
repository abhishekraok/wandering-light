# world-plus-v4-policy (#134): does adding the world corpus beat a matched v4-only baseline?

Step 3 of training-data-sources (#118). The #101 next-op policy, arm `1m`
(655,836 parameters), trained twice under one recipe: arm A on `stack_pbe_v4`
alone, arm B on `stack_pbe_v4` plus `stack_pbe_world_v3`. Seeds 0, 1 and 2, six
epochs, Adam at 1.25e-4, batch 256, CUDA. Both arms choose their best epoch on the
**same** v4 holdout, 26,040 rows, and are searched on #101's 300-task v4 cohort and
on world-v3's eval split with beam 64 × depth 48.

## Findings (written 2026-09-13; the tables below regenerate)

- **World data lowers next-op cross-entropy on v4's own holdout, on every seed.**
  0.9541 ± 0.0100 for v4-only, 0.8576 ± 0.0130 for v4+world; the paired delta is
  −0.0965 ± 0.0174 (−0.1085, −0.0765, −0.1044), about ten times the baseline's
  seed spread. The holdout rows are identical in both arms, and no world training
  behaviour shares a program or a skeleton with any of its 16 behaviours
  (`ce.json`, `leakage`).
- **Extra gradient steps do not explain it.** Arm B takes 1,677 batches per
  epoch against arm A's 1,225 and reaches its best epoch later (2, 4, 3 against
  1, 2, 1). But arm A's validation cross-entropy rises monotonically after its best
  epoch on every seed, to 1.36–1.51 by epoch 6, so no longer schedule in this
  recipe improves arm A. Arm B is below arm A's best at 3, 5 and 6 of its 6
  epochs. At the **same** epoch, arm B's training cross-entropy is higher than
  arm A's in all 18 epoch-seed pairs while its validation cross-entropy is lower:
  it memorises more slowly and generalises better. That is #101's "the corpus
  binds" tested directly rather than inferred from a flat size axis.
- **Verified solves on the v4 cohort rise from 52.3 to 86.0 of 300 on average.**
  Per seed 48, 63, 46 against 52, 104, 102: seed 0 barely moves (+4), seeds 1 and
  2 roughly double (+41, +56). A verified solve passes held-out grading **and**
  computes the behaviour: every distinct solved program is either a catalogue
  witness or agrees with the Python reference on 2,000 sampler draws and 2,000
  perturbed draws (`verified_solves.json`). The only programs that pass grading
  without computing the behaviour are the two already on record in `STATUS.md`
  (Superseded, 2026-09-09): `all_positive` at 28 ops in both arms, and
  `bounds_cover_list_flag` at 44 ops in arm B seed 1.
- **First verified solves past 20 ops for any `mark-28` policy.** Arm B solves
  1, 6 and 5 tasks in the 21–40 bucket, each one the exact catalogue witness of
  `shared_value_occurrences` (31 ops, 11 tasks) or `count_in_range` (40 ops, 1
  task). Arm A, and #101's arms at all three model sizes, solved none there. Arm
  B's one 41+ solve is the `bounds_cover_list_flag` shortcut.
- **The gain is in the short buckets, and one behaviour regresses.** Verified,
  1–5 goes from 32, 41, 31 to 32, 60, 51 and 6–10 from 0, 6, 0 to 8, 28, 33. The
  11–20 bucket falls from 16, 16, 15 to 11, 10, 13, and all of that is
  `drop_last`: 9, 9, 8 of its 10 tasks in arm A against 4, 3, 6 in arm B.
  `min_pair` is 7 of 7 in both.
- **On world eval, arm B predicts the next op far better and neither arm can
  search it.** On the 44 clean world eval behaviours, next-op cross-entropy is
  3.16 ± 0.61 for arm A and 1.02 ± 0.05 for arm B. Much of that is familiarity with
  the world corpus's authoring idioms, which arm A never saw, so it is transfer to
  unseen behaviours inside that corpus, not evidence of general capability. Solve
  rate at depth 48 is 1, 0, 0 of 44 in both arms: only 4 of the 44 witnesses fit
  inside the depth cap (median 149.5 ops).
- **World-v3's eval split was contaminated, in the baseline's favour.** 11 of its
  55 behaviours share a skeleton with a world training behaviour, because v3 split
  by behaviour-id hash rather than by skeleton; 9 share the program too, and 3 of
  the 11 also match v4 train (`mbpp-62` and `mbpp-924` verbatim, `mbpp-458` by
  skeleton). Those 11 carried 2, 2 and 3 of arm A's world eval solves. They are
  removed from `stack_pbe_world_v3` <!-- pending: revision -->, with the train split
  byte-identical, so both arms remain valid against the new revision.
- <!-- pending: structural depth-192 finding -->
- **Scope of the claim, stated rather than fixed.** World supplies 26.9% of arm
  B's fit rows from 6.9% of its fit tasks, 49.4% of those rows come from the 54
  world fit behaviours longer than 500 ops, and 94.8% from witnesses longer than
  100 ops, so this measures adding long translated witnesses. The tokeniser keeps
  the top 160 tokens of a stack (`STATE_CAP`), which truncates 42.7% of world
  training example stacks and none of v4's; 43.7% of world fit rows carry at least
  one truncated stack. Raising the cap would change the checkpoint format and
  break comparability with #101, so this run inherits it. Three paired seeds;
  seed 0's solve-rate gain is inside arm A's own seed spread.
- **The baseline reproduces #101.** Arm A seeds 0 and 1 reach #101's recorded best
  validation cross-entropy (0.9519, 0.9454), and their v4 timing cells match #101's
  recorded cells row by row: task order, solved set and every program.

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
identity against v4 eval finds the same three; literal input-output identity finds
none, because v4 draws fresh inputs. World's own 46 holdout tasks enter neither
arm's fit rows nor its model selection.

**Validation is pinned** with `--validation-corpus stack_pbe_v4`. Without it arm
B's validation set would be 37,352 rows, 30.3% world, and the two arms' best
validation cross-entropy would be different quantities.

## Tables

See [`paired.md`](paired.md), regenerated by `experiments/step3_report.py`.

## Files

- `v4only/`, `mixed/`: `policies/1m-s{0,1,2}/` (config, trainer state and training
  manifest; weights stay under `checkpoints/`), `timing/*.json` (every search cell,
  replayed with zero program mismatches), `runner.log`.
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
