# keep-below-ablation (#136): does the #134 gain survive removing the 156 world programs that contain `keep_below`'s witness?

Follow-on to [world-plus-v4-policy (#134)](../world-plus-v4-policy-20260912/README.md), which found that adding `stack_pbe_world_v3` lowers v4-holdout CE on every seed but that its search gain is one behaviour, `keep_below`, whose 6-op witness `[ lt? ] swap cons filter` occurs whole in 156 of the 522 world fit programs. This run retrains #134's v4 + world arm with those 156 behaviours dropped (arm `ablated`, seeds 0 / 1 / 2) and, as a corpus-size control, with 156 world train behaviours dropped at random (arm `random`, seed 0; the draw keeps 114 of the carriers, so it controls size, not `keep_below`). Same recipe: 1m, 6 epochs, snapshots at every epoch, validation pinned to the v4 holdout, every epoch searched on the 300-task v4 cohort at beam 64 × depth 48, every graded solve checked at 10,000 draws per mode. #134's arms are read from its epoch study, unchanged.

## Answer, 2026-09-14

- **The search gain was the copied chunk.** `keep_below` (19 cohort tasks) is solved 186 times over v4 + world's 18 cells, 7 times over the ablated arm's, 0 over v4-only's; all 7 are the exact witness, assembled from pieces. On the other 42 behaviours the three arms are level: 1,301 / 1,256 / 1,281 verified solves over the 18 cells (v4 only / v4 + world / ablated), and the ablated arm's per-seed peaks (80 / 100 / 83, mean 87.7) match v4-only's (87.0), against v4 + world's 105.0.
- **The loss gain was not.** Best holdout CE: ablated 0.8710 ± 0.0050 against v4 + world 0.8576 ± 0.0130 and v4 only 0.9541 ± 0.0100; per seed +0.028 / −0.003 / +0.015 against v4 + world, so 86% of the −0.097 survives losing 30% of the world programs (71k of 429k fit rows). The random drop of 156 (40k rows, one seed) keeps only 35% of that seed's gain: 0.9135 against 0.8434. Losing the `keep_below` carriers costs the loss less than losing average world programs.
- **A new longest verified solve, on the drop arms only:** a 46-op `bounds_cover_list_flag` program, half its 92-op catalogue witness, with 0 failures in 28,171 draws (ablated seed 0 epochs 5–6; random seed 0 epoch 4 on 4 of the behaviour's 8 tasks, and epoch 6). Neither #134 arm found it in any cell.

## Scope

- One seed for the random arm, and it retains 114 carriers; a keep_below-free random control would need a draw from the 363 non-carriers.
- The other-42 comparison rests on the 16 behaviours some arm ever solves; 26 of the 43 are never solved by any arm. The ablated arm verifies 17 behaviours, the most of any arm, three of them (`bounds_cover_list_flag`, `n_minus_each`, `sum_negative`) never verified by a #134 arm.
- Peaks are chosen on the eval cohort (upper bounds); matched-epoch and epoch-mean numbers are in `ablation.md`.

## Files

- `ablation.md` / `ablation.json`: CE by epoch, verified solves by epoch with `keep_below` split out, paired differences against v4 + world (`experiments/step3_ablation_report.py`).
- `verified_solves_ablation.json`: the 10k-draw check over every solved program in #134's four run dirs and the two here (45 programs, 35 genuine, 10 shortcuts, 0 inconclusive).
- `drop_ablated.txt`, `drop_random.txt`: the drop lists (the three #134 drops plus 156).
- `epochs/{ablated,random}/`: trainer states, manifests and search cells per epoch, no weights (weights in `checkpoints/step3-20260913-ablation/`, gitignored).
- `runner/`: `study.sh` (train and search), `resume.sh` (replay after a session restart killed `study.sh` mid-replay; the 24 cells were complete), `post.sh` (verify and tables), logs.

## Reproduce

```bash
bash checkpoints/step3-20260913-ablation/study.sh   # copies of runner/study.sh, post.sh
bash checkpoints/step3-20260913-ablation/post.sh
```

The drop lists are built from #134's `containment.json` definition: `drop_ablated.txt` lists the world fit behaviours whose program contains `keep_below`'s witness as whole ops; `drop_random.txt` is `random.Random(136).sample` of the 519 world train behaviours.
