# novelty-reduction (#171): M2 and M2b report JSONs

These are copies of the run reports. The full run dirs (policies, pools, rows, models) are in the main checkout under `checkpoints/novelty-reduction/`, which is not tracked.

- `m2-s0/cycle-{0..3}.json`: the M2 loop, seed 0, commit `b56ded2` (`experiments/novelty_reduction/cycle.py`).
- `m2b-s{0,1,2}/report.json`: M2b ridge probes on the frozen `m2-s0` cycle-3 policy, commit `0deebce` (`experiments/novelty_reduction/m2b.py`). The command is `m2b.py --run-dir <dir> --source checkpoints/novelty-reduction/m2-s0 --rnd-dump checkpoints/novelty-reduction/m2-s0-replay/cycle-3/dump.pkl.gz --seed <s>`. The RND dump comes from `replay.py --source m2-s0 --out m2-s0-replay`, which reproduces m2-s0 exactly.

The numbers are in [RESULTS_DETAILED](../../docs/RESULTS_DETAILED.md), and the argument is on [#171](https://github.com/abhishekraok/wandering-light/issues/171).
