# staged-witness-reauthoring (#155): four training arms on the staged corpus

`experiments/staged_arms.py`, run dir `checkpoints/staged-arms-20260920` (UltraDesk, RTX 4090, CUDA). Tables:
[`graded.md`](graded.md), [`verified.md`](verified.md); verdicts per distinct solved program in
`verified_solves.json` (`experiments/step3_verify_solves.py`, 10,000 draws in each of three modes).

**Arms.** The #101 1m policy (655,836 params), LR 1.25e-4, batch 256, seeds 0 / 1 / 2, **epoch 3 fixed in
advance**, validation = the fused holdout, cohort = 300 eval tasks / 43 behaviours, beam 64 x depth 48, guard
`both`. Arms differ only in the program each of the **123 treated fit behaviours** (4,920 of 6,360 fit tasks)
is replayed along: `fused` the v4 witness, `refused` the re-fused twin, `staged` the staged witness, `both`
one of fused / staged per draw. The 6 rows flagged contrived keep their fused witness. Every arm trains on
313,560 rows per epoch drawn task-, then witness-, then row-uniform (one draw per arm, shared by its seeds:
the seed spread is initialisation and shuffle noise). An independent Opus review found the driver valid.

## Result (2026-09-20)

| arm | graded tasks /300 | verified tasks | **verified behaviours /43** | union over seeds |
|---|---|---|---|---:|
| fused | 86.0 (88 / 81 / 89) | 83.7 (86 / 79 / 86) | **12.33** (13 / 11 / 13) | 15 |
| refused | 90.3 (86 / 90 / 95) | 87.0 (83 / 86 / 92) | **12.67** (12 / 13 / 13) | 15 |
| staged | 139.7 (143 / 128 / 148) | 137.7 (139 / 127 / 147) | **15.67** (16 / 16 / 15) | 16 |
| both | 130.3 (121 / 147 / 123) | 128.0 (120 / 145 / 119) | **14.00** (14 / 15 / 13) | 15 |

Verified behaviours by witness-length bucket, mean of seeds (fused / refused / staged / both): 1-5
6.00 / 4.67 / 6.00 / 6.00; 6-10 1.00 / 1.33 / **3.00** / 3.00; 11-20 3.67 / 4.00 / **5.00** / 4.67; 21-40
1.33 / 2.33 / 1.67 / 0.33; 41+ 0.33 / 0.33 / 0.00 / 0.00.

- **Staged text raises verified behaviours by 3.3 per seed (12.33 -> 15.67), and the shape does it, not the
  new algorithms.** The re-fused arm has the staged arm's algorithms with the cycles merged back and is level
  with fused (12.67). It is a partial control (18 of the 137 treated twins are themselves within the staging
  rules, 6 of them identical to the staged program), a dose too small to explain a 3-behaviour gap. Every staged seed is above every fused seed.
- **The gain is retrieval, not composition.** Splitting each arm's verified behaviours by whether the solving
  program occurs **verbatim inside that arm's own training text**:

  | arm | verified behaviours | by a program found nowhere in its train text | only ever by a contained program |
  |---|---|---|---|
  | fused | 12.33 | **8.33** (9 / 7 / 9) | 4.00 |
  | refused | 12.67 | 7.33 | 5.33 |
  | staged | 15.67 | **4.33** (4 / 6 / 3) | 11.33 |
  | both | 14.00 | 1.00 | 13.00 |

  Staging fills the training text with short cycles that are complete programs for small behaviours:
  `[ lt? ] swap cons filter` is `keep_below` and sits inside 7 staged train witnesses and no fused one;
  likewise `n_minus_each`, `negate_each`, `min_pair`, `drop_last`, `capped_sum`. The staged policy retrieves
  them (the 6-10 bucket goes from 1.7 to 49.7 graded tasks of 60). Behaviours solved by a program that had to
  be *assembled* fall from 8.3 to 4.3 per seed. The split is not an artefact of tiny programs such as
  `quote cat` being contained anywhere: restricted to solving programs of 6+ ops, not-in-text goes
  6.33 -> 4.33 and only-contained goes **0.00 -> 5.67** (the fused arm never verifies a 6+-op behaviour
  with a contained program, in any seed); and `both` scored against the staged text alone is still 1.00. That is the containment mechanism of keep-below-ablation
  (#136), now produced by the corpus's shape rather than by a copied world witness. It is also what staged
  data is for, a library written into the text; but it is not evidence that the policy composes cycles.
- **One behaviour is gained by composition**: `largest_inside_interval` (40-op witness, the 21-40 bucket),
  verified at 23 ops (one filter cycle and a max fold) and at 41, contained in no training text. `count_in_range` and
  `all_positive` are also assembled, and the fused arm assembles them too.
- **Two behaviours are lost**, both long fused-style programs the staged policy no longer writes:
  `shared_value_occurrences` (31 ops) and `farther_choice_from_target` (32 ops). Union over seeds is 15 -> 16:
  +3 (`keep_below`, `n_minus_each`, `largest_inside_interval`), -2.
- **Past 20 ops nothing moved.** 21-40: 1.33 -> 1.67 behaviours; 41+: 0.33 -> 0.00. The graded 21-40 count is
  *lower* for staged (4.7 tasks against 6.0; re-fused is best at 9.7).
- **`both` is the only arm whose validation CE does not rise** (seed mean 1.018 -> 0.987 over epochs 1-3; one
  seed ends 0.004 above its start; fused 0.96 -> 1.12) and it searches worse than staged. Validation CE misleads here again, as in #154.
- **Shortcuts** are unchanged in kind: `aligned_matches`, `all_positive`, `bounds_cover_list_flag` (leaky
  samplers, #141 and STATUS) plus shortcut programs for `count_in_range`, `largest_inside_interval` (two),
  `nondecreasing_indicator`, `prefix_sums`. Three of the four multi-cycle solves the first staged seed
  *graded* were shortcuts; graded counts flatter every arm by 2-3 tasks.

## Against the registered predictions

Prediction 1 (staged ties or loses under a token-level beam, fewer past 20 ops) is **wrong overall and a tie
past 20 ops**: its premise, that staged witnesses are longer, was already false. What it got right is the
part that matters for the length wall: staging the text did not carry the beam past 20 ops.

## A search pathology, filed separately

One staged seed's beam walks into `dup cat dup cat ...`, doubling a list for 2 fuel until states hold millions
of atoms (task 1594: about 2.0M atoms at depth 41; the fused policy finishes the same task in 2.0 s). The
beam has no state-size bound. Tasks are abandoned as unsolved after **10 s** of wall clock (six times the
slowest normal search, 1.6 s over 2,100 searches) and recorded `wall_clock_cap`: 0 / 15 / 2 tasks in the
three staged seeds (17 searches, 15 distinct tasks), none in any other arm. 5 of the 15 tasks are ones some
fused seed solves (all `shared_value_occurrences`). The cap is not what lost that behaviour: no solved search
in any of the 12 cells took over 0.93 s, staged seed 0 capped nothing and solved none of them, and
`farther_choice_from_target` was never capped. beam-state-size-unbounded (#157).

## Not yet in

The per-cycle diagnostic (prediction 2, `experiments/staged_cycles.py`) and a `--keep-contrived` cell.
