# Inner frames (#160), campaign 2: innermost-only frames, an exit-frame oracle, a commitment ranker

**Readings, as registered: oracle BETWEEN, ranker target met, stacking UNCLEAR.**

- **The ranker makes each `[`'s label cheap.** The label is the consumer, the `ifte` role and the capture count.
  - On v4 it charges a mean 0.37 nats per `[` against the shape-only prior's 1.24, and 1.3-1.5 per witness against 6.9.
  - The median, 0.025, is small because most labels follow from context.
  - A real choice still costs 1.5-3 nats, 35-75% of a search's budget: `map` against `filter`, a quotation with no consumer, or the hardest `ifte` guards.
  - With defined-body deficit removed, the world arm clears 33 of 45 under the learned charge, against 36 free and 23 under the prior. The ranker recovers 10 of the 13 clears the prior's charge cost.
  - It prices labels along correct prefixes only, not failed hole searches or sub-spec inference.
- **The policy makes little use of a sub-spec where the deficit is.** Shown the frame its body must end on, it pays 0.68 of the world arm's interior body deficit, but most of that is the current frame (0.81).
  - At the world arm's LR, the exit's own gain over the frame is 10% on v4 interior ops and none on world-v3.
  - On bodies of 7 or more ops, which hold 53% of v4's interior deficit and 84% of world-v3's, the exit adds nothing over the frame on v4.
  - At the last op before `]`, frame and exit determine the op uniquely in 236 of 265 v4 states. The policy still pays 0.53 of the world arm's deficit there.
- **Body cost is mostly exposure.** The world arm pays about as much per op in bodies as at the top level: 0.43 against 0.40 on v4. Bodies simply hold 76-78% of the ops.
  - Frames make body ops cheaper per op than top-level ones on v4: 0.32 under `frames_inner` and 0.23 under `exit_inner`, against 0.39-0.43 at the top level.
  - The lever is body length and verification.
- **Innermost-only against stacking is a world-v3 result.** There it cuts deep-level deficit to 0.88 of stacked. Its label helps at depth (0.89 of the world arm) where stacked labels hurt (1.17). On v4 the whole difference is one witness.
- **World-v3's floor is top-level op deficit.** With defined-body deficit removed and commitment free, 29 of its 44 witnesses still fail in a top-level segment.
- **Clears compare arms; they do not forecast coverage.** The solver verifies the same 12 of 43 cohort behaviours whether its beam ranks on raw or calibrated log-probabilities.
- **Next, the owner's call:** a perfect-decomposition search pilot with a stop rule set in advance, rescoring the solver's own programs, and an inverse-dynamics probe, before any solver is built (see What this licenses).

[Registration](https://github.com/abhishekraok/wandering-light/issues/160#issuecomment-5787411470) was posted before the campaign's training, after a one-epoch learning-rate probe it reports. Code on branch `claude/frame-campaign-2` at `c242efe`: `experiments/frame_rows.py`, `experiments/commit_ranker.py`, `experiments/frame_campaign2.py`.

## Setup

The policy arms use the world arm's recipe, as in the [first gate](../frame-gate-20260922/README.md):
- v4 train plus world-v3 train: 429,078 fit rows;
- the `1m` architecture, batch 256, six epochs, seeds 0-2, every epoch measured.

| arm | what the innermost defined open level shows | LR |
|---|---|---:|
| `tags_inner` | its label: consumer, `ifte` role, capture count | 1.25e-4 |
| `frames_inner` | label and the frame after the body prefix | 1.25e-4 |
| `exit_inner` | label, frame, and the frame the body must end on, from the authored witness | 2.5e-4 (probed) |
| `exit_inner_matched` | as `exit_inner` | 1.25e-4 |
| stacked `frames` | label and frame at *every* defined open level (the first gate's arm, retrained) | 1.25e-4 |

- **Ancestor levels.** In the inner arms they keep the machine's items.
- **The retrained stacked arm** reproduces the first gate's curves to the third decimal, and its frames/world of 0.862.
- **The ranker** is the same network with 41 commitment choices as its actions. It is trained on every `[` of the train witnesses, seen in `frames_inner`, for 30 epochs. Each seed picked epoch 3 on the even holdout tasks, and a temperature of 1.13-1.38 was fitted on the odd ones.

**Measure.** As in the first gate:
- calibrated deficit per op, on one witness per eval behaviour (45 on v4, 44 on world-v3);
- ratios of calibrated totals, as the median over 18 paired (seed, epoch) cells;
- clears at `s0-e5` under the prior's charge per `[`, the learned charge (ranker seed = policy seed), or free.

**Temperatures.** None sat at an edge of its search. Over the world arm's, median over cells:

| arm | temperature / world's |
|---|---:|
| `frames_inner` | 1.00 |
| `tags_inner` | 1.01 |
| `exit_inner` | 1.08 (0.99-1.17) |
| `exit_inner_matched` | 0.94 |

**What a clear predicts.** Clears count calibrated deficits, so they predict a beam ranking on the calibrated policy. The solver ranks on raw log-probabilities, which segment-composite (#160) counts: 9 flat, 14 with a budget per quotation, for the world arm on v4. Running both beams moves no coverage (see [What the clears predict](#what-the-clears-predict)), so read clears as comparisons between arms.

## Readings

| reading | predicted | measured | category |
|---|---:|---:|---|
| oracle: exit_inner/world, v4 interior defined ops | 0.70 (0.55-0.85) | **0.679** [0.59-0.78] | BETWEEN (< 0.5 with clears ≥ 18 for SKELETON_FIRST; > 0.8 for SPEC_NOT_THE_LEVER) |
| oracle clears at `s0-e5`, learned charge | 20 (17-23) | **20** | |
| same, at the world arm's LR (`exit_inner_matched`) | | 0.733, clears 17 | BETWEEN |
| ranker: median learned surprisal per `[`, v4, median of seeds | 0.30 (0.10-0.70) | **0.025** (0.015-0.029) | target (≤ 0.4) met |
| stacking: frames_inner/frames, v4, 2+ levels | 0.88 (0.78-1.00) | **0.840** [0.74-1.01] | ≤ 0.90, but |
| the control: same, 1 level (identical inputs) | 0.92-1.08 | **1.092** [0.98-1.27] | out of band: UNCLEAR |

## The ranker

| v4 eval | prior (shape only) | learned, seeds 0/1/2 |
|---|---:|---:|
| mean per `[` (371) | 1.242 | 0.368 / 0.373 / 0.389 |
| median per `[` | 1.022 | 0.015 / 0.029 / 0.025 |
| mean per first-of-group `[` (223) | 1.476 | 0.580 / 0.586 / 0.616 |
| median per first-of-group `[` | 1.138 | 0.034 / 0.064 / 0.052 |
| median per witness | 6.90 | 1.31 / 1.47 / 1.46 |
| mean per witness | 10.24 | 3.03 / 3.08 / 3.21 |
| world arm's clears at `s0-e5` (free: 18) | 12 | **18** |

- **World-v3.** The median per `[` is 0.64 under the prior and 0.08-0.13 learned. Per witness it is 17.5 under the prior and 8.4-9.6 learned.
- **The world arm's learned-charge clears** were predicted at 15 (13-17).
- **Other arms still lose clears to the learned charge.**
  - At `s0-e5`: `exit_inner` 20 against 22 free, stacked frames 19 against 22, `tags_inner` 19 against 21.
  - Over cells, the medians are 18-19 learned against 20-21.5 free.
  - The loss grows as bodies get cheaper (exchange table below).

**By choice.** Seed 0, v4, median nats per `[`, from Codex's audit:

| choice | opens | median nats |
|---|---:|---:|
| `ifte` roles 1-2 | 148 | 0.002 |
| `dip` | 76 | 0.017 |
| `fold` | 50 | 0.021 |
| `ifte` role 0 | 74 | 0.049 |
| `map` | 9 | 1.51 |
| `filter` | 10 | 1.70 |
| no consumer | 4 | 3.16 |

The tail is not only rare choices. Of the 136.6 nats charged, `map`, `filter` and no-consumer opens carry 34%, and `ifte` role 0 carries 35%.

**Three syntax baselines do not reach it.** Add-one-smoothed n-gram baselines fitted on the train split charge first-of-group opens a median of 0.35-0.44 nats:
- from the previous token;
- from the previous token and the enclosing tag;
- from the three previous tokens.

**Retrieval or generalisation** (`checks.py`)? No eval program occurs in train.

A skeleton is the witness's sequence of commitments. "Seen" means it occurs verbatim among the ranker's train-split witnesses, fit and holdout.

| per `[`, median / mean | seen skeleton | unseen skeleton |
|---|---:|---:|
| v4, learned: 22 of 43 witnesses with a `[`, 103 / 268 opens | 0.021-0.031 / 0.26-0.28 | 0.013-0.029 / 0.41-0.43 |
| v4, prior | 0.65 / 1.07 | 1.07 / 1.31 |
| world-v3, learned: 6 of 44, 39 / 1,124 opens | 0.04-0.06 / 0.20-0.28 | 0.08-0.14 / 0.49-0.51 |
| world-v3, prior | 0.38 / 0.60 | 0.68 / 1.01 |

- **v4.** The median is no higher on unseen skeletons. The mean is 1.5-1.6 times higher, but the prior, which retrieves nothing, already charges unseen opens 1.2 times more. So retrieval accounts for at most about 1.3 times.
- **World-v3.** The unseen median is 2-2.5 times the seen one, but only 6 witnesses are seen.

**Audit** (Codex): no leak and no bug found.
- **The state before the `[` holds nothing written at or after it.** Changing the quotation's consumer changes no encoded row: 1,680 example/open pairs, and eight real consumer substitutions, three of them nested. The enclosing level's label is shown. A search that commits at the `[` has it, because it committed that level earlier.
- **No v4 eval behaviour, program, or prompt occurs in the fit or holdout tasks.** The three world-v3 behaviours equal to a v4 eval behaviour are dropped.
- **Seed 0's charges reproduce** from the checkpoint to within 2e-6 nats.
- **Prior and learned charges are aligned** step for step in every cell.

**Charge per sub-search** (`checks.py`). Under decomposition, a `[` is paid by the search that writes it. The sub-searches are cut as the clears composite cuts them: the top level after each top-level combinator, and each body on its own. The budget per search is `ln 64` = 4.16 nats. Commitment charge alone, medians, with the share over budget in brackets:

| learned charge, seeds 0-2 | v4 | world-v3 |
|---|---:|---:|
| costliest top-level segment, over witnesses | 0.9-1.2 (0-5%) | 1.9-2.1 (7-18%) |
| bodies with a `[` inside, over bodies (112 / 293) | 0.05-0.09 (3-4%) | 0.22-0.36 (9%) |
| costliest sub-search, over witnesses | 1.3-1.4 (7-9%; prior 35%) | 3.0-3.4 (39-41%; prior 55%) |
| the whole top level as one search, over witnesses | 0.9-1.2 (12-14%) | 3.6-4.1 (41-48%) |

- **On v4 the skeleton decomposes with the search.**
- **On world-v3, commitment alone breaks the budget** in 39-41% of witnesses. It does so in 7-18% at the top level; the rest is witnesses with many nested bodies.
- **Commitment is not what holds world-v3 at its floor, though** (`checks.py`, world arm at `s0-e5`, defined-body deficit removed):

| | v4, free | v4, learned | world-v3, free | world-v3, learned |
|---|---:|---:|---:|---:|
| fails in a top-level segment | 7 | 10 | 29 | 32 |
| fails inside a body | 5 | 6 | 15 | 18 |
| clears | 36 | 33 | 11 | 9 |

  World-v3's floor is top-level op deficit, and body ops outside defined bodies. The learned charge adds three failures.

**Exchange table under the learned charge** (`checks.py`, seed 0). Clears at `s0-e5` as calibrated defined-body deficit is scaled, with free commitment in brackets.

| body deficit scale | 1.0 | 0.75 | 0.5 | 0.25 | 0 |
|---|---:|---:|---:|---:|---:|
| v4, world arm | 18 (18) | 22 (23) | 26 (27) | 33 (34) | 33 (36) |
| v4, `exit_inner` | 20 (22) | 25 (26) | 30 (33) | 32 (34) | 33 (35) |
| world-v3, world arm | 2 (2) | 2 (2) | 2 (3) | 5 (8) | 9 (11) |

- **This is the ceiling** for a decomposing search on this predictor, with defined-body deficit removed: 33 of 45 on v4 and 9 of 44 on world-v3. Re-encoded and untouched body ops stay charged.
- **On v4, halving body cost is worth 8 clears.**

## The oracle

Median of 18 cells, v4 defined-body ops, against the prediction:

| | predicted | exit_inner/world | exit_inner/frames_inner | same LR: matched/frames_inner | frames_inner/world |
|---|---:|---:|---:|---:|---:|
| first op | 0.85 | 0.714 | 0.864 | 0.875 | 0.819 |
| later ops | 0.62 | 0.666 | 0.847 | 0.915 | 0.808 |
| last op | 0.20 | **0.515** [0.43-0.75] | 0.745 | 0.827 | 0.719 |
| `]` | 0.10 | 0.111 | 0.234 | 0.245 | 0.476 |
| **interior** (first and later) | 0.70 | **0.679** [0.59-0.78] | 0.839 | 0.898 | 0.811 |
| all defined | 0.52 | 0.598 | 0.794 | 0.840 | 0.753 |

`exit_inner` trains at 2.5e-4, `frames_inner` at 1.25e-4. The matched column compares the exit at the frame arm's own rate.

Interior ops by body length, and the share of the world arm's interior deficit each length holds (`checks.py`):

| body | v4 share | v4 exit/world | v4 exit/frames_inner | v4 matched/frames_inner | world-v3 share | world-v3 exit/world | world-v3 exit/frames_inner | world-v3 matched/frames_inner |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-2 ops | 9% | 0.653 | 0.781 | 0.816 | 3% | 0.626 | 0.920 | 0.872 |
| 3-6 | 38% | 0.605 | **0.648** | 0.723 | 13% | 0.697 | **0.752** | 0.811 |
| 7+ | 53% | 0.716 | **1.026** | 1.069 | 84% | 0.818 | **0.939** | 1.076 |

- **The close.** The policy learns to write `]` once the frame matches the exit: 0.11 of the world arm's deficit.
- **The last op.** Most of its drop comes from the frame (0.72), not the exit (0.745 over `frames_inner`). In 236 of the 265 v4 last-op states, the witness op is the only op that maps every example's frame to its exit (`checks.py`). There the oracle still pays **0.531** [0.45-0.76] of the world arm's deficit, against `frames_inner`'s 0.708. The remaining cost is not ambiguity between equivalent ops.
- **Long bodies.** On v4 bodies of 7 or more ops, the exit adds nothing over the current frame: 1.03, lower in only 6 of 18 cells.
  - On world-v3 it adds 6% (0.94, lower in 15 of 18 cells). That is the largest saving in nats, because 7+ bodies hold 84% of world-v3's interior deficit.
  - At the same LR as the frame arm, the exit adds nothing on long bodies in either corpus: 1.07 and 1.08.
- **Top-1 at the witness op**, v4 defined ops: 87% under `exit_inner`, 83% `frames_inner`, 82% stacked, 79% world.
- **Clears do not separate the arms.** Under the learned charge, medians [range] over 18 cells:
  - `exit_inner` 19 [13-22], `exit_inner_matched` 19 [11-23];
  - stacked frames 18 [12-21], `frames_inner` 18 [11-21], world 18 [9-23];
  - `tags_inner` 16.5 [9-21].
  At epoch 5 by seed, world clears 18 / 21 / 17 and `exit_inner` 20 / 18 / 19 (`checks.py`).

**One explanation, not tested here.** The encoder max-pools per-token features and never aligns the frame with the exit item by item. That would stop it comparing where it is with where it must go. It would explain both results: a uniquely determined last op it still misses, and a distant target it cannot use.

## Innermost-only

Calibrated deficit over the world arm's, median of 18:

| | v4 frames_inner | v4 stacked | world-v3 frames_inner | world-v3 stacked |
|---|---:|---:|---:|---:|
| defined body ops (predicted 0.80 / 0.84) | 0.753 | 0.862 | **0.798** | 0.867 |
| same, without `bounds_cover_list_flag` | 0.827 | 0.818 | (v4 only) | |
| 1 open level (0.62 / 0.84) | 0.690 | 0.623 | 0.824 | 0.842 |
| 2 or more (0.82 / 0.86) | 0.769 | 0.929 | 0.800 | 0.893 |
| 2 or more, without `bounds_cover_list_flag` | 0.857 | 0.866 | (v4 only) | |
| top-level ops (0.90 / 0.93) | 0.937 | 0.876 | 0.928 | 0.922 |

Paired frames_inner/stacked frames, median of 18 (`checks.py`):

| | v4 | world-v3 |
|---|---:|---:|
| 1 open level, identical inputs: the control | 1.092 (17 of 18 cells above 1) | 0.979 |
| same, without `bounds_cover_list_flag` | 1.059 (15 of 18) | |
| top-level ops, also identical inputs | 1.062 (12 of 18) | |
| 2 or more levels | 0.840 | **0.881** (highest with one witness left out: 0.901) |
| 2 or more, without `bounds_cover_list_flag` | 1.000 | |

- **v4.** The innermost arm pays a small systematic penalty on states whose inputs are identical to stacked's: about 6% at one level without the outlier, and 6% at the top level.
  - Its deep gain is one witness, the first gate's outlier. Without it, depth reads 1.000, DEPTH_IS_INTRINSIC's pattern, and the control reads 1.059, inside its band.
  - The coded reading is UNCLEAR, and the data without the outlier leans to depth being intrinsic.
- **World-v3** fits STACKING_WAS_A_COST: 0.881 at depth with the control at 0.979, 17 of 18 cells below 1. It clears the 0.90 threshold by only 0.02. The registered reading is on v4.
- **The label**, `tags_inner`/world at 2 or more levels:
  - World-v3, unclipped states: **0.892**, against 1.165 stacked and a predicted 1.03, with 17 of 18 cells below 1.
  - v4: 0.964 against 1.147 stacked, but without `bounds_cover_list_flag` both read 1.014.
  - Innermost-only removes the label's cost on world-v3.
- **Clears at `s0-e5`, prior's charge:** `frames_inner` 13 (predicted 14), world-v3 2 (predicted 2). By clears, `tags_inner` is the weakest arm: median 16.5 under the learned charge.

## Where body cost comes from

The world arm pays about the same per op in bodies as at the top level. Calibrated deficit per op at `s0-e5`, with the op count:

| | v4 defined body | v4 top level | world-v3 defined body | world-v3 top level |
|---|---:|---:|---:|---:|
| world | 0.426 (1,360) | 0.404 (426) | 0.377 (6,325) | 0.510 (1,813) |
| `frames_inner` | 0.321 | 0.425 | 0.317 | 0.494 |
| `exit_inner` | 0.233 | 0.392 | 0.243 | 0.444 |

- **Bodies dominate the deficit because they are long**, not because they are harder per op.
- **Frames make a body op cheaper than a top-level op**, and the exit more so. This is the campaign's positive result.
- **Where it points:** shorter bodies (splitting, macros), and executing the body to check it against a sub-spec rather than predicting its end.

## What the clears predict

`tempered_beam.py` runs `BEST`'s search (beam 64 x depth 48, guard `both`, library on) with the world arm's `1m-s0-e5` on the #101 cohort of 300 tasks and 43 behaviours. It runs once on raw log-probabilities and once on logits divided by 1.564, the checkpoint's fitted temperature. Predicted before the run: within one verified behaviour.

| ranking on | tasks /300 | graded /43 | **verified /43** | clock |
|---|---:|---:|---:|---:|
| raw log-probabilities (T = 1) | 131 | 14 | **12** | 443 s |
| calibrated (T = 1.564) | 126 | 14 | **12** | 464 s |

- **Calibration does not move coverage.** The same 12 behaviours verify at both temperatures.
- **So clears are comparisons between arms on one predictor, not coverage forecasts.** On the cohort's behaviours, the raw composite gives the world arm 9 flat clears (segment-composite); the beam verifies 12.
- **The world arm verifies fewer behaviours than `BEST`** on this cohort: 12 against 16. A search pilot's flat baseline should be run with the checkpoint the pilot uses.

## What this licenses

**Not the solver build yet.** A Codex design review of skeleton-first-search (#163) and branch-and-check (#164), commissioned by the advisor session, [posted on #163](https://github.com/abhishekraok/wandering-light/issues/163#issuecomment-5792202085), finds that deriving a body's sub-spec is the central algorithm, not plumbing:
- **Few bodies have a sub-spec derivable from the answer.** Only 12% of v4 eval quotations, and 5.5% of world-v3's, are terminal outer bodies. Most of those are `ifte` and `fold`.
  - For `fold`, the intermediate accumulators are unknown.
  - For `ifte`, which examples take which branch depends on the unknown predicate.
- **The ranker does not price the rest of the search:** failed hole searches, suffix inversion, or backtracking across holes.

Campaign 2 adds three findings:
- **The label is cheap** on v4.
- **The policy does not exploit a sub-spec**, even one step from the exit.
- **Body cost is exposure.**

**Proposed next, the owner's call** (the advisor session's recommendation, from Codex's review):
1. **A perfect-decomposition search pilot** on the 45 v4 eval representatives.
   - **Supplied:** the correct skeleton and each hole's entry and exit constraints from the witness, including `fold` transitions and `ifte` active-example masks.
   - **Search:** holes solved bottom-up with the campaign's encoding, under one task budget matched to a flat beam-64/depth-48 run's primitive applications.
   - **Graded** on untouched hidden examples.
   - **Stop rule, set now:** at most +1 verified behaviour over the matched flat baseline stops the skeleton-first build at this budget; +6 or more licenses a sub-spec derivation pilot; 2-5 is inconclusive.
2. **Rescore the solver's own verified programs** against their authored witnesses, on the same tasks and checkpoint, with the same attribution. If body shares and clears look the same on programs the solver found, the diagnosis survives the imitation critique on the solved set.
3. **An inverse-dynamics probe.**
   - **Data:** (s, op, s') triples from executing random legal ops on real frames.
   - **Train** the current encoder to predict the op from (s, s'), then an encoder that sees each example's (s_i, s'_i) aligned.
   - **Far from 100%** on this one-step problem: fix the comparison first. The solver's ranker, its body policy and #164's value function V(s, g) all need it.
   - **Near 100%:** the encoder can compare two stacks, and the last-op result points to the training data or objective instead.

**World-v3 is predicted to stay near its floor** under any skeleton-first search until:
- its top-level skeleton gets a hierarchy;
- its bodies are split, by nested skeletons or macros.

## What this does not measure

- **Imitation.** Every number scores frontier-authored witnesses. A clear means the beam could reproduce the authored program, not that no other program solves the task.
- **Coverage.** No search that commits at the `[` or reads a sub-spec exists yet.
- **The exit comes from the authored witness.** A search would derive it. The oracle bounds what a derived sub-spec could buy the policy; it does not estimate it.
- **V.** Every arm here trains an op policy. None is a value function.

## Files

- `report.json`: `experiments/frame_campaign2.py report`, including the coded readings.
- `cells.json`: every measured cell without its per-op rows (the rows stay in `checkpoints/`).
- `ranker.json`: each ranker seed's calibration and per-step charges.
- `training.json`: validation curves.
- `checks.py`, `checks.json`, `checks.out`: the follow-up checks, reading per-op rows from `checkpoints/`:
  - seen and unseen skeletons;
  - the charge per sub-search;
  - the exchange table under the learned charge;
  - last-op uniqueness;
  - the control without the outlier;
  - leave-one-out on world-v3's stacking ratio.
- `tempered_beam.py`, `tempered-beam/`: the solver run at T = 1 and at the calibrated T.
- `runner.log`: the rerun's job log, local times (PDT).

Cost (`runner.log`): 2 h 40 min of GPU in three lanes, then about 2 hours of CPU measurement in two processes. The run was restarted once, after the Desktop session that owned its worktree was closed, deleting it.
