# Does the body-cost diagnosis hold on programs the solver finds? (solver-program-rescoring #165)

**WEAKENED, as registered, against a prediction of SURVIVES.** The programs `BEST` finds on its own spend a quarter of their deficit in bodies. The authored witnesses of the same behaviours spend 88%.

What they are:
- **Short bodies in pipelines.** The solver's programs chain short bodies with `map`, `filter` and `fold`. Their longest body has a median of 3.75 ops, against 12.5 for the authored witnesses.
- **Different from the authors' style.** The authors nest `ifte` inside a long `fold` body.

So the finding that bodies carry the deficit is partly the authoring style. On the behaviours it solves, the solver routes around long bodies.

**One confound limits this.** `BEST` was trained on staged-witness-reauthoring (#155)'s staged witnesses. Their longest body has a median of 8 ops, against 29 in the v4 training witnesses. So `BEST` writes the style it was trained on. What the data show is that short-body programs exist for these behaviours, and that a policy trained on them finds them. They do not show that the authored style is what makes bodies expensive for every policy.

The [registration](https://github.com/abhishekraok/wandering-light/issues/165) was posted before anything was scored. Driver: `experiments/solver_rescoring.py`; data: `rescoring.json`.

## What was scored

- **Programs.** The 19 verified programs the solver found on the #101 cohort that differ from the catalogue witness (`genuine` in #158's 10,000-draw test), over 12 behaviours. 17 were found by `BEST`'s cells and 2 by the world arm's.
- **Pairing.** Each is scored beside its behaviour's authored v4 eval witness, on that witness's task inputs. All 19 reproduce that task's outputs.
- **Attribution.** The inner-frames (#160) campaigns' own: `frame_gate.walk` under the machine mask, calibrated deficits, `composite` clears.
- **Medians.** Taken over behaviours, with the solver side the median over each behaviour's programs.

## Results

| median over 12 behaviours | `BEST`, authored | `BEST`, solver | world arm, authored | world arm, solver |
|---|---:|---:|---:|---:|
| body share of calibrated deficit | 0.884 | **0.241** | 0.806 | 0.633 |
| total calibrated deficit | 11.2 | **2.2** | 4.9 | 22.9 |
| worst body's deficit | 4.67 | 0.31 | 2.96 | 9.03 |
| ops | 17 | 19 | 17 | 19 |
| quotations | 4 | 3.25 | 4 | 3.25 |
| longest body, ops | 12.5 | **3.75** | 12.5 | 3.75 |

| clears, flat / both decompositions | `BEST`, authored (of 12) | `BEST`, solver (of 19) | world arm, authored | world arm, solver |
|---|---:|---:|---:|---:|
| commitment free | 5 / 6 | 17 / 17 | 6 / 6 | 4 / 5 |
| prior's charge | 4 / 5 | 3 / 10 | 4 / 5 | 2 / 4 |
| learned charge | 5 / 6 | 9 / 15 | 6 / 6 | 3 / 5 |

- **Under `BEST`**, the checkpoint that found them, the solver's programs cost a fifth of the authored witnesses. Almost none of that cost is in bodies.
- **Under the world arm**, which found only two of them, they cost almost five times more. Two policies like different programs, so this column says little about the diagnosis.
- **Example.** For `count_in_range`, `BEST` finds `[ lt? ] swap cons filter`, then `[ swap lt? ] swap cons filter`, then `0 [ drop 1 add ] fold`. The authored witness is one `fold` whose 31-op body nests two `ifte`s. `BEST` charges the authored witness 122 calibrated nats and its own programs about 2.

**Against the prediction:**

| | predicted | measured |
|---|---|---|
| reading | SURVIVES | WEAKENED |
| body-share gap, `BEST` | about 0.05 | 0.64 |
| total deficit, solver over authored, `BEST` | about 0.7 | about 0.2 |
| total deficit, solver over authored, world arm | 0.9-1.1 | about 4.7 |
| flat clears under `BEST`, solver at least as often as authored | yes | held: 17 of 19 against 5 of 12 |

## What it means, and what it does not

- **Selection.** Every program here was found by a flat beam, so it is within that beam's reach by construction. A low total deficit is therefore expected. A low body share is not: the found programs could have spent their small budget in bodies, and they did not.
- **Only solved behaviours.** Nothing here says whether cheap-bodied programs exist for the unsolved ones.
- **Training style.** `BEST`'s training witnesses (#155, staged) have short bodies: the median longest body is 8 ops, against 29 in the v4 training witnesses. The world arm, trained in the authors' style, found only two of these programs, too few to test whether it also routes around long bodies.
- **What it changes.** The campaigns' "the lever is inside bodies" was measured on the authors' programs, whose bodies are long. On the behaviours the solver solves, it finds programs whose bodies are short.
  - That agrees with campaign 2's "body cost is exposure": bodies are long in authored witnesses, not harder per op.
  - It suggests two remedies: training data in a short-body pipeline style, which #155 already is and which `BEST` shows works on these behaviours, and a search that prefers composing short bodies. Both are cheaper than deriving sub-specs for long ones.
  - The perfect-decomposition pilot proposed on #163 would decompose the *authored* witnesses. It should also report whether a short-body program exists for the tasks it solves.
