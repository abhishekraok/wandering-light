# What these cells mean

Prose for the generated tables in `README.md`. Numbers live there; this says how
to read them and what not to conclude.

## One host, one commit, one cell at a time

Every cell here ran alone. That is not fussiness: the recorded `mark-28` cells
this work compares against were run **three concurrently**
(`reports/world-plus-v4-policy-20260912/runner/study.sh:48`, "search 18 cells in
3 lanes"), and wall-clock is `docs/metrics.md`'s primary unit. The runner's own
comment says solve counts survive concurrency; it does not say the timings do.

The consequence is that a `mark-28` wall-clock is comparable only within one
host, one commit **and one load condition**. #101's cost-to-clear table and the
941 s row predate all three. The arms here compare with each other.

The noise floor was measured, not assumed: the same cell run twice under
different task orders gave **195.0 s and 208.7 s timed**, about 7%, with
**identical** execution counts (71,700,695) because a search is independent per
task. Both figures are `timed_seconds`; the matching `cell_seconds` are 197.0
and 210.7, and the two units must not be mixed in one comparison.

## The baseline is the published baseline

The control arm was retrained here under #101's recipe and is bitwise equal to
[world-plus-v4-policy (#134)](../../../../issues/134)'s v4-only arm: validation
cross-entropy 0.952 / 0.956 / 1.094 / 1.210 / 1.362 / 1.509 at epochs 1-6 on
seed 0, and per-seed cohort results 94/12, 74/10, 89/11 tasks/behaviours at epoch
4 — every figure matching its recorded cells. So these are measurements against
the published number, not against a re-implementation of it.

## Read behaviours, not tasks

The cohort is **43 distinct behaviours behind 300 tasks**; `negate_each` alone is
21 of them. Two consequences run through every table:

1. **Per-task coverage has an effective n near 43.** Within a single arm the
   seed spread on tasks is 7-10 of 300 while the spread on behaviours is 0.5-0.8
   of 43, so a task difference under about 15 says nothing on its own.
2. **A change can move tasks without moving anything real.** The solved-program
   cache is the clean case: +23 tasks and zero genuine behaviours.

Older tables here carry a `behaviours*` column that drops `all_positive` and
`bounds_cover_list_flag` by name, whose samplers `docs/STATUS.md` records as
leaking. **That column is superseded and should not be read as a quality
measure.** Draw-testing contradicts it in both directions:
`bounds_cover_list_flag` finds a genuine 46-op program at beam 512, and
`aligned_matches` -- which the column let through -- is a shortcut. Use the
verified counts in `verified_width.json` / `verified_epoch_grid.json` /
`verified_all.json`, and note that "verified" is the union of draw-tested
`genuine` verdicts and `equals_catalogue_witness` ones, the latter never drawn
against at all.

## Grading has per-instance noise that a behaviour verdict does not

A program that fails held-out grading on the task it was found on can pass on a
sibling task of the same behaviour, because grading draws different examples per
task. That is the mechanism by which a cache appears to credit a behaviour no
search credited, and it is why both apparent behaviour gains land on leaky
samplers. Anything trained or selected against per-task verdicts inherits that
noise.

## The budget, and the error that was made with it

The beam keeps the `W` highest cumulative log-probabilities, so a path `d` nats
behind the best has roughly `e^d` ahead of it and a width-`W` beam tolerates
about `ln W` nats — 4.159 at 64. Dividing by witness length gives a per-step
allowance, and behaviour coverage tracks it across buckets: allowance 0.832 /
0.693 / 0.260 / 0.107 / 0.053 nats per step against coverage 100% / 67% / 67% /
33% / 11%.

**The budget is sound; one inference from it was not.** A x4 width step is
`ln 4` = 1.386 nats, which was converted into "0.33 extra depth" using the 4.2
nats-per-depth figure from `reports/materialized-rollout-20260829` — a figure
measured on 5-8 op witnesses, where one depth is one op. On a 16-op witness the
same 1.386 nats raises the allowed per-step deficit from 0.260 to 0.347, a third
more across every step, and beam 256 duly measured +2 non-leaky behaviours with
the gain concentrated in the 11-20 bucket.

So: a per-step quantity must not be compared against a total-nats requirement.
That error voided 13 of 28 refutations in the review sweep behind
[clear-stack-pbe-v4 (#99)](../../../../issues/99), and the cells that re-open
them are in this directory.

What survives unchanged is the 41+ bucket, 19 of the cohort's 43 behaviours at a
median 78-op witness: clearing it by width alone would need `W ≈ 4 x 10^33`.
Length reduction remains necessary there, which is what
[highway-construction (#96)](../../../../issues/96),
[mine-v4-library (#148)](../../../../issues/148) and the staged re-authoring in
[witness-trajectory-shape (#150)](../../../../issues/150) are for.
