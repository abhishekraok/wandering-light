# Where a search decomposition would pay: the budget law per segment and per body (#160, #156)

The budget law (`docs/STATUS.md`) says a beam of width 64 ranked on cumulative
log-probability carries a witness only if its summed per-step deficit stays
under `ln 64` = 4.159 nats. A search that decomposes the witness gets one
`ln 64` per sub-search instead. This asks which decomposition would let the
policies we have fit the witnesses we have, before any such search is built.

Driver `experiments/segment_composite.py`, CPU, under a minute per policy. It
walks every witness under a policy exactly as `staged_transfer.score` does and
keeps the per-step deficit, one witness per behaviour, on the eval splits.
Each row charges a sub-search the **actual** summed deficit of the ops it
contains, not a length times an average:

- **flat**: the whole witness is one sub-search. This reproduces
  `witness_deficit.py`'s `within_budget` exactly in all four cells.
- **segmented**: cut after every top-level combinator (a closed state); a
  segment carries every op inside it, bodies included. This is what a
  segment-reset beam would get.
- **quotations**: each quotation body is its own sub-search -- an op belongs to
  the innermost quotation open when it is written, a `[` to the level that
  opens it and the `]` to the body it closes -- and the top level is one more.
  The shape of inner-frames (#160) E2.
- **both**: quotations, with the top level also segmented.

| policy | cell | witnesses | flat | segmented | quotations | both |
|---|---|---:|---:|---:|---:|---:|
| world arm `1m-s0-e5` | v4 eval | 45 | 9 | **10** | 13 | 14 |
| | #101 cohort behaviours | 43 | 9 | **10** | 13 | 14 |
| | world-v3 eval | 44 | 2 | **2** | 2 | 2 |
| BEST `staged-s0-e3` | v4 eval | 45 | 5 | 5 | 5 | 5 |
| | #101 cohort behaviours | 43 | 5 | 5 | 5 | 5 |
| | world-v3 eval | 44 | 1 | 1 | 1 | 1 |

## Segment-reset search is predicted null, and was not run

On the world arm, resetting the budget at closed states adds **one** witness
on v4 eval (`span_inside_interval`) and **none** on world-v3. The deficit lives
inside segments, not between them:

- **Bodies carry 79-82% of the world arm's deficit** (BEST: 90-92%), counting
  each op to its innermost quotation; by step kind, body ops are 59-77% and the
  `[` that opens a quotation another 14-20%.
- The median v4 eval witness has **one** segment: 4.6x the budget whole and
  4.4x after cutting. On world-v3 the median goes from 26.5x to 17.5x.

This was the proposed first stage of the build (a search-only experiment:
segment-reset ranking on the world-arm checkpoint, cumulative log-probability
as the control). It is recorded here as a prediction **not run**. Its expected
effect is +1 of 45 and +0 of 44. The ranking key it needs is not a small change,
because resetting the score at closed states erases the only quantity that
separates a program three segments in from one a single segment in. And the
flat law already underpredicts v4 about threefold (BEST: 5 within budget
against 16-18 behaviours solved), so even a large move would not cleanly
refute the law.

## Bodies are where a decomposition has to act

Even giving every body its own budget moves world-v3 by **zero**. Under full
decomposition the costliest sub-search is a **body** in 31 of 44 world-v3
witnesses and 33 of 45 v4 ones, at a median 7.1x the budget on world-v3 and
1.9x on v4. The longest sub-search is a median 36.5 raw ops on world-v3 and 7
on v4, so fitting it needs a per-step deficit of **0.114** on world-v3 against
the arm's 0.661, and 0.594 on v4. Bodies are short in *macro tokens* (median
3-4 top-level tokens), but under a base-op policy they are long, and they are
where the policy pays.

So decomposition is not a lever until the per-step deficit **inside bodies**
falls or bodies get shorter. The candidates are a frame-conditioned body ranker
(#160 E2) and macros under a mark (#162, which needs #149).

## What this does not say

The quotation rows charge a body the deficit the *flat* policy pays for it,
with the whole stack in view. A sub-search over a frame would pay something
else, and that is E2's question, so these rows bound it rather than predict it.
Every row assumes the correct intermediate state stays in the top 64 at every
stage, which no path score guarantees, and the witnesses are authored programs
where a search may find others. `1m-s0-e5` is the lowest-deficit world-arm
cell, **selected on the world-v3 eval split** these rows are scored on (spread
0.631-0.776 across 18 cells); it is fixed here so later measurements pair
against it. The 44- and 45-witness rows and the 43 #101 cohort behaviours are
different denominators and are not compared with each other.
