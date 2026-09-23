# How often the beam re-converges on a state it has already seen (#164, step 1)

`stack_beam` keeps a visited set and discards any successor already in it with
the verdict `duplicate`. Nothing recorded that the second route existed. The
authored-alternatives probe on closed-state-value-search (#156) found 1 of 320
witness pairs re-converging on an interior closed state, and branch-and-check
(#164) rests on whether that holds for the neighbours a *search* generates,
because a value function propagates only through states two routes share.

`experiments/seen_set_collisions.py`, a `BeamObserver` that reads the verdicts
and changes nothing in the search (the test asserts the result is identical
watched and unwatched). A duplicate is a **cycle** when the state is on the
duplicating node's own path, **same-layer** when another node reached it in
the same layer (two programs of equal length), **earlier-layer** when the
second route is longer. **Closed** means no open quotation and no code on the
stack, the states a ranker over closed states would archive. Under a
depth-synchronous beam the second route is never shorter, so the visited set
loses no path length here; it would in a best-first search.

Cells: `BEST`'s search (beam 64 x depth 48, guard both, state cap 16,384),
library off, CPU. Policies: the world arm `1m-s0-e5` (the checkpoint the
segment composite and the #160 gates read; `checkpoints/world-arm-20260922/policies/1m-s0-e5`)
and `BEST` = `staged-s0-e3` (Hub `e5a0b139`). Task sets: the #101 cohort (300
tasks, 43 behaviours; `checkpoints/best-system-20260922/cohort.jsonl.gz`) and
the `stack_pbe_world_v3` eval split (44 tasks). Code at `c13f679`. Three runs
gave identical counts (the third added the uncensored column); timings were
not read (the first run overlapped campaign 2's CPU cells, 01:25-01:55 PDT
2026-09-23). Per-cell tables and JSON
in `cells/`.

## Results

| cell | closed cross-path duplicates per closed retained state | distinct closed states met | tasks with one | meetings at length >= 6 | >= 10 | second route same length | duplicates, share of ranked | duplicates, share of all allowed successors | closed share of retained | beam filled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| world arm, #101 cohort | **7.2%** (5,127 / 71,061) | 5.8% (4,128) | 288 / 300 | 53.0% | 47.3% | 22.9% | 2.7% | 2.8% | 11.8% | 97.6% |
| world arm, world-v3 eval | **8.7%** (1,537 / 17,578) | 6.3% (1,101) | 44 / 44 | 65.1% | 56.4% | 27.8% | 4.0% | 4.5% | 13.5% | 98.3% |
| BEST, #101 cohort | **7.6%** (9,606 / 126,365) | 5.4% (6,838) | 288 / 300 | 76.0% | 68.7% | 18.7% | 5.6% | 6.3% | 24.6% | 97.2% |
| BEST, world-v3 eval | **18.9%** (2,558 / 13,509) | 9.0% (1,222) | 44 / 44 | 78.1% | 70.6% | 11.5% | 7.2% | 7.8% | 10.4% | 98.3% |

**Censoring.** The search checks the visited set only for the successors the
ranking reaches before the beam fills, about 68 of ~1,300 allowed successors
per layer (5.1-5.6%). Run 3 judged every successor below the cut as the search
would have, guards first, against the set as it stands at the end of the
layer: the duplicate share below the cut is 2.8 / 4.5 / 6.3 / 7.8%, against
2.7 / 4.0 / 5.6 / 7.2% among the ranked. The censoring is real and changes
nothing: a top-ranked successor and an unranked one are already-seen at the
same rate. (Below-cut duplicates are not split by class; they include the
node's own loops.)

Duplicates by class, cross-path share: 31.5 / 31.7 / 38.3 / 45.0%; the rest
are cycles. On the world arm's cohort cell, tasks that fit re-converge at 9.8%
of closed retained states, tasks that run to depth 48 at 6.7%. Second route
longer by >= 3 ops: 36.1 / 40.7 / 52.2 / 67.7%.

What the deep meetings are (world arm, cohort; `first` was retained, `second`
was discarded):

| first | second |
|---|---|
| `0 [ add ] fold` | `nil [ cons ] fold 0 [ add ] fold` |
| `nil [ cons ] fold [ dup mul ] map` | `[ dup mul ] map nil [ cons ] fold` |
| `[ dup mul ] map` | `nil [ cons ] fold [ dup mul ] map nil [ cons ] fold` |
| `drop nil` | `dup [ swap [ eq? ] swap cons filter null? ] swap cons filter` |
| `uncons [ [ over lt? ] [ drop ] [ swap drop ] ifte ] swap cons map` | `uncons [ [ swap lt? ] [ drop ] [ swap drop ] ifte ] swap cons map` |

Shallow ones: `dup dup` / `dup over`, `null?` / `nil eq?`, `nil` / `[ ]`,
`0 quote` / `nil 0 cons`.

## Reading

- **Search-generated routes do re-converge.** 5-9% of the closed states the
  beam keeps are reached again by a different route, in 288 of 300 cohort
  tasks and every world-v3 task. The authored-pairs probe (1 of 320) measured
  the wrong population: two authors choose different approaches; a beam
  permutes and pads one.
- **Most meetings are deep.** 53-78% at program length >= 6, 47-71% at >= 10,
  up to the depth cap. They are not only prefix identities.
- **What meets:** a no-op cycle inserted (`nil [ cons ] fold` twice is the
  identity), independent cycles commuted (`map` and reverse), a long detour to
  a small state (`drop nil` against a 14-op filter that empties the list), and
  bodies that differ but agree on the shown examples (`over lt?` against
  `swap lt?`). The last is aliasing on k examples, the held-out concern.
- **The visited set costs the beam nothing.** Duplicates are 2.7-7.2% of ranked
  successors and 2.8-7.8% of all allowed ones, cycles are 55-69% of the ranked
  ones, and 97-98% of beam slots fill every layer. It discarded evidence, not
  width.
- The staged checkpoint meets more and deeper than the world arm: its text
  has more cycles to permute.

Not measured: whether a backup through a met state would give any state a
label it does not already get from its own path, which is the value of the
re-convergence for a ranker, and anything about best-first search.
