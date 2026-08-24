# BFS baseline on `stack_pbe_v2`

The number a learned solver has to beat, measured before training anything.
Regenerate with:

```
uv run python experiments/stack_machine_baseline.py \
  wandering_light/evals/data/stack_pbe_v2.jsonl.gz eval
```

Corpus: 1,857 tasks certified against the `executable-26` basis to depth 6;
355 in the eval split. Search is the same behaviour-space expansion the corpus
generator uses, so "nodes" counts distinct behaviours expanded — the unit a
learned policy would reduce.

```

BFS baseline on 355 eval tasks

The number a learned policy has to beat is NODES, not accuracy.
Search already reaches everything the corpus contains -- the corpus was
built by expanding to that depth -- so solve rate cannot separate a good
policy from a bad one. Expansions per solved task is what it can move.

   nodes to solution  median     1,119   p90    42,416   max   104,485   total    4,237,377

 node budget    solved     rate
         100        70    19.7%
       1,000       174    49.0%
      10,000       267    75.2%
     100,000       353    99.4%
   1,000,000       355   100.0%

 shortest   tasks   solved   median nodes   max nodes
        1       5        5              1           1
        2      25       25             11          15
        3      79       79             98         189
        4      87       87            772       1,527
        5      81       81          5,455      13,553
        6      78       78         40,666     104,485

BFS length disagreed with the certified label: 0
```

## Reading this

**Unlimited-budget solve rate is 100% and therefore useless.** A corpus built by
expanding to depth *d* is by construction one that search reaches at depth *d*;
forward generation cannot produce problems that defeat forward search. This is
the same reason `random_inputs_500_shortest_v1` was solved 100% at depth 4, and
it is a property of how the corpus is made, not a flaw to fix by going deeper.

**At a fixed budget it discriminates.** At 10,000 nodes BFS solves 75.2%, so
there are 25 points of headroom for a policy that orders expansion well. That
was not true of the depth-5 version of this corpus, where 10,000 nodes already
bought 96.9%.

**Node cost grows about 7.5× per unit of program length**: 98 → 772 → 5,455 →
40,666 median. Extrapolating, depth 7 costs roughly 300k nodes per task and
depth 8 roughly 2.3M, which is where both certification and search stop being
cheap.

**BFS length agreed with the certified label on all 355 tasks.** That is an
independent check on the corpus: the labels were produced by the generator's
expansion and confirmed here by a separate search to a separate target.

## Two ways to score a solver against this

| metric | baseline | what it rewards |
|---|---|---|
| solve rate at 10,000 nodes | 75.2% | finding more within a fixed compute budget |
| median nodes per solved task | 1,119 | finding the same things for less compute |

The second is the one that connects to Kt, and it stays meaningful even if a
solver reaches 100%.
