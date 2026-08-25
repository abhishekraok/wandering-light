# One-step policy probes on `deep-corpus-v1`

The cheapest test of #33's premise, run before building the search hybrid.

#33 proposes a policy head that orders BFS expansion. If the optimal first
action is not predictable from `(input, output)` beyond what the type mask
already gives away, ordering will not work either. `deep-corpus-v1` stores
`optimal_first_action_names`, certified by complete BFS expansion and marked
complete on 85% of tasks, so the label already exists and no search is needed.

Reproduce:

```
uv run python experiments/policy_probe.py <path-to-deep_corpus_v1>          # first action
uv run python experiments/policy_probe.py <path-to-deep_corpus_v1> --last   # last action
```

20,738 train / 2,907 validation / 2,905 test tasks with complete labels.

## The metric is ordering, not accuracy

A policy head's job is to make BFS try the right op sooner, so what matters is
how far down the predicted ranking the first optimal action sits. Rank 1 means
the search expands no wrong branch at that level; rank 10 means it expands
nine. Ties are broken **pessimistically** — an action tied with an optimal one
counts as tried first — so a flat distribution cannot score well.

    One-step policy probe -- 2905 test tasks
    
    predictor                 top-1  mean rank   median    p90
    uniform over all          0.0%     117.91      118    118
    type mask                 0.0%       9.78        6     21
    marginal frequency        6.4%      26.81       15     69
    conditional lookup       42.1%       3.18        2      7
    learned (mean-pool)      40.3%       2.98        2      7
    learned (conv)           42.2%       2.86        2      6
    
    Mean rank by certified distance
    
    predictor                   d1      d2      d3      d4      d5      d6      d7
    uniform over all        117.97  117.91  117.90  117.90  117.90  117.91  117.94
    type mask                14.75    9.57    9.55    9.56    9.56    9.56    5.34
    marginal frequency       55.08   26.37   26.51   24.42   24.03   23.70   15.80
    conditional lookup        3.61    2.73    3.25    3.28    3.29    3.38    2.21
    learned (mean-pool)       4.07    2.67    2.95    2.91    2.96    3.14    2.04
    learned (conv)            3.18    2.43    2.73    3.05    3.06    3.09    2.03

## What it says

**The type mask is worth a lot, and it is free.** Mean rank 117.91 → 9.78, a
12x cut. Measured separately: 9.87 of 118 functions are type-compatible on
average, so masking takes the per-step choice from 6.88 bits to 2.95. That
corroborates #33's masking claim (it reported 6.88 → 3.30) and is slightly
better than #33 assumed.

**A lookup table on two fields does most of the rest.** `P(action | input
type, output type)`, a dict with no learning in it, takes mean rank 9.78 →
3.18 and top-1 from 0% to 42.1%.

**The learned encoders barely beat the table.** A byte-level mean-pool encoder
gets 2.98 mean rank and *loses* on top-1 (40.3% vs 42.1%). A convolutional one,
which can see local structure the mean-pool destroys, reaches 2.86 and 42.2% —
a **10% improvement in mean rank over a dict, and a tie on top-1**.

**Prediction does not degrade with distance.** This is the result that most
constrains #33. If first-action prediction got harder as tasks got longer, a
policy would help least where search is most expensive. It does not: the
conditional lookup runs 2.73–3.38 flat across distances 2–6, and distance 7 is
the *easiest* row for every predictor. Distance 1 is the hardest, which is an
artifact — a distance-1 task's optimal first action is its entire program, so
the label is maximally specific.

## What this does and does not establish

It sets a bar: **any #33 encoder must beat mean rank 3.18 and top-1 42.1%**,
which is what a two-field dict achieves. Two architectures got within 10% of
that, so the burden is now on showing that a better encoder extracts signal
the types do not already carry.

It does not prove none can. Both encoders here are small, trained for 12
epochs with no tuning, and read a serialized JSON string — the mean-pool one is
close to a bag of bytes. A transformer over a structured encoding may do
better. What the probe shows is that the obvious cheap architectures do not,
and that the gap to close is small enough to be worth measuring before
building the hybrid.

It also tests **one step only**. #33 needs ordering at every step of a search,
and nothing here says the second action is as predictable as the first.


---

# The last action, and why it is the more interesting end

The first-action probe answers half the question. A policy head orders search
at *every* step, so the second half is whether some other step is more
learnable. `deep-corpus-v1` also stores `optimal_last_action_ids`, so the same
machinery answers it.

**The mask is not symmetric.** A first action must *consume* the task's input
type; a last action must *produce* its output type. Masking on the wrong field
would compare every predictor against a mask that does not apply. Measured:
9.87 functions consume a given input type on average, but **25.88 produce a
given output type** — so masking is worth 6.88 → 2.95 bits at the front and
only 6.88 → 4.53 bits at the back.

    Last-action policy probe -- 2905 test tasks

    predictor                 top-1  mean rank   median    p90
    uniform over all          0.0%     117.83      118    118
    type mask                 0.3%      25.72       26     37
    marginal frequency        7.6%      18.06       11     42
    conditional lookup       22.3%       4.80        4     10
    learned (mean-pool)      30.7%       4.10        3      9
    learned (conv)           42.1%       3.43        2      8

    Mean rank by certified distance

    predictor                   d1      d2      d3      d4      d5      d6      d7
    conditional lookup        6.62    5.48    5.10    4.41    4.18    4.21    4.27
    learned (conv)            4.26    3.75    3.60    3.17    3.09    3.21    3.47

## The finding inverts

At the front, a dict matches a trained encoder. At the back it does not:

| | lookup | conv encoder | gain |
|---|---:|---:|---:|
| **first** action, mean rank | 3.18 | 2.86 | 10% |
| **first** action, top-1 | 42.1% | 42.2% | tie |
| **last** action, mean rank | 4.80 | 3.43 | **29%** |
| **last** action, top-1 | 22.3% | 42.1% | **+89%** |

**Learning pays at the back and barely pays at the front.** The same encoder
that could not beat a two-field dict on the first action nearly doubles its
top-1 on the last one.

The reason is visible in the mask numbers. At the front, the input type alone
narrows 118 functions to about 10, and a dict keyed on two type fields
captures most of what is left. At the back the type constraint is much weaker —
26 functions produce any given output type — so there is far more residual
signal, and it is in the values rather than the types.

Note the absolute ordering still favours the front: 2.86 against 3.43 mean
rank. The first action is *easier*; the last action is where a **model** adds
more than a table.

## What that suggests for #33

#33 proposes a forward policy: order BFS expansion from the input. These
numbers say the learnable signal is stronger running **backwards from the
target**, which is a different architecture — bidirectional search with a
learned backward policy, meeting a cheap forward expansion in the middle.

That is a claim about one step at each end, not about a full search, and it
should be tested before it is built. But it is a concrete alternative to the
design in #33 and it comes from measurement rather than preference.

**Both probes agree on one thing:** prediction does not degrade with distance.
Last-action rank *improves* from 6.62 at distance 1 to 4.27 at distance 7 for
the lookup, and 4.26 to 3.47 for the encoder. Longer programs do not make
either endpoint harder to guess.
