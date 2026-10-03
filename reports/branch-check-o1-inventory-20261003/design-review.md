Claude (Opus 5.5) wrote this — independent other-family design review, high effort, read-only, of `reports/branch-check-o1-inventory-20261003/preregistration.md` at `13d0a22` for branch-and-check (#164) O1-B. No archive was loaded and no counts were computed.

# Verdict: CHANGES_REQUIRED

The core construction is sound, but five rules are missing or contradictory. Each changes the gate number or what an implementer must pick, so each needs a one- or two-sentence visible amendment before any count. Implementation can proceed in parallel.

## What checks out against source

- **Expanded-root reconstruction:** "union of roots through the current segment" is exactly the set `_next_roots` filters on (`stack_beam.py:878`, `:898`). Roots are archived in full, in run order (`search.py:497-500`).
- **Cut and closure:** every archived record is a top-level combinator cut by construction (`stack_beam.py:685`, `:701`). `closed_product` rejects marks and code at any depth, on every example (`staged_cycles.py:135`, `witness_trajectory_shape.py:70-71`).
- **Loop-spliced suffixes:** the machine is deterministic on full products, so any path in the witness-state graph replays to the goal from its source. The tie rule is well defined.
- **Goal dependence:** correctly disclaimed (lines 17, 144-147). The label is a function of state and one witness, with no goal contrast, so this gate cannot establish it.
- **Schema:** `example_set`, `split`, `witness`, `witness_cuts`, `segments[*].stop/roots/states` all exist as the design assumes.

## Blocking findings

1. **Out-of-fuel at the full 2,000 allocation is unclassified** (lines 93-99 against 120-121).
   - It reads as an "ordinary refusal", so the trial is complete and the task stays primary.
   - The search ran at 100,000 fuel (`stack_beam.py:296`), so a shorter catalogue sequence can be censored while a longer one succeeds. That inflates the minimum and manufactures unequal pairs.
   - Amend: classify refusals by reason (`"out of fuel"` at `stack_machine.py:278`, `:305`, op-prefixed at `:314-316`, so match the suffix). Admit a pair only if the larger-labelled member has no fuel- or atom-censored trial on a sequence shorter than its label.

2. **Censoring of a prefix replay has no scope, and the atom bound's mismatch is unstated.**
   - The search kept states up to 16,384 atoms (`solve.py:138`); the design caps at 4,096, so valid archived roots and candidates can breach it.
   - A censored root path leaves the expanded set unknown, so whole-task exclusion is right there. For a censored candidate prefix the text gives both "censor that trial" and "exclude the whole task".
   - Amend: pick one for candidates, and disclose the atom difference as the fuel difference already is.

3. **The charging rule does not fix the counters, and the caps bind on arithmetic alone.**
   - Root programs extend earlier roots, and candidates extend roots. Whether a shared prefix is charged once or per path is unstated, so two faithful implementations get different completed-task sets.
   - Unshared root replay is up to 16 roots × 8 ops × (0+…+11) = 8,448 outer ops, times the example count, against 50,000 calls.
   - A full 32 × 32 matrix of refusals is 2,048,000 charged fuel against the 2,000,000 per-task cap.
   - Run-wide caps average 31,250 calls and 1.56M fuel per task, below the per-task caps, so tasks late in the order starve.
   - A cap hit excludes whole tasks and reruns are forbidden, so this can burn the registration. Amend: state the sharing rule. Charge 2,000 only for refusals of `runs_code` ops; a guard refusal executes nothing (`stack_machine.py:283-286`), so charging 1 is still conservative.

4. **Provenance errors: falsifier or exclusion?** Line 15 lists them as a falsifier; line 133 computes the gate from tasks without them. O1-A used `not errors`. Say which.

5. **The pair and group units are undefined.**
   - The same product pair can recur in a later segment with different retained programs. State whether it counts once per task or once per segment.
   - The "transitive union" (line 137) differs depending on whether it runs over the 64 selected rows or all train behaviours; an unselected behaviour can link two selected ones. State the population, and that a group counts only if it holds at least one primary pair.

## Nonblocking limitations

- **Power:** O1-A's 466,241 kept records over 592 tasks is about 788 per task. Thirty-two of those across up to 12 segments is under 3 per segment before exclusions. A fail will mostly reflect sampling; stratifying the draw by segment would fix it without using outcomes.
- **Within-segment equal products cannot occur** in a valid archive, since `closed` is keyed by product (`stack_beam.py:713-715`). A nonzero collapse count signals a replay bug and should be an error.
- **Path length is not a state property:** it is the length of the best-scoring merged program.
- **Labels are upper bounds on both sides,** so an unequal pair does not imply unequal true distance. An on-witness member's label is true by construction; report both-off-witness pairs separately.
- **Lexicographic ties** must be resolved globally (reverse distances from the goal, then greedy forward), not by first-found order. State that the comparison is on token tuples by code point.
- **The 900 s deadline is machine-dependent:** record which bound ended the run.
- **Memory:** atoms are checked only after outer ops, so growth inside one combinator call (stack-machine-quotation-growth, #172) can hit MemoryMax. Decide now what a kill means.

## Synthetic tests required

Each is a small hand-built row.

1. A witness with a repeated state returns the spliced suffix, and two equal-length routes resolve to the lexicographically smaller.
2. A goal occurring mid-witness is absorbing, with no empty sequence in the catalogue.
3. A candidate equal to an earlier or current-segment root is excluded; one that is a root only in the next segment is kept.
4. A cut state with code inside a list on one example only is excluded.
5. Candidates in different segments, or with different path lengths, form no pair; equal minima form no pair.
6. A pair with both members on the witness is rejected; membership ignores the `on_witness` flag.
7. A candidate with no successful sequence is unlabelled and forms no pair.
8. A shorter sequence that runs out of fuel while a longer one succeeds is handled per amended rule 1.
9. A root over the atom bound excludes the task; a candidate over it follows amended rule 2.
10. Counters are exact on a known row: a guard refusal, a combinator refusal, a shared prefix, and a cap reached mid-matrix giving an incomplete task and preserved output.
11. Selection and sampling order are pinned to literal digests, with no refill after exclusion.
12. A duplicate `(segment, program)` with differing fields, and a `stop`-non-None segment, are an error and an exclusion respectively.
13. Grouping: two behaviours linked only through a third merge or not, per amended rule 5.
14. A provenance error in one task has the gate effect chosen in amended rule 4.

A passing design review does not replace a later code and results review before anything lands.
