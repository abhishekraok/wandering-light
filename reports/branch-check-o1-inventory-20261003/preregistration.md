The program-synthesis janthu (e-den, Codex) wrote this.

# branch-and-check (#164), O1-B: bounded continuation inventory

Design frozen before new archive counts or suffix trials, 2026-10-03. O1-A's
recorded-suffix labels are not reused as optimal-distance orderings. This is
an inventory of a fixed, shown-example continuation catalogue, not a ranking
model, behavioural verification, or a solver improvement measurement.

**Prediction:** the train archive contains at least 100 unequal catalogue-bound
pairs across at least 20 distinct behaviour/skeleton groups, with distinct
states from the same completed segment at the same executed prefix length.
At least one member of each pair differs from every state on the authored
witness. Both members must have a successful replay-confirmed continuation.
**Falsifier:** fewer pairs or groups, provenance errors, or operation/runtime
censoring that prevents this gate. A pass only warrants designing the next
controlled diagnostic; it does not establish goal dependence or warrant training.

## Source and selection

Use only main's
`checkpoints/short-segment-search/short-segment-difference-20260923/archive/train-capped-s0-e3-segment-k16-d8.jsonl.gz`,
SHA256 `e358fca13e69880a48f0b3794d36769b1815f3dd249c765bc08b52fc7c1fca78`.
Require unique `(behaviour, example_set)` rows, archived split train, and live
`assign_split(BY_NAME[name].skeleton) == "train"`. No eval archive, hidden
examples, fitted-program verification draws, new authoring, or model calls are
inputs to construction.

Select example_set 0 only. Sort behaviours by SHA256 of UTF-8
`o1b-v1|<behaviour>` (lexical behaviour breaks digest ties); retain the first 64.
Selection does not use O1-A counts, fit status, witness flags or trial outcomes.
Retain the selected key manifest and source hash in the run output. Never
replace a selected row after rejection or censoring.

For each selected task, flatten archived `segments[*].states` from completed
segments (`stop is None`), attaching segment number. Deduplicate exact
`(segment, program)` records; inconsistent duplicates are provenance errors.
Sort by SHA256 of UTF-8 `o1b-v1|<behaviour>|<example_set>|<segment>|<program>`
(then that complete string), retaining the first 32 records, before replay or
closure checks. Do not refill. This is a sample of the retained merged-cut
archive, already biased toward policy scores and witness/path retention; it
is not all generated states, the full root frontier or flat-beam successors.

## Executed states and frontier membership

Replay the full witness and require the shown goal and saved combinator cuts
to agree, subject to the resource censoring below. Replay archived root paths
in segment order; the union of roots through the current segment is the
search's already-expanded product set. Replay each selected candidate path.
Its last step must be a top-level combinator cut according to `closed_cuts`.
Exclude products already expanded, products equal to the goal, and products
that fail genuine closure on **any** shown example (`closed_product`).
A retained archived record alone is not evidence that the product was eligible
for `_next_roots`; the expanded-root check is required. Incomplete-segment
records do not enter the primary panel.

Within a segment collapse equal products, retaining the record with the
smallest selection digest, even if another path is shorter. Keep discarded
path metadata in telemetry. A pair must have distinct products, equal executed
path length and the same segment/task/example axes. Report raw retained cuts,
code-bearing cuts, expanded cuts, closed eligible candidates, and dropped
repeats separately. Witness-only closed states absent from this archive never
enter the primary population; witness membership is recomputed by equality
against **all** witness states, not by `on_witness`.

## Catalogue and loops

Build the directed graph of **all** replayed witness product states: each
transition has its actual op and unit outer-op length. Equal products are the
same vertex, so recorded loops can be spliced. Treat the goal as absorbing.
For each unique genuinely closed non-goal witness product, choose its shortest
path to the goal in this graph; ties use lexicographically smallest full op
sequence. Replay every reconstructed path from its source and require the
shown goal; silent graph/replay disagreement is a provenance error.

Deduplicate those op sequences. Exclude empty sequences and sequences longer
than 96 ops. Sort by `(op count, full op tuple)` and retain the first 32.
Record the entire retained catalogue and truncation counts. No returned-fit
program or candidate-specific suffix selection is used. A single catalogue is
tested on **every** eligible candidate of that task, regardless of the original
witness cut or prefix position. The catalogue is target-selected and
witness-derived: this selection bias must accompany any interpretation.

Replay each catalogue sequence to completion on every candidate. Only exact
final product equality on every shown example is success. Failures, refusals
and capped trials supply no adverse optimal-distance evidence. The label is
minimum outer-op length among successful sequences in this fixed catalogue;
not minimum fuel, runtime, or optimal distance. A candidate with no success is
unlabelled, never a negative. Equal minima form ties, never ordered pairs.

## Bounds, ordering and incomplete work

Use one CPU process. Each per-example primitive call to `step_raw` receives
fuel limit 2,000 and the existing quotation-depth limit. Products must remain
at most 4,096 atoms by `atoms_exceed` after every outer op, including prefix
and root replays. A refusal charges the full 2,000 fuel allocation to a
conservative budget; a success charges its reported fuel. This bound differs
from the archive's default interpreter fuel and may censor useful paths;
report the difference, and never classify it as bad provenance or unsolvable.

All witness, roots, reconstructed catalogue paths, candidate prefixes and
suffix trials count toward per-task caps of 50,000 per-example primitive
calls and 2,000,000 charged fuel, and run-wide caps of 2,000,000 calls and
100,000,000 charged fuel. Check before every primitive call, allocating only
remaining fuel when less than 2,000; a trial hit by that reduced allocation is
resource-censored. Atom-limit breaches also censor that trial. No hidden
unlimited replay helper may bypass these counters.

Task order is the selection order above. Root replay order is segment order,
then archived root order; candidate and trial order is candidate digest order,
then catalogue order. Graph source states use first witness occurrence order.
A monotonic 900-second run deadline includes hashing, reading, graph work and
replays. The persistent unit gets RuntimeMaxSec=1800 and MemoryMax=4G.
Expected runtime is at most 15 minutes; this is not a quiet-machine timing or
coverage comparison. Record start/end, revision, all counters and peak resource
observations available from the unit.

If root/witness construction or any candidate's catalogue matrix is incomplete,
exclude the **whole task** from primary pairs. Successful partial trials remain
separately reported observations. Ordinary refusal of a fully attempted
catalogue sequence is complete but unsuccessful, not a distance negative.
On run-wide resource/deadline exhaustion, preserve an incomplete output with
all selected/unprocessed task keys. A killed unit or a log lacking `exit 0`
is not a result. Do not silently rerun with larger caps or a different subset.

## Reporting, grouping and next decision

Emit a machine-readable manifest, per-task catalogue, selected records,
replayed products, membership/closure exclusions, full trial outcomes and
successful continuations, plus aggregate primary pairs and distinct groups.
Outputs/data live under main `checkpoints/o1-inventory-20261003/`; commit only
compact telemetry/report necessary to reproduce the conclusion. The gate is
computed solely from completed tasks with no provenance errors; a resource
hit is reported explicitly. A gate pass under a partial manifest is a bounded
existence result for completed selected tasks, not a prevalence estimate.

Group by the transitive union of behaviour, live skeleton and identical full
witness op sequence, including all example sets and paths; no train/validation
split is drawn here. A later head-only split would still have a policy trunk
exposed to train behaviours. Publication in the issue/repository will paraphrase
relevant owner ideas (Shortcut learning; Snapshot to snapshot computation;
Learning efficiency alongside correctness), never copy his notes.

No threshold, architecture, shuffle comparison or model seed is chosen from
this inventory. Even a pass needs preregistered cheap metadata/state-distance
controls and target-switched or goal-shuffled tests that preserve the candidate
panel; equal-prefix matching alone cannot establish useful goal dependence.
Only after that separate diagnostic may training be proposed. Solver
integration still needs a three-seed verified coverage/total CPU-clock gate;
BEST remains under best-system-command (#158)'s own cohort rule.
