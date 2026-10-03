# branch-and-check (#164), O1-B continuation inventory

The registered O1-B gate failed. The bounded run completed as a command but
stopped at its global primitive-call cap, leaving the manifest incomplete;
completed tasks produced no qualifying pair. This settles the utility of this
exact bounded construction, not the existence of useful ranking data elsewhere.
No trainer, architecture, solver change or BEST promotion follows.

[Compact telemetry and independently checked counts](inventory_summary.json),
[registered definition and Amendment A](preregistration.md),
[initial design review](design-review.md), [pre-trial code PASS](code-review.md).
The numerical ledger is `docs/RESULTS_DETAILED.md`; issue argument is on
[branch-and-check (#164)](https://github.com/abhishekraok/wandering-light/issues/164).

The first and only run used a clean detached `c664a7c` (code `42c67f5`),
starting 2026-10-03 11:00:10 PDT and ending 11:00:45 PDT. This is not a
quiet-machine timing or coverage comparison; JSON's internal elapsed time
excludes import and serialization. Log ends `exit 0`; unit journal shows no
runtime/memory kill. Raw telemetry (~140 MB) stays under main
`checkpoints/o1-inventory-20261003/inventory.json`; its SHA256 is sealed in
compact telemetry. It is not copied into git.

Independent verification re-derived the task/record digest selections and
transitive all-row groups, recomputed catalogue minima on every complete
eligible matrix and enumerated primary pairs without the measured builder's
helpers. It replayed every successful sequence and checked its source closure,
non-goal status, cut membership, witness membership and unexpanded eligibility.
The one labelled candidate in a censored task remains a partial observation.

Interpretation: positive continuations are sparse in this retained-record
sample, and full-path replay consumed the registered call budget. A failed or
unlabelled candidate is censored for distance, never adverse distance evidence.
Target-selected witness suffixes and archived policy/witness retention bias
this panel; equal-prefix matching cannot establish goal dependence. We will
not silently increase caps, change selection or train on the surviving bounds.
The next queued research question is top-level glue versus quotation-body loss
under the capped policies, beginning with the existing #168 diagnostics.

Reproduce the independent verification from a checkout of the run revision
and the original run output, writing a new compact summary:

```bash
uv run python reports/branch-check-o1-inventory-20261003/verify.py --source-tree /path/to/c664a7c-tree --run-dir /home/abhishekrao/repos/wandering-light/checkpoints/o1-inventory-20261003 --out /path/to/new/verification.json
```

Original run command (reproduction is not authorization to rerun the campaign):

```bash
uv run python experiments/branch_check/o1_inventory.py --archive /home/abhishekrao/repos/wandering-light/checkpoints/short-segment-search/short-segment-difference-20260923/archive/train-capped-s0-e3-segment-k16-d8.jsonl.gz --out /home/abhishekrao/repos/wandering-light/checkpoints/o1-inventory-20261003/inventory.json
```

Use the prescribed persistent CPU unit with RuntimeMaxSec=1800 and MemoryMax=4G;
commit before launch and log the exact unit, revision, command and timing.
The CLI rejects an existing output and any source hash mismatch. The bounded
run preserves resource censoring and unprocessed keys; only a final `exit 0`
is a result. An unequal minimum successful continuation in this fixed catalogue
is not an unequal optimal distance or evidence of goal-dependent ranking.

Synthetic tests exercise graph splicing/ties, goal absorption, genuine closure,
expanded-root membership, outcome-independent selection/no refill, exact replay
charging, censorship/partial output, unique pair/group counting and fatal
provenance errors. They do not read the archive or held-out examples:

```bash
uv run pytest -q tests/test_o1_inventory.py tests/test_o1_archive.py tests/test_segment_beam.py tests/test_stack_beam_equivalence.py
```
