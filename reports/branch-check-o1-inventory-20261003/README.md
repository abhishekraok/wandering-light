# branch-and-check (#164), O1-B continuation inventory

This is an implemented, preregistered diagnostic awaiting independent code
review before archive trials. No inventory result, model or solver change is
claimed. [Definition and Amendment A](preregistration.md),
[initial design review](design-review.md).

Run once from a committed worktree, with outputs under main checkpoints:

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
