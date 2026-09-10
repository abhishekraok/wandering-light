# Stack debugger

Watch the `mark-28` beam think on a `stack_pbe_v4` eval task, one op at a time.
The stack-machine twin of the [rollout debugger](../rollout_debugger/README.md):
the same instrument over a product of k example stacks instead of one typed list.

```bash
uv run python -m wandering_light.stack_debugger.cli serve --port 8010
```

Then open <http://127.0.0.1:8010>. The policy defaults to the
compute-optimal-mark-28 (#101) `2m-s1` weights where they sit on UltraDesk, else
the W&B model artifact the campaign archived (`--artifact` picks another arm;
`--checkpoint DIR` a local one). The v4 eval payload is fetched from the Hub
against the committed manifest. A first page load runs one search: 0.5–9 s on
CPU for beam 64 × depth 48.

To dump one trace without the UI:

```bash
uv run python -m wandering_light.stack_debugger.cli trace --task 359
uv run python -m wandering_light.stack_debugger.cli trace --behaviour keep_negative --seed 1
```

## Keys

| replay | | drive | | play | |
|---|---|---|---|---|---|
| `↑ ↓` | op tried, across nodes and depths | `a s d f g h j k l` | apply the policy's rank 1–9 | `→ ←` | program step |
| `→` `←` | follow the op at the cursor deeper / back to the parent | `↑ ↓` then `→` or `Enter` | apply any legal op | `p` `o` | witness / found program |
| `1`–`9` `,` `.` `g` | select beam node / the witness node | `←` or `w` | undo | `m` | back |
| `space` | run | `m` | back | | |
| `t` | witness marks | `e` | tokens the model saw | | |
| `x` | masked ops in the ranking | | | | |
| `v` `e` | raw logits / tokens | | | | |
| `m` `p` `o` | drive / play witness / play found | | | | |
| `n` | new task from the #101 cohort | | | | |

## What it shows

**The search itself.** `beam_solve` takes an optional `BeamObserver`; every hook
is told what the search already decided and nothing it returns is read, so a
recorded run and an unrecorded one are the same run
(`tests/test_stack_debugger_trace.py`). A layer is: all 28 ops tried on every
beam node (the machine's verdict per op, with its reason), one policy row per
node, then the ranking's verdict on every successor — kept, already seen,
dropped by the quotation guard (`close` or `type`), or cut below the beam width.

**A state is k stacks.** The inspector draws every shown example's current stack
beside its target. An open `[` is drawn as a dashed group: what follows is
captured as symbols until `]` folds it into one list. Gold rings mark what the
op at the cursor changed.

**The witness is display-only.** It is read from the task after the search
finishes. The strip under the header says at which depth the beam last held the
witness prefix; where a node is on the witness its next op is marked with the
rank the policy gave it. Off the witness path nothing is marked.

**Play mode restates the machine's runners.** For an op that runs a quotation,
`walk.inner_frames` re-states the runner's sequencing one item at a time —
each `map` element, each `fold` accumulation, each `ifte` branch — applying
leaf ops with `step_raw`. Its final stack is checked against the machine's own
result and a disagreement is shown on the frame rather than hidden.

**Size.** A depth-48 trace is 82,796 op attempts. Successor stacks are recorded
for the first example only and truncated; the full product state, the raw
logits and the token stream are one request away (`/api/state`).
