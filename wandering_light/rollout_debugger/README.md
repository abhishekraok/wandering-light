# Rollout debugger

Watch the materialized-state rollout solver think, one expansion at a time.

```bash
uv run python -m wandering_light.rollout_debugger.cli serve --port 8000
```

Then open <http://127.0.0.1:8000>. The private issue-55 checkpoint is fetched
from the Hub on first use and cached, so `HF_TOKEN` or an `hf auth login` token
must be available. `--checkpoint DIR` uses a local one instead.

To dump one trace without the UI:

```bash
uv run python -m wandering_light.rollout_debugger.cli trace --distance 5 --seed 7
```

## Keys

| replay | | drive | |
|---|---|---|---|
| `↑ ↓` | expansion | `a s d f g h j k l` | apply the rank 1–9 action |
| `→ ←` | layer | `w` | undo one step |
| `1`–`9` | select beam node | `m` | back to replay |
| `space` | run / pause | | |
| `r` | reset | | |
| `v` | raw logits, whole basis | | |
| `e` | exact model input | | |
| `x` | show incompatible actions | | |
| `t` | witness overlay | | |
| `m` | drive mode | | |
| `n` | new task | | |

## What makes it trustworthy

**It shows the production search.** `RolloutPredictor.search_with_cost` takes an
optional `RolloutObserver`. Every hook is told what the search already decided;
nothing an observer returns is read, so a recorded run and an unrecorded one are
the same run. `tests/test_rollout_trace.py` asserts that across solved,
alternate-optimal, depth-terminated and budget-terminated searches, and
`tests/test_rollout_debugger_replay.py` asserts it again against the real
checkpoint (`-m network`).

**The encoding is the real one.** The encoding panel calls the policy's own
`pair_batch`, so byte truncation, separators, padding and tensor shapes are what
the model was actually fed.

**The witness cannot reach the search.** It is read from the corpus record after
the search finishes, and the overlay is computed from the finished trace.

## Ground truth

`t` marks the certified witness's next action wherever there is one: in the
beam (the state on the witness path), in the actions tried, and in the policy's
ranking, with a line saying where the policy put it -- *witness step 1:
`tuple_to_index_dict`, ranked 4 of 6 at 0.7%*. That is the per-step version of
the rank the policy probes report in aggregate.

It is deliberately absent off the witness path. A witness is ground truth only
along its own trajectory; from a state it never visits, no action is "the
correct one", and marking one would invent an answer. The page says *off the
witness path -- it says nothing here* instead. The same rule holds while
driving: step off the witness and the marker disappears rather than nagging.

**The page itself is tested.** `tests/test_rollout_debugger_browser.py` drives
it in headless Chromium -- keys, panels, drive steps, painted geometry, console
errors. `playwright` is a dev extra, so these run under a plain `uv run pytest`;
the browser binary is a separate download and the tests skip themselves without
it:

```bash
uv run playwright install chromium
```

`tests/ui/browser_check.py` runs the same checks as a script against a live
server and writes screenshots, which is the faster loop while changing the page:

```bash
uv run python tests/ui/browser_check.py http://127.0.0.1:8000 shots
```

## Why it is arranged this way

A session runs the whole search eagerly and ships the trace to the browser
whole -- 456 KiB gzipped at beam 64 and depth 8. Stepping, changing layers and
selecting nodes are then index moves against data already in the page, with no
round-trip, which is what keeps navigation instant. Only three things need the
server: building a session, inspecting one node on demand, and taking a drive
step; each is a few milliseconds.

Raw logit rows -- 118 floats per node per layer -- are the one thing large
enough to be worth excluding, so they are fetched per node when the panel opens
rather than shipped with every session.

Node state is recomputed from its path rather than cached. A path plus the task
fully describes a node, so there is no session store to grow or go stale.

## The 2D maps

Depth and cumulative log-probability are honest axes but say nothing about
whether a state is getting *closer*. Both maps instead use the policy's own
pooled `(state, target)` encoding -- what the action head reads.

**Replay** projects the retained beam onto the first two principal components
of the visited states, anchored at φ(target, target), where a solved state
lands. PCA is fitted per session, so the axes describe one task's search and
are not comparable across tasks.

**Drive** uses a fixed frame instead: `x` is progress along
φ(start, target) → φ(target, target), where 0 is the input and 1 is solved, and
`y` is how far off that line the state sits. PCA would be wrong here, because
refitting as the walk grows moves points for reasons that have nothing to do
with the step just taken. Two anchors cannot rotate, so a step moves the dot
only when the state moved.

The map is the only feature needing more than the `ActionScorer` interface. A
search driven by a scorer with no encoder is still fully debuggable; the map is
simply omitted.

## Extending it

The debugger depends on `ActionScorer.score_actions_batch`, the same seam the
CLI solver uses, so a larger policy or an out-of-process backend needs no change
here.
