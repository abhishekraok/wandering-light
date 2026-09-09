# Undirected walks on the stack machine

Does a long uniform-random walk reach past the BFS horizon? On `wl-core-v1` it
does not because the walk drains into scalar types (`experiments/walk_collapse.py`,
STATUS "Undirected expansion drains"). This is the same protocol on the stack
machine, for the two live bases and the guard search uses.

The `executable-26` half was first measured on 2026-08-27 outside the repo and
cited in STATUS as unreproduced; this run replaces that citation.

## Protocol

300 walks per root from the four `experiments/stack_machine_bfs.py` roots
(empty; one list; two ints; list and int). Each step picks uniformly among
legal, non-identity successors — `Op.guard` decides legality, and under
`--guard` a successor `quotation_guard.blocked` rejects is dropped, with
`remaining` = steps left in the walk. True distance is read off an exhaustive
BFS ball to depth 5 around each root, complete layers only, so every recorded
distance is exact and every miss is beyond depth 5. Seeds are fixed (11 for
collapse, 7 for travel).

```
uv run python experiments/walk_collapse.py --stack --basis=executable-26 --guard=off
uv run python experiments/walk_collapse.py --stack --basis=mark-28 --guard=off
uv run python experiments/walk_collapse.py --stack --basis=mark-28 --guard=both
```

Console output is beside this file, one `.txt` per command.

## Result

| | `executable-26` | `mark-28`, no guard | `mark-28`, guard `both` |
|---|---|---|---|
| distinct endpoints, 300 walks of 40 | 300/300, every root | 300/300 | 300/300 |
| inside the depth-5 ball at 6 / 10 / 20 / 40 / 80 steps | 83–87% / 48–57% / 13–17% / ~1% / 0% | 59–66% / 25–34% / 5–9% / 1–2% / 0–2% | 64–73% / 33–34% / 5–9% / 0–3% / 0% |
| stack depth at 40 / 80 steps | 15.5–16.4 / 29.5–30.4 | 23.4–23.8 / 39–41 | 8.3–9.7 / 12.0–13.5 |
| endpoint out-degree | 12.3–12.6 | 24.0–24.9 | 14.6–14.7 |
| top of stack at 40 steps | bool 42–49%, int 24–32%, list 24–27% | **sym 54–58%**, bool 14–16% | **list 76–78%**, bool 12–16% |
| empty-root ball, depth 5 | 9,503 (= #41's pin) | 726,064 | 726,064 |

Three things to read off it.

**The stack machine diffuses; it never contracts.** No basis and no guard
gives fewer than 300 distinct endpoints, and at 80 steps at most 2% of walks
are within 5 of where they started. `wl-core-v1` cannot leave home; this
machine cannot come home.

**Unguarded `mark-28` diffuses into open quotations.** Over half the walks end
with a symbol on top, meaning inside an open `[` where every op is legal and
captured rather than run. That is the branching #70 measured (`b_eff` 23.1
against 13.4) seen from the walk's side, and the 76x larger depth-5 ball.

**The guard changes where the junk goes, not whether it accumulates.** Forcing
marks to close keeps the stack a third as deep, because `]` folds everything
above the mark into one list — and three quarters of endpoints then carry that
list on top. Distinctness and distance are unchanged. A directed mechanism has
to bring the walk back; the guard only stops it wandering off inside a
quotation.

## What this does not measure

Uniform-random policy, so this bounds undirected expansion only; the rows in
STATUS on `gpt-5.6-terra` and the `mark-28` policy are what directed search
reaches. Fuel is ignored, as in `stack_machine_bfs.py`. Distances beyond 5
are not resolved, only known to exceed it.
