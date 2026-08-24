"""Can the optimal first action be predicted from (input, output)?

This is the cheapest test of #33's premise. #33 proposes a policy head that
orders BFS expansion; if the first action is not predictable beyond what the
type mask already gives away, ordering will not work either -- and that is
worth knowing before building the search hybrid.

`deep-corpus-v1` stores `optimal_first_action_names` per task, certified by
complete BFS expansion and marked complete on 85% of tasks. So the label
already exists; nothing here needs search.

The metric is **ordering**, not accuracy. A policy head's job is to make BFS
try the right op sooner, so what matters is how far down the predicted ranking
the first optimal action sits. Rank 1 means the search never expands a wrong
branch at that level; rank 10 means it expands nine.

Predictors form a ladder, each strictly more informed than the last:

    uniform      no information at all
    type mask    #33's claimed win, free and exact
    marginal     which actions are common, ignoring the task
    conditional  a lookup table on (input type, output type) -- no learning
    learned      a byte-level encoder over the serialized pair, as #33 proposes

The decisive comparison is **learned against conditional**. If a lookup table
on two type fields matches a trained encoder, then types are the whole signal
and #33's encoder is buying nothing that a dict does not.

    uv run python experiments/policy_probe.py
"""

from __future__ import annotations

import gzip
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_CORPUS = Path("wandering_light/training/data/deep_corpus_v1")
SPLITS = ("discovery", "validation", "test")


def resolve_corpus(argv: list[str]) -> Path:
    """Locate the corpus payload.

    Only `manifest.json` is tracked (#40); the payload lives on the Hub at
    `abhishekraok/wandering-light-deep-corpus-v1`, revision `fd22029`. So a
    fresh checkout will not have it, and a git worktree will not either even
    when the main checkout does.
    """
    candidates = [Path(argv[1])] if len(argv) > 1 else []
    candidates.append(DEFAULT_CORPUS)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / DEFAULT_CORPUS)
    for candidate in candidates:
        if (candidate / "discovery.jsonl.gz").exists():
            return candidate
    raise SystemExit(
        "deep-corpus-v1 payload not found. Pass its directory as the first "
        "argument, or fetch it from the Hub:\n"
        "  abhishekraok/wandering-light-deep-corpus-v1 (revision fd22029)\n"
        f"Looked in: {', '.join(str(c) for c in candidates[:3])}"
    )


def load(corpus: Path, split: str) -> list[dict]:
    """Tasks with a *complete* optimal-action set.

    An incomplete set means the certifier could not prove it had found every
    optimal first action, so a prediction outside it might still be optimal and
    would be scored wrong. Those tasks are dropped rather than mislabelled.
    """
    rows = []
    with gzip.open(corpus / f"{split}.jsonl.gz", "rt") as handle:
        for line in handle:
            row = json.loads(line)
            if row["metadata"]["optimal_first_actions_complete"]:
                rows.append(row)
    return rows


def basis_names() -> list[str]:
    from wandering_light.basis_set import load_basis_set

    return [f.name for f in load_basis_set("wl-core-v1").to_function_def_set()]


def type_table() -> dict[str, str]:
    from wandering_light.basis_set import load_basis_set

    return {
        f.name: f.input_type for f in load_basis_set("wl-core-v1").to_function_def_set()
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def rank_of_first_optimal(scores: dict[str, float], optimal: set[str]) -> int:
    """Position of the first optimal action in the predicted ordering, 1-based.

    Ties are broken pessimistically -- an action tied with an optimal one is
    counted as tried first -- so a predictor cannot score well by returning a
    flat distribution.
    """
    # `kv[0] in optimal` is True for optimal actions, and True sorts after
    # False, so among equal scores every non-optimal action is tried first.
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0] in optimal))
    for position, (name, _) in enumerate(ordered, start=1):
        if name in optimal:
            return position
    return len(ordered) + 1


def evaluate(predict, tasks: list[dict], label: str) -> dict:
    ranks, hits = [], 0
    by_distance: dict[int, list[int]] = defaultdict(list)
    for task in tasks:
        optimal = set(task["metadata"]["optimal_first_action_names"])
        rank = rank_of_first_optimal(predict(task), optimal)
        ranks.append(rank)
        hits += rank == 1
        by_distance[task["metadata"]["certified_distance"]].append(rank)
    ordered = sorted(ranks)
    return {
        "predictor": label,
        "top1": hits / len(tasks),
        "mean_rank": sum(ranks) / len(ranks),
        "median_rank": ordered[len(ordered) // 2],
        "p90_rank": ordered[int(len(ordered) * 0.9)],
        "by_distance": {d: sum(v) / len(v) for d, v in sorted(by_distance.items())},
    }


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------


def make_uniform(names):
    return lambda task: dict.fromkeys(names, 0.0)


def make_type_mask(names, in_type_of):
    """Uniform over actions whose input type matches the task's input type."""

    def predict(task):
        want = task["metadata"]["input_type"]
        return {n: (1.0 if in_type_of[n] == want else 0.0) for n in names}

    return predict


def make_marginal(names, train):
    counts = Counter()
    for task in train:
        for name in task["metadata"]["optimal_first_action_names"]:
            counts[name] += 1
    total = sum(counts.values()) or 1
    table = {n: counts[n] / total for n in names}
    return lambda task: dict(table)


def make_conditional(names, train, in_type_of):
    """P(action | input type, output type), backing off to input type alone.

    No learning: a dict. This is the baseline a trained encoder has to beat to
    justify itself.
    """
    joint: dict[tuple[str, str], Counter] = defaultdict(Counter)
    backoff: dict[str, Counter] = defaultdict(Counter)
    for task in train:
        meta = task["metadata"]
        key = (meta["input_type"], meta["output_type"])
        for name in meta["optimal_first_action_names"]:
            joint[key][name] += 1
            backoff[meta["input_type"]][name] += 1

    def predict(task):
        meta = task["metadata"]
        counts = joint.get((meta["input_type"], meta["output_type"]))
        if not counts:
            counts = backoff.get(meta["input_type"], Counter())
        total = sum(counts.values()) or 1
        mask_want = meta["input_type"]
        return {
            n: (counts[n] / total) + (1e-6 if in_type_of[n] == mask_want else 0.0)
            for n in names
        }

    return predict


def train_learned(
    names, train, valid, in_type_of, epochs: int = 12, seed: int = 0, conv: bool = False
):
    """A byte-level encoder over the serialized (input, output) pair.

    Deliberately small and CPU-only: the point is to find out whether the
    signal exists, not to squeeze it. If a model this size cannot beat a lookup
    table, a larger one is unlikely to change the conclusion qualitatively --
    and if it can, that is the finding.
    """
    import torch
    from torch import nn

    torch.manual_seed(seed)
    index = {n: i for i, n in enumerate(names)}
    max_len = 256

    def encode(task) -> list[int]:
        text = f"{task['input']}\x00{task['output']}"
        raw = text.encode("utf-8", "replace")[:max_len]
        return list(raw) + [0] * (max_len - len(raw))

    def batch(tasks):
        x = torch.tensor([encode(t) for t in tasks], dtype=torch.long)
        y = torch.zeros(len(tasks), len(names))
        for row, task in enumerate(tasks):
            for name in task["metadata"]["optimal_first_action_names"]:
                y[row, index[name]] = 1.0
        return x, y / y.sum(dim=1, keepdim=True).clamp(min=1)

    class Encoder(nn.Module):
        """Embed bytes, mean-pool, then a two-layer head over the basis."""

        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(256, 48)
            self.conv = nn.Conv1d(48, 96, kernel_size=5, padding=2) if conv else None
            width = 96 if conv else 48
            self.head = nn.Sequential(
                nn.Linear(width, 256), nn.ReLU(), nn.Linear(256, len(names))
            )

        def forward(self, x):
            e = self.embed(x)
            if self.conv is None:
                return self.head(e.mean(dim=1))
            # local n-gram structure, which mean-pooling destroys
            h = self.conv(e.transpose(1, 2)).relu()
            return self.head(h.max(dim=2).values)

    model = Encoder()
    optimiser = torch.optim.Adam(model.parameters(), lr=2e-3)
    order = list(train)
    rng = random.Random(seed)
    vx, vy = batch(valid)

    best, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        rng.shuffle(order)
        for start in range(0, len(order), 256):
            chunk = order[start : start + 256]
            x, y = batch(chunk)
            loss = -(torch.log_softmax(model(x), dim=1) * y).sum(dim=1).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            vloss = -(torch.log_softmax(model(vx), dim=1) * vy).sum(dim=1).mean().item()
        print(f"    epoch {epoch + 1:>2}  val loss {vloss:.4f}", file=sys.stderr)
        if vloss < best:
            best, best_state = (
                vloss,
                {k: v.clone() for k, v in model.state_dict().items()},
            )
    model.load_state_dict(best_state)
    model.eval()

    def predict(task):
        with torch.no_grad():
            logits = model(torch.tensor([encode(task)], dtype=torch.long))[0]
        want = task["metadata"]["input_type"]
        return {
            n: logits[i].item() + (0.0 if in_type_of[n] == want else -1e9)
            for i, n in enumerate(names)
        }

    return predict


def main() -> None:
    corpus = resolve_corpus(sys.argv)
    names = basis_names()
    in_type_of = type_table()
    train, valid, test = (load(corpus, s) for s in SPLITS)
    print(f"corpus: {corpus}", file=sys.stderr)
    print(
        f"{len(train)} train / {len(valid)} val / {len(test)} test tasks "
        f"with complete labels; basis {len(names)}",
        file=sys.stderr,
    )

    mask_sizes = [
        sum(1 for n in names if in_type_of[n] == t["metadata"]["input_type"])
        for t in test
    ]
    print(
        f"mean type-compatible actions: {sum(mask_sizes) / len(mask_sizes):.2f} "
        f"of {len(names)}  ({math.log2(len(names)):.2f} bits -> "
        f"{sum(math.log2(m) for m in mask_sizes) / len(mask_sizes):.2f})",
        file=sys.stderr,
    )

    rows = [
        evaluate(make_uniform(names), test, "uniform over all"),
        evaluate(make_type_mask(names, in_type_of), test, "type mask"),
        evaluate(make_marginal(names, train), test, "marginal frequency"),
        evaluate(
            make_conditional(names, train, in_type_of), test, "conditional lookup"
        ),
    ]
    print("  training the mean-pool encoder...", file=sys.stderr)
    rows.append(
        evaluate(
            train_learned(names, train, valid, in_type_of),
            test,
            "learned (mean-pool)",
        )
    )
    print("  training the conv encoder...", file=sys.stderr)
    rows.append(
        evaluate(
            train_learned(names, train, valid, in_type_of, conv=True),
            test,
            "learned (conv)",
        )
    )

    print(f"\nOne-step policy probe -- {len(test)} test tasks\n")
    print(f"{'predictor':<22} {'top-1':>8} {'mean rank':>10} {'median':>8} {'p90':>6}")
    for row in rows:
        print(
            f"{row['predictor']:<22} {row['top1']:>7.1%} {row['mean_rank']:>10.2f} "
            f"{row['median_rank']:>8} {row['p90_rank']:>6}"
        )
    distances = sorted(rows[0]["by_distance"])
    print("\nMean rank by certified distance\n")
    print(f"{'predictor':<22}" + "".join(f"{('d' + str(d)):>8}" for d in distances))
    for row in rows:
        cells = "".join(
            f"{row['by_distance'].get(d, float('nan')):>8.2f}" for d in distances
        )
        print(f"{row['predictor']:<22}{cells}")
    Path("policy_probe.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
