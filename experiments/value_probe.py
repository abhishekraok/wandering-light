"""Is distance-to-go learnable? The value half of #33.

#33 proposes two heads: a policy that orders expansion, and a value that
estimates distance to the target. `reports/policy-probe-20260824/` measured the
policy half. This measures the value half -- the one an A*-style best-first
search actually needs, since `f = g + h` is only as good as `h`.

**The labels are free and exact.** A `deep-corpus-v1` task carries a witness
program of length `d` certified shortest. Replay its first `j` ops on the input
to reach `s_j`; the true remaining distance from `s_j` to the target is exactly
`d - j`. If some `p` reached the target from `s_j` in fewer than `d - j` steps,
then `w_1..w_j` followed by `p` would solve the task in fewer than `d`,
contradicting certification. So every prefix of every witness is an exactly
labelled point -- 137,140 of them -- with no search and no bootstrapping.

The metric is **mean absolute error in steps, broken down by true remaining
distance**. The breakdown is the point: an `h` accurate at 2 and useless at 7
buys nothing, because 7 is where search is expensive (BFS spends ~805k
executions at distance 7 and ~4.4M at 8). The bar, stated before looking: `h`
must be within roughly 1 step at remaining distance 6 and beyond, or best-first
will not rescue the search.

Predictors form a ladder, each strictly more informed than the last:

    constant     the train median, which is the L1-optimal constant
    type pair    a dict on (state type, target type) -- no learning
    type + size  the same dict, plus how many items each side carries
    learned      a byte-level encoder over the serialized pair, as #33 proposes

The decisive comparison is **learned against the dicts**. On the policy side a
two-field lookup table matched a trained encoder; if that repeats here, #33's
value head is a much smaller build than the issue assumes.

Two secondary measurements, because on-path MAE alone would mislead:

  * **Sign of the error.** A learned `h` that overestimates is inadmissible, so
    A* stops guaranteeing shortest paths. That is acceptable -- the corpus
    already knows the shortest length and what is wanted is speed -- but the
    bias decides whether greedy best-first or weighted A* is the right frame.
  * **Successor discrimination.** Every training state sits *on* an optimal
    path. Search does not: it spends its time on states that fell off one. So
    the probe also ranks the successors of each test input by `h` and asks
    whether the optimal one comes first. This is out of distribution by
    construction, and it is what best-first actually depends on.

    uv run python experiments/value_probe.py [corpus-dir] [--epochs N] [--limit N]
"""

from __future__ import annotations

import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from policy_probe import resolve_corpus

SPLITS = ("discovery", "validation", "test")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def basis_pieces():
    """The executable palette, indexed by name, plus an executor over it."""
    from wandering_light.basis_set import load_basis_set
    from wandering_light.executor import Executor

    functions = load_basis_set("wl-core-v1").to_function_def_set()
    by_name = {f.name: f for f in functions.functions}
    return functions, by_name, Executor(functions)


def build_examples(corpus: Path, split: str, by_name, executor, limit: int = 0):
    """One example per witness prefix: `(state, target) -> remaining steps`.

    Prefixes `j = 0 .. d-1` give labels `d .. 1`. The full witness (`j = d`) is
    excluded on purpose: its state *is* the target, so a search would have
    settled it with a goal test before ever consulting `h`, and including it
    would credit the predictor for a case it is never asked about.
    """
    from wandering_light.typed_list import TypedList

    examples = []
    with gzip.open(corpus / f"{split}.jsonl.gz", "rt") as handle:
        for count, line in enumerate(handle):
            if limit and count >= limit:
                break
            row = json.loads(line)
            meta = row["metadata"]
            names = row["witness_function_names"]
            distance = meta["certified_distance"]
            state = TypedList.from_str(row["input"])
            state_text = row["input"]
            for step, name in enumerate(names):
                examples.append(
                    {
                        "state": state_text,
                        "target": row["output"],
                        "state_type": state.item_type.__qualname__,
                        "target_type": meta["output_type"],
                        "state_size": len(state),
                        "target_size": meta_output_size(row),
                        "remaining": distance - step,
                        "certified_distance": distance,
                    }
                )
                state = executor.execute(by_name[name], state)
                state_text = state.to_string()
    return examples


def meta_output_size(row: dict) -> int:
    """Item count of the target, read off the serialized form."""
    return len(json.loads(row["output"])["items"])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def median(values: list[float]) -> float:
    """Lower median.

    Under absolute error the optimal constant is the *median*, not the mean --
    every dict predictor here returns one. Using the mean would hand the
    learned model a baseline weaker than the trivial one, which is the whole
    comparison this probe rests on.
    """
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]


def evaluate(predict, examples: list[dict], label: str) -> dict:
    errors, signed = [], []
    by_remaining: dict[int, list[float]] = defaultdict(list)
    by_distance: dict[int, list[float]] = defaultdict(list)
    for example in examples:
        error = predict(example) - example["remaining"]
        signed.append(error)
        errors.append(abs(error))
        by_remaining[example["remaining"]].append(abs(error))
        by_distance[example["certified_distance"]].append(abs(error))
    return {
        "predictor": label,
        "mae": sum(errors) / len(errors),
        "bias": sum(signed) / len(signed),
        "within1": sum(1 for e in errors if e <= 1.0) / len(errors),
        "by_remaining": {r: sum(v) / len(v) for r, v in sorted(by_remaining.items())},
        "by_distance": {d: sum(v) / len(v) for d, v in sorted(by_distance.items())},
    }


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------


def make_constant(train):
    value = median([e["remaining"] for e in train])
    return lambda example: value


def size_bucket(n: int) -> int:
    """Item counts are small; bucket only enough to keep the cells populated."""
    return min(n, 8)


def make_lookup(train, keys):
    """Conditional median over a tuple of raw fields, backing off one at a time.

    No learning: a dict. This is what a trained encoder has to beat to justify
    itself, and on the policy side the equivalent dict was never beaten by
    more than 10%.
    """
    tables = []
    for width in range(len(keys), 0, -1):
        grouped: dict[tuple, list[int]] = defaultdict(list)
        for example in train:
            grouped[keys_of(example, keys[:width])].append(example["remaining"])
        tables.append((keys[:width], {k: median(v) for k, v in grouped.items()}))
    fallback = median([e["remaining"] for e in train])

    def predict(example):
        for fields, table in tables:
            hit = table.get(keys_of(example, fields))
            if hit is not None:
                return hit
        return fallback

    return predict


def keys_of(example: dict, fields: tuple[str, ...]) -> tuple:
    return tuple(
        size_bucket(example[f]) if f.endswith("_size") else example[f] for f in fields
    )


def train_learned(
    train,
    valid,
    epochs: int = 12,
    seed: int = 0,
    conv: bool = False,
    device: str = "cpu",
):
    """A byte-level encoder over the serialized (state, target) pair.

    Deliberately small and CPU-only, matching `policy_probe.py` so the two
    halves of #33 are measured at the same weight class. State and target are
    truncated *separately* -- concatenating first and cutting at the end would
    hide the target behind a long state, and `h` cannot estimate a distance to
    a target it cannot see.

    Trained on L1 to match the metric; an L2 objective would optimise the mean
    while the report scores the median.
    """
    import torch
    from torch import nn

    torch.manual_seed(seed)
    half = 224

    def encode(example) -> list[int]:
        out = []
        for field in ("state", "target"):
            raw = example[field].encode("utf-8", "replace")[:half]
            out.extend(list(raw) + [0] * (half - len(raw)))
        return out

    def batch(items):
        x = torch.tensor([encode(e) for e in items], dtype=torch.long, device=device)
        y = torch.tensor(
            [float(e["remaining"]) for e in items], dtype=torch.float, device=device
        )
        return x, y

    class Encoder(nn.Module):
        """Embed bytes, pool, then a two-layer head onto one scalar."""

        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(256, 48)
            self.conv = nn.Conv1d(48, 96, kernel_size=5, padding=2) if conv else None
            width = 96 if conv else 48
            self.head = nn.Sequential(
                nn.Linear(width, 256), nn.ReLU(), nn.Linear(256, 1)
            )

        def forward(self, x):
            e = self.embed(x)
            if self.conv is None:
                return self.head(e.mean(dim=1)).squeeze(-1)
            # local n-gram structure, which mean-pooling destroys
            h = self.conv(e.transpose(1, 2)).relu()
            return self.head(h.max(dim=2).values).squeeze(-1)

    model = Encoder().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=2e-3)
    order = list(train)
    rng = random.Random(seed)
    valid_batches = [batch(valid[i : i + 1024]) for i in range(0, len(valid), 1024)]

    best, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        rng.shuffle(order)
        for start in range(0, len(order), 256):
            x, y = batch(order[start : start + 256])
            loss = (model(x) - y).abs().mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            total = sum((model(x) - y).abs().sum().item() for x, y in valid_batches)
        vloss = total / len(valid)
        print(f"    epoch {epoch + 1:>2}  val MAE {vloss:.4f}", file=sys.stderr)
        if vloss < best:
            best, best_state = (
                vloss,
                {k: v.clone() for k, v in model.state_dict().items()},
            )
    model.load_state_dict(best_state)
    model.eval()

    def predict_many(items) -> list[float]:
        out = []
        with torch.no_grad():
            for start in range(0, len(items), 1024):
                x, _ = batch(items[start : start + 1024])
                out.extend(model(x).tolist())
        return out

    # Underfitting and absent signal look identical in the test table, and they
    # point at opposite conclusions: one says train a bigger model, the other
    # says stop. Fitting error on data the model has already seen separates
    # them. A subsample keeps the diagnostic cheap.
    sample = rng.sample(train, min(len(train), 16384))
    fitted = predict_many(sample)
    train_mae = sum(
        abs(p - e["remaining"]) for p, e in zip(fitted, sample, strict=True)
    ) / len(sample)
    print(f"    best val MAE {best:.4f}   train MAE {train_mae:.4f}", file=sys.stderr)
    return predict_many, {"val_mae": best, "train_mae": train_mae}


def cached(predict_many, examples):
    """Wrap a batched predictor as the per-example callable `evaluate` wants."""
    table = dict(zip(map(id, examples), predict_many(examples), strict=True))
    return lambda example: table[id(example)]


# ---------------------------------------------------------------------------
# Successor discrimination -- the out-of-distribution check
# ---------------------------------------------------------------------------


def successor_sets(corpus: Path, functions, executor, limit: int = 0):
    """For each test task, the distinct states one op from its input.

    A first action outside `optimal_first_action_names` leads to a state at
    distance >= d, while an optimal one lands at exactly d-1. So ranking these
    successors by `h` asks precisely what best-first search asks. Only tasks
    whose optimal set is certified *complete* are used; otherwise an action
    that is in fact optimal would be scored as a miss.
    """
    from wandering_light.typed_list import TypedList

    palette = list(functions.functions)
    sets = []
    with gzip.open(corpus / "test.jsonl.gz", "rt") as handle:
        for count, line in enumerate(handle):
            if limit and count >= limit:
                break
            row = json.loads(line)
            meta = row["metadata"]
            if not meta["optimal_first_actions_complete"]:
                continue
            state = TypedList.from_str(row["input"])
            optimal = set(meta["optimal_first_action_names"])
            seen: dict[object, dict] = {}
            for function in palette:
                if function.input_type != meta["input_type"]:
                    continue
                try:
                    successor = executor.execute(function, state)
                except Exception:
                    continue
                key = successor.canonical_key()
                entry = seen.get(key)
                if entry is None:
                    entry = seen[key] = {
                        "state": successor.to_string(),
                        "target": row["output"],
                        "state_type": successor.item_type.__qualname__,
                        "target_type": meta["output_type"],
                        "state_size": len(successor),
                        "target_size": meta_output_size(row),
                        # placeholders so `evaluate`-shaped predictors still work
                        "remaining": meta["certified_distance"] - 1,
                        "certified_distance": meta["certified_distance"],
                        "optimal": False,
                    }
                entry["optimal"] |= function.name in optimal
            if len(seen) > 1 and any(e["optimal"] for e in seen.values()):
                sets.append(
                    {
                        "successors": list(seen.values()),
                        "certified_distance": meta["certified_distance"],
                    }
                )
    return sets


def rank_successors(predict, group) -> int:
    """Position of the first optimal successor under ascending `h`, 1-based.

    Ties break pessimistically -- a state tied with an optimal one counts as
    expanded first -- so a flat `h` cannot score well. This mirrors
    `policy_probe.rank_of_first_optimal`, whose earlier version had the
    tie-break backwards and scored a uniform predictor at 100% top-1.
    """
    scored = [(predict(s), s["optimal"]) for s in group["successors"]]
    scored.sort(key=lambda pair: (pair[0], pair[1]))
    for position, (_, is_optimal) in enumerate(scored, start=1):
        if is_optimal:
            return position
    return len(scored) + 1


def evaluate_successors(predict, groups, label: str) -> dict:
    ranks = [rank_successors(predict, g) for g in groups]
    by_distance: dict[int, list[int]] = defaultdict(list)
    for group, rank in zip(groups, ranks, strict=True):
        by_distance[group["certified_distance"]].append(rank)
    return {
        "predictor": label,
        "top1": sum(1 for r in ranks if r == 1) / len(ranks),
        "mean_rank": sum(ranks) / len(ranks),
        "by_distance": {d: sum(v) / len(v) for d, v in sorted(by_distance.items())},
    }


# ---------------------------------------------------------------------------


def print_table(rows, key: str, axis: str, title: str) -> None:
    columns = sorted({c for row in rows for c in row[key]})
    print(f"\n{title}\n")
    print(f"{'predictor':<22}" + "".join(f"{(axis + str(c)):>8}" for c in columns))
    for row in rows:
        cells = "".join(f"{row[key].get(c, float('nan')):>8.2f}" for c in columns)
        print(f"{row['predictor']:<22}{cells}")


def main() -> None:
    argv = [a for a in sys.argv if not a.startswith("--")]
    flags = {a for a in sys.argv if a.startswith("--")}
    epochs = next(
        (int(a.split("=")[1]) for a in flags if a.startswith("--epochs=")), 12
    )
    limit = next((int(a.split("=")[1]) for a in flags if a.startswith("--limit=")), 0)
    device = next((a.split("=")[1] for a in flags if a.startswith("--device=")), "cpu")

    corpus = resolve_corpus(argv)
    functions, by_name, executor = basis_pieces()
    train, valid, test = (
        build_examples(corpus, split, by_name, executor, limit) for split in SPLITS
    )
    print(f"corpus: {corpus}", file=sys.stderr)
    print(
        f"{len(train)} train / {len(valid)} val / {len(test)} test prefix points "
        f"(exactly labelled, no search)",
        file=sys.stderr,
    )

    rows = [
        evaluate(make_constant(train), test, "constant (median)"),
        evaluate(
            make_lookup(train, ("state_type", "target_type")), test, "type pair lookup"
        ),
        evaluate(
            make_lookup(
                train,
                ("state_type", "target_type", "state_size", "target_size"),
            ),
            test,
            "type + size lookup",
        ),
    ]
    learned = {}
    for label, conv in (("learned (mean-pool)", False), ("learned (conv)", True)):
        print(f"  training {label}...", file=sys.stderr)
        learned[label], fit = train_learned(
            train, valid, epochs=epochs, conv=conv, device=device
        )
        row = evaluate(cached(learned[label], test), test, label)
        row.update(fit)
        rows.append(row)

    print(f"\nValue probe -- {len(test)} test prefix points\n")
    print(
        f"{'predictor':<22} {'MAE':>8} {'within 1':>10} {'bias':>8} {'train MAE':>10}"
    )
    for row in rows:
        fitted = f"{row['train_mae']:>10.3f}" if "train_mae" in row else f"{'--':>10}"
        print(
            f"{row['predictor']:<22} {row['mae']:>8.3f} {row['within1']:>9.1%} "
            f"{row['bias']:>8.3f} {fitted}"
        )
    print_table(
        rows,
        "by_remaining",
        "r",
        "MAE by true remaining distance (the axis that matters)",
    )
    print_table(rows, "by_distance", "d", "MAE by the task's certified distance")

    print("\n  building successor sets...", file=sys.stderr)
    groups = successor_sets(corpus, functions, executor, limit)
    flat = [s for g in groups for s in g["successors"]]
    successor_rows = [
        evaluate_successors(make_constant(train), groups, "constant (median)"),
        evaluate_successors(
            make_lookup(train, ("state_type", "target_type")),
            groups,
            "type pair lookup",
        ),
        evaluate_successors(
            make_lookup(
                train, ("state_type", "target_type", "state_size", "target_size")
            ),
            groups,
            "type + size lookup",
        ),
    ]
    for label, predict_many in learned.items():
        successor_rows.append(
            evaluate_successors(cached(predict_many, flat), groups, label)
        )
    print(
        f"\nSuccessor discrimination -- {len(groups)} test tasks, "
        f"{sum(len(g['successors']) for g in groups) / len(groups):.2f} distinct "
        f"successors each\n"
    )
    print(f"{'predictor':<22} {'top-1':>8} {'mean rank':>10}")
    for row in successor_rows:
        print(f"{row['predictor']:<22} {row['top1']:>7.1%} {row['mean_rank']:>10.2f}")
    print_table(
        successor_rows, "by_distance", "d", "Mean successor rank by certified distance"
    )

    Path("value_probe.json").write_text(
        json.dumps({"on_path": rows, "successors": successor_rows}, indent=1)
    )


if __name__ == "__main__":
    main()
