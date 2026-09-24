"""Can the policy's encoder name the op between two states? (inverse-dynamics-probe #166)

inner-frames (#160) campaign 2 showed the exit-frame oracle both a body's frame
and the frame it must end on, and the policy still paid half the world arm's
deficit on a last op the two determine uniquely. `StackPairPolicy` reads each
example as one token sequence (the state bottom-to-top, a separator, the target)
through one width-5 convolution max-pooled over positions, so nothing lines up
item k of the state with item k of the target. This trains three encoders on the
one-step problem itself, with nothing authored:

- **triples** (s, op, s'): s is a product state over a task's shown examples
  from replaying a train witness (a machine state with no open mark, or a
  defined frame), op is a legal op other than `[` and `]` applied to every
  example, s' is the result. The label is the set of ops that give s', since
  some ops collide. Only states whose encoding is never truncated are used, so
  a failure cannot be the token cap.
- **arms**: ``current``, `StackPairPolicy` unchanged with s' as its target;
  ``difference``, the same with position-specific tokens saying, for each of
  the top 8 items, whether s' kept, changed, added or removed it; ``aligned``,
  an encoder that embeds each value with the same convolution and compares the
  state's and the target's items pairwise from the top before pooling.

Stages, each skipped where its output exists::

    uv run python experiments/inverse_dynamics.py data
    uv run python experiments/inverse_dynamics.py probe --arm current --device cuda
    uv run python experiments/inverse_dynamics.py train --arm current --size 1000000 --device cuda
    uv run python experiments/inverse_dynamics.py report
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

import experiments.compute_optimal_mark28 as co  # noqa: E402
import experiments.world_arm as wa  # noqa: E402
from experiments.frame_gate import (  # noqa: E402
    STRUCTURAL,
    eval_witnesses,
    step_classes,
)
from experiments.frame_rows import example_walks, level_exits  # noqa: E402
from wandering_light.rollout_policy import PolicyArchitecture  # noqa: E402
from wandering_light.stack_beam import step_all_quiet  # noqa: E402
from wandering_light.stack_machine import Lst, Mark  # noqa: E402
from wandering_light.stack_pbe import OPS  # noqa: E402
from wandering_light.stack_policy import (  # noqa: E402
    OP_INDEX,
    OP_NAMES,
    PAD,
    SEP,
    STATE_CAP,
    TARGET_CAP,
    VOCAB,
    StackPairPolicy,
    _encode_value,
    collate,
    encode_stack,
    item_array,
)

RUN_DIR = REPO / "checkpoints" / "inverse-dynamics-20260923"
ARMS = ("current", "difference", "aligned")
SIZES = (10_000, 100_000, 1_000_000)
ARCH = PolicyArchitecture(*co.ARMS["1m"]["dims"])
MOVES = tuple(op for op in OPS if op.name not in STRUCTURAL)
TOP = 8  # items compared from the top, in `difference` and `aligned`
EQUAL, CHANGED, ADDED, REMOVED = range(4)
FLAG_BASE = VOCAB  # `difference`'s extra tokens: FLAG_BASE + 4 * position + flag
VALUE_CAP = 64  # tokens per value in `aligned`
TEST_TRIPLES, VALIDATION_TRIPLES = 20_000, 5_000
BATCH = 256
PROBE_LRS = (2.5e-4, 1e-3, 4e-3)
PROBE_SIZE = 100_000
MAX_EPOCHS = {10_000: 60, 100_000: 20, 1_000_000: 6}
PATIENCE = 3
SEED = 0


# ---------------------------------------------------------------------------
# Triples
# ---------------------------------------------------------------------------


def tokens(stack) -> int:
    out: list[int] = []
    for value in stack:
        _encode_value(value, out)
    return len(out)


def source_states(task) -> list[tuple]:
    """Distinct product states along a witness: closed machine states and defined frames."""
    try:
        walks = example_walks(task)
    except ValueError:
        return []
    classes, _ = step_classes(task, walks)
    seen, out = set(), []
    for step in range(len(task.program) + 1):
        machine = tuple(states[step] for states, _views in walks)
        candidates = []
        if not any(isinstance(v, Mark) for stack in machine for v in stack):
            candidates.append(machine)
        if step < len(task.program) and classes[step] == "defined":
            candidates.append(tuple(views[step][-1][3] for _s, views in walks))
        for product in candidates:
            if all(tokens(s) <= STATE_CAP for s in product) and product not in seen:
                seen.add(product)
                out.append(product)
    return out


def successors(product) -> list[dict[str, Any]]:
    """Every legal move from ``product``, labelled with every move that reaches the same state."""
    reached: dict[tuple, list[int]] = defaultdict(list)
    for op in MOVES:
        after = step_all_quiet(product, op)[0]
        if after is not None and all(tokens(s) <= TARGET_CAP for s in after):
            reached[after].append(OP_INDEX[op.name])
    return [
        {"s": product, "t": after, "op": op, "set": tuple(ops)}
        for after, ops in reached.items()
        for op in ops
    ]


def triples(tasks, count: int, rng: random.Random, log) -> list[dict[str, Any]]:
    """At least ``count`` triples, every legal move of each sampled state, in sampled order."""
    pool = [product for task in tasks for product in source_states(task)]
    log(f"{len(pool)} source states from {len(tasks)} tasks")
    rng.shuffle(pool)
    out: list[dict[str, Any]] = []
    for product in pool:
        if len(out) >= count:
            break
        out.extend(successors(product))
    return out


def last_op_states() -> list[dict[str, Any]]:
    """Campaign 2's v4 eval last-op states: frame to exit, the witness op as label."""
    out = []
    for task in eval_witnesses("stack_pbe_v4"):
        walks = example_walks(task)
        classes, _ = step_classes(task, walks)
        exits = [level_exits(task.program, views) for _s, views in walks]
        prog = task.program
        for step, name in enumerate(prog):
            if classes[step] != "defined" or name in STRUCTURAL:
                continue
            if not (step + 1 < len(prog) and prog[step + 1] == "]"):
                continue
            frames = tuple(views[step][-1][3] for _s, views in walks)
            ends = [e[step][-1] if e.get(step) else None for e in exits]
            if any(x is None for x in ends):
                continue
            ends = tuple(tuple(x) for x in ends)
            same = tuple(
                OP_INDEX[op.name]
                for op in MOVES
                if step_all_quiet(frames, op)[0] == ends
            )
            if OP_INDEX[name] not in same:
                continue
            out.append({"s": frames, "t": ends, "op": OP_INDEX[name], "set": same})
    return out


def data(run_dir: Path, log) -> Path:
    path = run_dir / "triples.pkl"
    if path.exists():
        return path
    fit, held = co.split_tasks(wa.configure())
    rng = random.Random(SEED)
    train = triples(fit, max(SIZES), rng, log)
    held_even = [t for i, t in enumerate(held) if i % 2 == 0]
    held_odd = [t for i, t in enumerate(held) if i % 2 == 1]
    out = {
        "train": train,
        "validation": triples(held_even, VALIDATION_TRIPLES, rng, log),
        "test": triples(held_odd, TEST_TRIPLES, rng, log),
        "last_op": last_op_states(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(out, handle)
    log("RESULT data: " + ", ".join(f"{k} {len(v)}" for k, v in out.items()))
    return path


def load_triples(run_dir: Path) -> dict[str, list]:
    with open(run_dir / "triples.pkl", "rb") as handle:
        return pickle.load(handle)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def flags(s, t) -> list[int]:
    """Per position from the top: kept, changed, added or removed."""
    out = []
    for k in range(TOP):
        a = s[-1 - k] if k < len(s) else None
        b = t[-1 - k] if k < len(t) else None
        if a is None and b is None:
            continue
        flag = (
            ADDED
            if a is None
            else REMOVED
            if b is None
            else EQUAL
            if a == b
            else CHANGED
        )
        out.append(FLAG_BASE + 4 * k + flag)
    return out


def sequence_items(triple, arm: str) -> np.ndarray:
    items = []
    for s, t in zip(triple["s"], triple["t"], strict=True):
        item = [*encode_stack(s, STATE_CAP), SEP, *encode_stack(t, TARGET_CAP)]
        if arm == "difference":
            item += [SEP, *flags(s, t)]
        items.append(item)
    return item_array(items)


def value_tokens(value) -> list[int]:
    out: list[int] = []
    _encode_value(value, out)
    return out[:VALUE_CAP]


def aligned_block(triple) -> dict[str, np.ndarray]:
    """Per example: the top items of s and s' as token rows, and both depths."""
    rows, depth = [], []
    for s, t in zip(triple["s"], triple["t"], strict=True):
        for stack in (s, t):
            for k in range(TOP):
                rows.append(value_tokens(stack[-1 - k]) if k < len(stack) else [])
        depth.append((min(len(s), 31), min(len(t), 31)))
    width = max(1, *(len(r) for r in rows))
    block = np.full((len(rows), width), PAD, dtype=np.int16)
    for i, r in enumerate(rows):
        block[i, : len(r)] = r
    return {"values": block, "depth": np.array(depth, dtype=np.int64)}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class DifferencePolicy(StackPairPolicy):
    """`StackPairPolicy` with room in its vocabulary for the difference tokens."""

    def __init__(self, architecture: PolicyArchitecture) -> None:
        super().__init__(architecture)
        self.embedding = nn.Embedding(
            FLAG_BASE + 4 * TOP, architecture.byte_embedding, padding_idx=PAD
        )


class AlignedPolicy(nn.Module):
    """Each value embedded by one convolution; s and s' compared item by item from the top."""

    def __init__(self, architecture: PolicyArchitecture) -> None:
        super().__init__()
        e, c, h = (
            architecture.byte_embedding,
            architecture.conv_channels,
            architecture.hidden,
        )
        self.embedding = nn.Embedding(VOCAB, e, padding_idx=PAD)
        self.convolution = nn.Conv1d(e, c, kernel_size=5, padding=2)
        self.absent = nn.Parameter(torch.zeros(c))
        self.position = nn.Embedding(TOP, c)
        self.depth = nn.Embedding(32, c)
        self.pair = nn.Sequential(nn.Linear(5 * c, c), nn.ReLU())
        self.example = nn.Sequential(nn.Linear(TOP * c + 2 * c, h), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, len(OP_NAMES))
        )

    def forward(self, values: torch.Tensor, depth: torch.Tensor, counts: torch.Tensor):
        real = values != PAD
        embedded = self.embedding(values) * real.unsqueeze(-1)
        convolved = self.convolution(embedded.transpose(1, 2)).relu()
        convolved = convolved.masked_fill(~real.unsqueeze(1), -1e9).max(dim=2).values
        present = real.any(dim=1, keepdim=True)
        encoded = torch.where(present, convolved, self.absent)
        examples = encoded.view(-1, 2, TOP, encoded.shape[-1])
        s, t = examples[:, 0], examples[:, 1]
        pos = self.position.weight.unsqueeze(0).expand_as(s)
        pairs = self.pair(torch.cat([s, t, s - t, s * t, pos], dim=-1))
        d = torch.cat([self.depth(depth[:, 0]), self.depth(depth[:, 1])], dim=-1)
        per_example = self.example(torch.cat([pairs.flatten(1), d], dim=-1))
        dense = nn.utils.rnn.pad_sequence(
            per_example.split(counts.tolist()), batch_first=True
        )
        mask = (
            torch.arange(dense.shape[1], device=dense.device)[None, :] < counts[:, None]
        ).unsqueeze(-1)
        mean = (dense * mask).sum(1) / mask.sum(1).clamp(min=1)
        peak = dense.masked_fill(~mask, -1e9).max(1).values
        return self.head(torch.cat([mean, peak], dim=-1))


def build(arm: str) -> nn.Module:
    if arm == "current":
        return StackPairPolicy(ARCH)
    if arm == "difference":
        return DifferencePolicy(ARCH)
    return AlignedPolicy(ARCH)


def featurise(rows: list[dict], arm: str) -> list:
    if arm == "aligned":
        return [aligned_block(r) for r in rows]
    return [sequence_items(r, arm) for r in rows]


def logits(model: nn.Module, features: list, arm: str, device) -> torch.Tensor:
    if arm != "aligned":
        return model(*collate(features, device))
    width = max(f["values"].shape[1] for f in features)
    values = np.full(
        (sum(f["values"].shape[0] for f in features), width), PAD, dtype=np.int64
    )
    at = 0
    for f in features:
        n, w = f["values"].shape
        values[at : at + n, :w] = f["values"]
        at += n
    depth = np.concatenate([f["depth"] for f in features])
    counts = torch.tensor([f["depth"].shape[0] for f in features], device=device)
    return model(
        torch.from_numpy(values).to(device), torch.from_numpy(depth).to(device), counts
    )


def set_nll(out: torch.Tensor, sets: list[tuple[int, ...]]) -> torch.Tensor:
    """-log of the probability the model puts on the label set."""
    logp = out.log_softmax(dim=-1)
    mask = torch.zeros_like(logp, dtype=torch.bool)
    for i, s in enumerate(sets):
        mask[i, list(s)] = True
    return -(logp.masked_fill(~mask, -1e9).logsumexp(dim=-1))


def evaluate(model, features, rows, arm, device) -> dict[str, Any]:
    model.eval()
    losses, hits = [], []
    with torch.inference_mode():
        for start in range(0, len(rows), 1024):
            chunk = rows[start : start + 1024]
            out = logits(model, features[start : start + 1024], arm, device)
            losses += set_nll(out, [r["set"] for r in chunk]).tolist()
            top = out.argmax(dim=-1).tolist()
            hits += [p in r["set"] for p, r in zip(top, chunk, strict=True)]
    return {
        "loss": statistics.fmean(losses),
        "set_accuracy": statistics.fmean(hits),
        "hits": hits,
    }


def strata(rows: list[dict], hits: list[bool]) -> dict[str, dict[str, float]]:
    """Set-accuracy by stack depth and by what the move changed."""
    groups: dict[str, list[bool]] = defaultdict(list)
    for row, hit in zip(rows, hits, strict=True):
        s, t = row["s"][0], row["t"][0]
        depth = len(s)
        groups[
            f"depth {'0-1' if depth <= 1 else '2-3' if depth <= 3 else '4-7' if depth <= 7 else '8+'}"
        ].append(hit)
        f = [x - FLAG_BASE for x in flags(s, t)]
        changed = [k // 4 for k in f if k % 4 != EQUAL]
        lists = any(
            isinstance(v, Lst)
            for k in changed
            for v in (
                s[-1 - k] if k < len(s) else None,
                t[-1 - k] if k < len(t) else None,
            )
        )
        kind = (
            "length change"
            if len(s) != len(t)
            else "list changed"
            if lists
            else "scalar changed"
        )
        groups[kind].append(hit)
        groups[f"op {OP_NAMES[row['op']]}"].append(hit)
    return {
        k: {"n": len(v), "set_accuracy": statistics.fmean(v)}
        for k, v in sorted(groups.items())
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    run_dir: Path,
    arm: str,
    size: int,
    lr: float,
    max_epochs: int,
    device: str,
    log,
    tag: str,
) -> dict:
    out_path = run_dir / "runs" / f"{tag}.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    split = load_triples(run_dir)
    rows = split["train"][:size]
    features = featurise(rows, arm)
    validation = featurise(split["validation"], arm)
    model = build(arm).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    history, best, best_state, stale = [], float("inf"), None, 0
    order = list(range(len(rows)))
    started = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        rng.shuffle(order)
        for start in range(0, len(order), BATCH):
            batch = order[start : start + BATCH]
            out = logits(model, [features[i] for i in batch], arm, device)
            loss = set_nll(out, [rows[i]["set"] for i in batch]).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        v = evaluate(model, validation, split["validation"], arm, device)
        history.append(
            {
                "epoch": epoch,
                "validation_loss": v["loss"],
                "validation_set_accuracy": v["set_accuracy"],
            }
        )
        log(
            f"{tag} e{epoch}: loss {v['loss']:.4f} set-acc {v['set_accuracy']:.4f} ({time.time() - started:.0f}s)"
        )
        if v["loss"] < best:
            best, stale = v["loss"], 0
            best_state = {k: x.detach().clone() for k, x in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    result: dict[str, Any] = {
        "arm": arm,
        "size": len(rows),
        "lr": lr,
        "history": history,
        "best_epoch": min(history, key=lambda r: r["validation_loss"])["epoch"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "seconds": round(time.time() - started, 1),
    }
    for name in ("test", "last_op"):
        e = evaluate(model, featurise(split[name], arm), split[name], arm, device)
        result[name] = {
            "loss": e["loss"],
            "set_accuracy": e["set_accuracy"],
            "n": len(split[name]),
        }
        if name == "test":
            result["test_strata"] = strata(split[name], e["hits"])
        else:
            unique = [
                h
                for r, h in zip(split[name], e["hits"], strict=True)
                if len(r["set"]) == 1
            ]
            result[name]["unique"] = {
                "n": len(unique),
                "set_accuracy": statistics.fmean(unique) if unique else None,
            }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1) + "\n")
    log(
        f"RESULT {tag}: test {result['test']['set_accuracy']:.4f} last-op {result['last_op']['set_accuracy']:.4f} "
        f"(unique {result['last_op']['unique']['set_accuracy']})"
    )
    return result


def probe_lr(run_dir: Path, arm: str, device: str, log) -> float:
    """One epoch at 100k per rate; the lowest validation loss wins."""
    path = run_dir / "runs" / f"lr-{arm}.json"
    if path.exists():
        return json.loads(path.read_text())["chosen"]
    losses = {}
    for lr in PROBE_LRS:
        r = train(run_dir, arm, PROBE_SIZE, lr, 1, device, log, f"probe-{arm}-{lr:g}")
        losses[lr] = r["history"][0]["validation_loss"]
    chosen = min(losses, key=losses.get)
    path.write_text(
        json.dumps({"losses": {str(k): v for k, v in losses.items()}, "chosen": chosen})
        + "\n"
    )
    log(f"RESULT lr {arm}: {losses} -> {chosen:g}")
    return chosen


def report(run_dir: Path, log) -> dict:
    rows = {}
    for arm in ARMS:
        for size in SIZES:
            path = run_dir / "runs" / f"{arm}-{size}.json"
            if path.exists():
                rows[f"{arm}-{size}"] = json.loads(path.read_text())
    table = {
        k: {
            "test": v["test"]["set_accuracy"],
            "last_op": v["last_op"]["set_accuracy"],
            "last_op_unique": v["last_op"]["unique"]["set_accuracy"],
            "best_epoch": v["best_epoch"],
        }
        for k, v in rows.items()
    }
    current = rows.get(f"current-{max(SIZES)}")
    reading = None
    if current is not None:
        acc = current["test"]["set_accuracy"]
        reading = (
            "CAN_COMPARE"
            if acc >= 0.97
            else "CANNOT_COMPARE"
            if acc <= 0.90
            else "INCONCLUSIVE"
        )
    out = {"table": table, "reading": reading}
    (run_dir / "report.json").write_text(json.dumps(out, indent=1) + "\n")
    log(f"RESULT report: {reading} {json.dumps(table)}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=("data", "probe", "train", "report"))
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--size", type=int, choices=SIZES)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)

    def log(line: str) -> None:
        stamped = f"{time.strftime('%H:%M:%S')} {line}"
        print(stamped, flush=True)
        with open(args.run_dir / "log.txt", "a") as handle:
            handle.write(stamped + "\n")

    if args.stage == "data":
        data(args.run_dir, log)
    elif args.stage == "probe":
        probe_lr(args.run_dir, args.arm, args.device, log)
    elif args.stage == "train":
        lr = probe_lr(args.run_dir, args.arm, args.device, log)
        train(
            args.run_dir,
            args.arm,
            args.size,
            lr,
            MAX_EPOCHS[args.size],
            args.device,
            log,
            f"{args.arm}-{args.size}",
        )
    else:
        report(args.run_dir, log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
