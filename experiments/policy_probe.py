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
    learned      a byte-level encoder over the task, as #33 proposes

The decisive comparison is **learned against conditional**. If a lookup table
on two type fields matches a trained encoder, then types are the whole signal
and #33's encoder is buying nothing that a dict does not.

**Two encoders, because the first one could not see the task.** The published
encoder concatenated the two serialized sides and truncated to 256 bytes, which
overruns on 46.0% of `deep-corpus-v1` tasks -- so on nearly half the corpus the
model never saw the end of the target it was being asked to route to. Every
number it produced is a floor on input bandwidth, not on architecture.

The fix is structural rather than a bigger budget. Every basis function is
applied elementwise (`executor.py:118`), and input and output item counts match
on 24,498 of 24,498 discovery tasks, so a task is not one string pair: it is
3-16 *aligned* example pairs, and this is programming by example. `encode_pairs`
encodes each pair on its own, the trunk runs over items in the batch dimension,
and a permutation-invariant pool collapses them to one vector. At 96 bytes per
item, 0.26% of items are cut instead of 46.0% of tasks. `encoder="flat"` keeps
the published model, bit-identical and the same size to within 7%, so the two
can be read against each other in one run rather than across hardware.

**Bits per step.** Alongside rank, each predictor reports

    H = -log2( sum over b in optimal(s, t) of pi(b | s, t) )

over the type-compatible actions. Summing over the optimal *set* rather than one
witness op is deliberate -- a policy should not be punished for preferring an
equally short route. This is a proper scoring rule, it composes over steps, and
the repo reports substrate branching in bits/step already, so the two subtract.
The type mask, which is uniform over what it admits, lands on exactly
`log2(mask width)`, which anchors the column against that figure. Rungs that
already return probabilities are renormalised rather than softmaxed, or a
distribution over [0, 0.1] would flatten and every rung would report the mask
width. `pi` is interpolated with uniform-over-compatible at weight `eps`,
reported at 0, 0.01 and 0.1: a search consumes a policy as an ordering with the
whole compatible set still reachable, so zero mass on an op is a modelling
artefact rather than an infinite cost -- but `eps = 0` is kept as the
calibration diagnostic it is.

Passing `--last` probes the optimal *last* action instead, which brackets the
question from the other end: if the last step is much more predictable than the
first, a policy is worth more running backwards from the target than forwards
from the input.

    uv run python experiments/policy_probe.py [corpus-dir] [--last]
        [--epochs=N] [--device=cuda]
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


def action_field(last: bool) -> tuple[str, str]:
    """Which corpus fields carry the label and its completeness flag.

    The corpus is not symmetric: the first action is stored as both ids and
    names, the last action only as ids. `load` normalises the last-action ids
    into a names field so everything downstream reads one key.
    """
    if last:
        return "optimal_last_action_names", "optimal_last_actions_complete"
    return "optimal_first_action_names", "optimal_first_actions_complete"


def id_to_name() -> dict[str, str]:
    from wandering_light.basis_set import load_basis_set

    return {f.function_id: f.name for f in load_basis_set("wl-core-v1").functions}


def load(
    corpus: Path, split: str, complete_flag: str, last: bool = False
) -> list[dict]:
    """Tasks with a *complete* optimal-action set.

    An incomplete set means the certifier could not prove it had found every
    optimal first action, so a prediction outside it might still be optimal and
    would be scored wrong. Those tasks are dropped rather than mislabelled.
    """
    names_by_id = id_to_name() if last else {}
    rows = []
    with gzip.open(corpus / f"{split}.jsonl.gz", "rt") as handle:
        for line in handle:
            row = json.loads(line)
            meta = row["metadata"]
            if not meta[complete_flag]:
                continue
            if last:
                meta["optimal_last_action_names"] = [
                    names_by_id[i] for i in meta["optimal_last_action_ids"]
                ]
            rows.append(row)
    return rows


def basis_names() -> list[str]:
    from wandering_light.basis_set import load_basis_set

    return [f.name for f in load_basis_set("wl-core-v1").to_function_def_set()]


def type_table(last: bool) -> tuple[dict[str, str], str]:
    """The mask is not symmetric between the two ends of a program.

    A *first* action has to consume the task's input type; a *last* action has
    to produce its output type. Masking on the wrong one would compare every
    predictor against a mask that does not apply to it.
    """
    from wandering_light.basis_set import load_basis_set

    functions = load_basis_set("wl-core-v1").to_function_def_set()
    if last:
        return {f.name: f.output_type for f in functions}, "output_type"
    return {f.name: f.input_type for f in functions}, "input_type"


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


BITS_EPS = (0.0, 0.01, 0.1)


def distribution(
    scores: dict[str, float], compatible: list[str], normalize: str
) -> list[float]:
    """The predictor's scores as a distribution over the type-compatible ops.

    The rungs do not speak the same language: `marginal` and `conditional`
    already return probabilities, the learned encoder returns logits. Softmaxing
    a probability in [0, 0.1] flattens it almost to uniform, which would make
    every rung report the mask width and the column say nothing -- so each is
    given the normalisation its own scores are in.
    """
    values = [scores[n] for n in compatible]
    if normalize == "softmax":
        top = max(values)
        weights = [math.exp(v - top) for v in values]
    else:
        weights = [max(v, 0.0) for v in values]
        if sum(weights) <= 0.0:  # a predictor with nothing to say, e.g. uniform
            weights = [1.0] * len(values)
    total = sum(weights)
    return [w / total for w in weights]


def optimal_mass(
    scores: dict[str, float], optimal: set[str], compatible: list[str], normalize: str
) -> tuple[float, int, int]:
    """Probability on the optimal *set*, plus what it took to get there.

    Summing over the whole optimal set rather than one witness op is
    deliberate: a policy should not be punished for preferring an equally short
    route. Returns `(mass, |optimal & compatible|, |compatible|)` so that any
    interpolation weight can be applied afterwards without rescoring.
    """
    probabilities = distribution(scores, compatible, normalize)
    pairs = zip(compatible, probabilities, strict=True)
    mass = sum(p for n, p in pairs if n in optimal)
    return mass, sum(1 for n in compatible if n in optimal), len(compatible)


def bits(mass: float, hits: int, width: int, eps: float) -> float:
    """`-log2` of the smoothed mass on the optimal set: bits left per step.

    A search never consumes a policy raw -- it consumes it as an ordering with
    the whole type-compatible set still reachable -- so the scored distribution
    is interpolated with uniform-over-compatible at weight `eps`. At `eps = 0`
    this is the raw proper score, which is also the calibration diagnostic: a
    lookup table that never saw the right op pays infinitely.
    """
    smoothed = (1 - eps) * mass + eps * (hits / width if width else 0.0)
    return -math.log2(smoothed) if smoothed > 0 else float("inf")


def summarise_bits(records: list[tuple[float, int, int]]) -> dict:
    """Mean and median bits at each interpolation weight.

    At `eps = 0` the mean is taken over the tasks that are finite and the
    infinite fraction is reported beside it, because one unreachable optimal op
    would otherwise swallow the whole column.
    """
    out = {}
    for eps in BITS_EPS:
        values = [bits(m, h, w, eps) for m, h, w in records]
        finite = [v for v in values if v != float("inf")]
        ordered = sorted(values)
        out[f"{eps:g}"] = {
            "mean": sum(finite) / len(finite) if finite else float("inf"),
            "median": ordered[len(ordered) // 2],
            "unreachable": 1 - len(finite) / len(values),
        }
    return out


def evaluate(
    predict,
    tasks: list[dict],
    label: str,
    names_field: str,
    *,
    names: list[str] | None = None,
    type_of: dict[str, str] | None = None,
    match_key: str | None = None,
    normalize: str = "renorm",
) -> dict:
    """Rank statistics, and bits per step when the type mask is supplied."""
    ranks, hits = [], 0
    by_distance: dict[int, list[int]] = defaultdict(list)
    records: list[tuple[float, int, int]] = []
    for task in tasks:
        optimal = set(task["metadata"][names_field])
        scores = predict(task)
        rank = rank_of_first_optimal(scores, optimal)
        ranks.append(rank)
        hits += rank == 1
        by_distance[task["metadata"]["certified_distance"]].append(rank)
        if names is not None and type_of is not None and match_key is not None:
            want = task["metadata"][match_key]
            compatible = [n for n in names if type_of[n] == want] or list(names)
            records.append(optimal_mass(scores, optimal, compatible, normalize))
    ordered = sorted(ranks)
    row = {
        "predictor": label,
        "top1": hits / len(tasks),
        "mean_rank": sum(ranks) / len(ranks),
        "median_rank": ordered[len(ordered) // 2],
        "p90_rank": ordered[int(len(ordered) * 0.9)],
        "by_distance": {d: sum(v) / len(v) for d, v in sorted(by_distance.items())},
    }
    if records:
        row["bits"] = summarise_bits(records)
        row["mask_width"] = sum(w for _, _, w in records) / len(records)
    return row


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------


def make_uniform(names):
    return lambda task: dict.fromkeys(names, 0.0)


def make_type_mask(names, type_of, match_key):
    """Uniform over actions whose relevant type matches the task's."""

    def predict(task):
        want = task["metadata"][match_key]
        return {n: (1.0 if type_of[n] == want else 0.0) for n in names}

    return predict


def make_marginal(names, train, names_field):
    counts = Counter()
    for task in train:
        for name in task["metadata"][names_field]:
            counts[name] += 1
    total = sum(counts.values()) or 1
    table = {n: counts[n] / total for n in names}
    return lambda task: dict(table)


def make_conditional(names, train, type_of, match_key, names_field):
    """P(action | input type, output type), backing off to input type alone.

    No learning: a dict. This is the baseline a trained encoder has to beat to
    justify itself.
    """
    joint: dict[tuple[str, str], Counter] = defaultdict(Counter)
    backoff: dict[str, Counter] = defaultdict(Counter)
    for task in train:
        meta = task["metadata"]
        key = (meta["input_type"], meta["output_type"])
        for name in meta[names_field]:
            joint[key][name] += 1
            backoff[meta["input_type"]][name] += 1

    def predict(task):
        meta = task["metadata"]
        counts = joint.get((meta["input_type"], meta["output_type"]))
        if not counts:
            counts = backoff.get(meta["input_type"], Counter())
        total = sum(counts.values()) or 1
        mask_want = meta[match_key]
        return {
            n: (counts[n] / total) + (1e-6 if type_of[n] == mask_want else 0.0)
            for n in names
        }

    return predict


# ---------------------------------------------------------------------------
# The learned predictor
# ---------------------------------------------------------------------------

MAX_LEN = 256  # the published flat budget, kept for the baseline row
WIDE_LEN = 1024  # the same encoder with the truncation lifted: 0.22% over
ITEM_CAP = 96  # the pair encoder's byte budget per serialized item
MAX_ITEMS = 16  # deep-corpus-v1 tops out at 16 items per task
PAD, SEP = 256, 257  # outside the byte range, so padding is not a byte value
HIDDEN = 176  # narrowed from 256 to pay for the pair encoder's wider pooling


def type_vocab() -> dict[str, int]:
    """Type name -> embedding row, with 0 reserved for anything unseen."""
    from wandering_light.basis_set import load_basis_set

    functions = load_basis_set("wl-core-v1").to_function_def_set()
    types = {f.input_type for f in functions} | {f.output_type for f in functions}
    return {t: i + 1 for i, t in enumerate(sorted(types))}


def encode_flat(task, max_len: int = MAX_LEN) -> list[int]:
    """The published encoder: one string pair, truncated to `max_len` bytes.

    Retained unchanged at `MAX_LEN` so the pair encoder can be read against the
    model that produced the published table, in the same run and on the same
    device. At `WIDE_LEN` it becomes the control that separates the two things
    the pair encoder changes at once: the architecture is untouched and the
    parameter count is identical -- a convolution and a pool do not know how
    long their input is -- so the only difference from the published row is
    that 46.0% of tasks stop being cut off, and 0.22% do instead.
    """
    text = f"{task['input']}\x00{task['output']}"
    raw = text.encode("utf-8", "replace")[:max_len]
    return list(raw) + [0] * (max_len - len(raw))


def encode_pairs(task) -> tuple[list[list[int]], str, str]:
    """The task as aligned `(input item, output item)` byte sequences.

    Every basis function is applied elementwise (`executor.py:118`), so the
    i-th output item is the program's image of the i-th input item and nothing
    else. A task is therefore 3-16 aligned example pairs, not one string pair,
    and encoding it that way is what removes the truncation: the flat encoder
    loses the tail of 46.0% of `deep-corpus-v1` tasks, while a 96-byte budget
    per item loses 0.26% of items.
    """
    src, dst = json.loads(task["input"]), json.loads(task["output"])
    pairs = []
    aligned = list(zip(src["items"], dst["items"], strict=True))
    for left, right in aligned[:MAX_ITEMS]:
        a = json.dumps(left, separators=(",", ":")).encode("utf-8", "replace")
        b = json.dumps(right, separators=(",", ":")).encode("utf-8", "replace")
        pairs.append([*a[:ITEM_CAP], SEP, *b[:ITEM_CAP]])
    return pairs or [[SEP]], src["type"], dst["type"]


def truncation_rates(tasks) -> dict[str, float]:
    """What fraction of each encoder's input does not fit in its budget."""
    over_task, over_wide, items, over_item, dropped = 0, 0, 0, 0, 0
    for task in tasks:
        text = f"{task['input']}\x00{task['output']}"
        size = len(text.encode("utf-8", "replace"))
        over_task += size > MAX_LEN
        over_wide += size > WIDE_LEN
        src, dst = json.loads(task["input"]), json.loads(task["output"])
        pairs = list(zip(src["items"], dst["items"], strict=True))
        dropped += max(len(pairs) - MAX_ITEMS, 0)
        for left, right in pairs:
            for value in (left, right):
                items += 1
                raw = json.dumps(value, separators=(",", ":"))
                over_item += len(raw.encode("utf-8", "replace")) > ITEM_CAP
    return {
        "flat_tasks_over_budget": over_task / max(len(tasks), 1),
        "wide_tasks_over_budget": over_wide / max(len(tasks), 1),
        "pairs_items_over_budget": over_item / max(items, 1),
        # a whole example lost past MAX_ITEMS is a different failure from a
        # truncated one, and deep-corpus-v1 tops out exactly at the cap
        "pairs_items_dropped": dropped / max(items / 2, 1),
    }


def flat_batch(tasks, max_len: int = MAX_LEN):
    """One row per task: the truncated concatenation, padded to `max_len`."""
    import torch

    rows = [encode_flat(t, max_len) for t in tasks]
    return (torch.tensor(rows, dtype=torch.long),)


def pair_batch(tasks, vocab: dict[str, int]):
    """Items of every task stacked along the batch dimension.

    Only *real* items are handed to the trunk -- `slot` says where each one
    belongs in the (task, item) rectangle that the cross-item pool reads, and
    `item_mask` marks which of those cells were filled. Padding after the trunk
    rather than before it costs one 96-float row per empty cell instead of a
    whole byte sequence, and it makes the pool exact rather than approximate.
    """
    import torch

    encoded = [encode_pairs(t) for t in tasks]
    # Pad to the batch's own extent rather than to the caps: the median item is
    # 8 bytes, so padding every pair out to 2 * ITEM_CAP + 1 would be almost
    # entirely padding.
    width = max(len(s) for pairs, _, _ in encoded for s in pairs)
    seqs, counts, tags = [], [], []
    for pairs, src_type, dst_type in encoded:
        tags.append([vocab.get(src_type, 0), vocab.get(dst_type, 0)])
        counts.append(len(pairs))
        seqs += [seq + [PAD] * (width - len(seq)) for seq in pairs]
    counts = torch.tensor(counts)
    # Items stay grouped by task in `seqs`, so the cross-item pool can recover
    # the rectangle with a split rather than a scatter -- one source of truth
    # for which cells are real, and no index arithmetic to get wrong.
    item_mask = torch.arange(int(counts.max()))[None, :] < counts[:, None]
    return (
        torch.tensor(seqs, dtype=torch.long),
        counts,
        item_mask,
        torch.tensor(tags, dtype=torch.long),
    )


def build_encoder(kind: str, n_names: int, conv: bool, n_types: int):
    """The two encoders, sharing a trunk and differing only in what feeds it.

    Held to within 7% of each other on parameter count: the question is whether
    the model can *see* more, not whether it is bigger. The trunk is
    bit-identical to the published model's; the pair encoder pays for its wider
    pooled input (mean and max across items, plus a type tag per side) by
    narrowing the head from 256 to `HIDDEN`.
    """
    import torch
    from torch import nn

    class Trunk(nn.Module):
        """Embed bytes, then either mean-pool or convolve and max-pool."""

        def __init__(self, vocab_size: int):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, 48)
            self.conv = nn.Conv1d(48, 96, kernel_size=5, padding=2) if conv else None
            self.width = 96 if conv else 48

        def forward(self, x, real=None):
            e = self.embed(x)
            if real is None:  # the flat encoder, whose input is a fixed width
                return e.mean(dim=1) if self.conv is None else self._peak(e)
            # Zeroing padded positions rather than embedding them is what makes
            # an item's encoding independent of what it was batched with: the
            # convolution's own edge padding is zeros, so a trailing PAD and
            # the end of the sequence look the same to it. Otherwise a short
            # item would encode differently next to a long one.
            m = real.unsqueeze(-1).float()
            e = e * m
            if self.conv is None:
                return e.sum(dim=1) / m.sum(dim=1).clamp(min=1)
            return self._peak(e, real)

        def _peak(self, e, real=None):
            # local n-gram structure, which mean-pooling destroys
            h = self.conv(e.transpose(1, 2)).relu()
            if real is not None:
                h = h.masked_fill(~real.unsqueeze(1), -1e9)
            return h.max(dim=2).values

    class FlatEncoder(nn.Module):
        """The published model: one truncated string, two-layer head."""

        def __init__(self):
            super().__init__()
            self.trunk = Trunk(256)
            self.head = nn.Sequential(
                nn.Linear(self.trunk.width, 256), nn.ReLU(), nn.Linear(256, n_names)
            )

        def forward(self, x):
            return self.head(self.trunk(x))

    class PairEncoder(nn.Module):
        """The same trunk over items, pooled across them.

        Items carry no order that matters -- the program acts on each
        independently -- so the cross-item pool is permutation invariant: mean
        for what holds across the examples, max for what any one of them
        witnesses. The type tag per side restores the one field the flat
        encoder read out of the JSON header for free.
        """

        def __init__(self):
            super().__init__()
            self.trunk = Trunk(258)  # 256 byte values, plus PAD and SEP
            self.tags = nn.Embedding(n_types + 1, 16)
            pooled = 2 * self.trunk.width + 32
            self.head = nn.Sequential(
                nn.Linear(pooled, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, n_names)
            )

        def forward(self, x, counts, item_mask, tags):
            v = self.trunk(x, x != PAD)
            dense = nn.utils.rnn.pad_sequence(
                v.split(counts.tolist()), batch_first=True
            )
            m = item_mask.unsqueeze(-1)
            mean = (dense * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
            peak = dense.masked_fill(~m, -1e9).max(dim=1).values
            tag = self.tags(tags).flatten(1)
            return self.head(torch.cat([mean, peak, tag], dim=1))

    return PairEncoder() if kind == "pairs" else FlatEncoder()


def train_learned(
    names,
    train,
    valid,
    type_of,
    match_key,
    names_field,
    epochs: int = 12,
    seed: int = 0,
    conv: bool = False,
    encoder: str = "pairs",
    device: str = "cpu",
):
    """A byte-level encoder over the task, as one string or as aligned pairs.

    Deliberately small: the point is to find out whether the signal exists, not
    to squeeze it. Three arms, differing only in what the trunk is shown:

        flat   the published model -- both sides concatenated, cut at 256 bytes,
               which overruns on 46.0% of tasks
        wide   the same model and the same parameters at 1024 bytes, where
               0.22% overrun: truncation removed, architecture held fixed
        pairs  each aligned example pair encoded on its own, trunk run over
               items in the batch dimension, pooled across them

    `wide` is what makes the result readable. Without it, `pairs` moves two
    things at once and a gain cannot be attributed to either.
    """
    import torch

    torch.manual_seed(seed)
    index = {n: i for i, n in enumerate(names)}
    dev = torch.device(device)
    vocab = type_vocab()

    def make_batch(tasks):
        if encoder == "pairs":
            inputs = pair_batch(tasks, vocab)
        else:
            inputs = flat_batch(tasks, MAX_LEN if encoder == "flat" else WIDE_LEN)
        y = torch.zeros(len(tasks), len(names))
        for row, task in enumerate(tasks):
            for name in task["metadata"][names_field]:
                y[row, index[name]] = 1.0
        y = y / y.sum(dim=1, keepdim=True).clamp(min=1)
        return tuple(t.to(dev) for t in inputs), y.to(dev)

    model = build_encoder(encoder, len(names), conv, len(vocab)).to(dev)
    total = sum(p.numel() for p in model.parameters())
    print(f"    {encoder} encoder: {total} parameters", file=sys.stderr)

    optimiser = torch.optim.Adam(model.parameters(), lr=2e-3)
    order = list(train)
    rng = random.Random(seed)
    # The pair encoder holds one sequence per item, so the whole validation
    # split in one forward is thousands of sequences at once. Chunked, the
    # per-example criterion is the same; only the peak allocation changes.
    validation = [make_batch(valid[i : i + 256]) for i in range(0, len(valid), 256)]

    best, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        rng.shuffle(order)
        for start in range(0, len(order), 256):
            chunk = order[start : start + 256]
            x, y = make_batch(chunk)
            loss = -(torch.log_softmax(model(*x), dim=1) * y).sum(dim=1).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            vloss = sum(
                -(torch.log_softmax(model(*x), dim=1) * y).sum(dim=1).sum().item()
                for x, y in validation
            ) / len(valid)
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
            x, _ = make_batch([task])
            logits = model(*x)[0]
        want = task["metadata"][match_key]
        return {
            n: logits[i].item() + (0.0 if type_of[n] == want else -1e9)
            for i, n in enumerate(names)
        }

    return predict


def baseline_ladder(names, train, test, type_of, match_key, names_field) -> list[dict]:
    """The four unlearned rungs, scored identically wherever they are reported.

    Shared by both probes so the bits/step normalisation cannot drift between
    them: `renorm` for the rungs that already return probabilities, and the
    type mask therefore lands on exactly `log2(mask width)`.
    """
    scoring = {
        "names": names,
        "type_of": type_of,
        "match_key": match_key,
        "normalize": "renorm",
    }
    return [
        evaluate(make_uniform(names), test, "uniform over all", names_field, **scoring),
        evaluate(
            make_type_mask(names, type_of, match_key),
            test,
            "type mask",
            names_field,
            **scoring,
        ),
        evaluate(
            make_marginal(names, train, names_field),
            test,
            "marginal frequency",
            names_field,
            **scoring,
        ),
        evaluate(
            make_conditional(names, train, type_of, match_key, names_field),
            test,
            "conditional lookup",
            names_field,
            **scoring,
        ),
    ]


def bits_columns(row: dict) -> str:
    cells = row.get("bits")
    if not cells:
        return ""
    return "".join(f"{cells[f'{e:g}']['mean']:>9.2f}" for e in BITS_EPS)


def main() -> None:
    argv = [a for a in sys.argv if not a.startswith("--")]
    flags = [a for a in sys.argv if a.startswith("--")]
    last = "--last" in flags
    end = "last" if last else "first"
    epochs = next(
        (int(a.split("=")[1]) for a in flags if a.startswith("--epochs=")), 12
    )
    device = next((a.split("=")[1] for a in flags if a.startswith("--device=")), "cpu")

    corpus = resolve_corpus(argv)
    names = basis_names()
    type_of, match_key = type_table(last)
    names_field, complete_flag = action_field(last)
    train, valid, test = (load(corpus, s, complete_flag, last) for s in SPLITS)
    print(f"corpus: {corpus}   probing the {end} action", file=sys.stderr)
    print(
        f"{len(train)} train / {len(valid)} val / {len(test)} test tasks "
        f"with complete labels; basis {len(names)}",
        file=sys.stderr,
    )

    mask_sizes = [
        sum(1 for n in names if type_of[n] == t["metadata"][match_key]) for t in test
    ]
    print(
        f"mean type-compatible actions: {sum(mask_sizes) / len(mask_sizes):.2f} "
        f"of {len(names)}  ({math.log2(len(names)):.2f} bits -> "
        f"{sum(math.log2(m) for m in mask_sizes if m) / len(mask_sizes):.2f})",
        file=sys.stderr,
    )
    rates = truncation_rates(test)
    print(
        f"over budget on test: flat {rates['flat_tasks_over_budget']:.1%} of "
        f"tasks, wide {rates['wide_tasks_over_budget']:.2%} of tasks, "
        f"pairs {rates['pairs_items_over_budget']:.2%} of items",
        file=sys.stderr,
    )

    rows = baseline_ladder(names, train, test, type_of, match_key, names_field)
    # The two published rungs, then the conv trunk over the wider window and
    # over aligned pairs. Mean-pooling is not run over the wider window: at
    # 1024 bytes three quarters of the positions are padding, and a mean is
    # diluted by them where a max is not.
    arms = (
        ("mean-pool", False, "flat"),
        ("conv", True, "flat"),
        ("conv", True, "wide"),
        ("conv", True, "pairs"),
    )
    for pool, conv, encoder in arms:
        label = f"learned ({pool}, {encoder})"
        print(f"  training {label}...", file=sys.stderr)
        rows.append(
            evaluate(
                train_learned(
                    names,
                    train,
                    valid,
                    type_of,
                    match_key,
                    names_field,
                    epochs=epochs,
                    conv=conv,
                    encoder=encoder,
                    device=device,
                ),
                test,
                label,
                names_field,
                names=names,
                type_of=type_of,
                match_key=match_key,
                normalize="softmax",
            )
        )

    print(f"\n{end.capitalize()}-action policy probe -- {len(test)} test tasks\n")
    header = (
        f"{'predictor':<26} {'top-1':>8} {'mean rank':>10} {'median':>8} {'p90':>6}"
    )
    header += "".join(f"{('H e=' + f'{e:g}'):>9}" for e in BITS_EPS)
    print(header)
    for row in rows:
        print(
            f"{row['predictor']:<26} {row['top1']:>7.1%} {row['mean_rank']:>10.2f} "
            f"{row['median_rank']:>8} {row['p90_rank']:>6}{bits_columns(row)}"
        )
    distances = sorted(rows[0]["by_distance"])
    print("\nMean rank by certified distance\n")
    print(f"{'predictor':<26}" + "".join(f"{('d' + str(d)):>8}" for d in distances))
    for row in rows:
        cells = "".join(
            f"{row['by_distance'].get(d, float('nan')):>8.2f}" for d in distances
        )
        print(f"{row['predictor']:<26}{cells}")
    Path(f"policy_probe_{end}.json").write_text(
        json.dumps({"truncation": rates, "rows": rows}, indent=1)
    )


if __name__ == "__main__":
    main()
