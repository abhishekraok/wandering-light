"""novelty-reduction (#171): P and E heads on the frozen trunk, and RND novelty.

P and E read the policy's pooled ``(s, t)`` features, computed with the policy
frozen, so no gradient reaches the trunk (the stop-gradient is exact). P is
P(reach t within B), fit to ``hits / n``; E is P(the frozen P fails a fresh
n-rollout Wilson check), fit on pairs P did not see. (V is reserved for the
value function, negative expected steps to go, which is not built.) Three partitions
grouped by lineage: P on the first, E on the second, thresholds on the
third; nothing is refit (#171 F6).

RND: a frozen random encoder and a trained predictor, both the policy's
pooled item encoder with their own weights, reading one state against an
empty target (so the input is the state alone). N(x) is the prediction error
over the median error of the reference set it was normalised on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from experiments.certified_coverage import score as scoring
from wandering_light.stack_policy import MACHINE, StackPairPolicy, collate, node_encoder

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wandering_light.rollout_policy import PolicyArchitecture


def pair_features(scorer, pairs: Sequence[tuple], batch: int = 512) -> np.ndarray:
    """The frozen policy's pooled features of each ``(s, t)``."""
    out = []
    for start in range(0, len(pairs), batch):
        chunk = pairs[start : start + batch]
        out.append(scorer.pooled_features([scorer.encode(s, t) for s, t in chunk]))
    width = out[0].shape[1] if out else 0
    return np.concatenate(out) if out else np.zeros((0, width), dtype=np.float32)


@dataclass(frozen=True, slots=True)
class HeadFit:
    hidden: int = 64
    steps: int = 300
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


HEAD_FIT = HeadFit()


class Head(nn.Module):
    """A probability from pooled features: standardised input, one hidden layer."""

    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("scale", torch.ones(width))
        self.net = nn.Sequential(
            nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.scale).squeeze(-1)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if len(features) == 0:
            return np.zeros(0)
        with torch.inference_mode():
            return torch.sigmoid(self(torch.from_numpy(features).float())).numpy()


def fit_head(
    features: np.ndarray, targets: np.ndarray, config: HeadFit, seed: int
) -> Head:
    """Full-batch Adam on BCE against soft targets in [0, 1]."""
    torch.manual_seed(seed)
    if len(features) == 0:  # an empty partition: an untrained head, logged upstream
        return Head(features.shape[1], config.hidden).eval()
    x = torch.from_numpy(features).float()
    y = torch.from_numpy(np.asarray(targets, dtype=np.float32))
    head = Head(x.shape[1], config.hidden)
    head.mean.copy_(x.mean(dim=0))
    head.scale.copy_(x.std(dim=0).clamp(min=1e-6))
    optimizer = torch.optim.Adam(
        head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    for _ in range(config.steps):
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(head(x), y)
        loss.backward()
        optimizer.step()
    head.eval()
    return head


def partition(groups: Sequence, seed: int, parts: int = 3) -> np.ndarray:
    """A part per item, whole groups (lineages) at a time, as even as groups allow."""
    order = list(dict.fromkeys(groups))
    np.random.default_rng(seed).shuffle(order)
    sizes = dict.fromkeys(order, 0)
    for g in groups:
        sizes[g] += 1
    load = [0] * parts
    assign = {}
    for g in order:  # greedy: each group to the lightest part so far
        part = int(np.argmin(load))
        assign[g] = part
        load[part] += sizes[g]
    return np.array([assign[g] for g in groups], dtype=int)


def fit_partitioned(
    features: np.ndarray,
    hits: np.ndarray,
    n: int,
    parts: np.ndarray,
    seed: int,
    config: HeadFit = HEAD_FIT,
    z: float = scoring.Z90,
) -> tuple[Head, Head, dict]:
    """P on part 0, frozen; E on part 1; part 2 is left for thresholds (#171 F6).

    E's target is "the frozen P fails a fresh n-rollout Wilson check" on a
    pair P never saw; P is not refit afterwards.
    """
    hits = np.asarray(hits)
    rate = hits / n
    first, second = parts == 0, parts == 1
    p = fit_head(features[first], rate[first], config, seed)
    miss = ~scoring.hit_wilson(p.predict(features[second]), hits[second], n, z)
    e = fit_head(features[second], miss.astype(np.float32), config, seed + 1)
    facts = {
        "pairs_by_part": np.bincount(parts, minlength=3).tolist(),
        "e_target_rate": float(miss.mean()) if len(miss) else None,
    }
    return p, e, facts


# ---------------------------------------------------------------------------
# RND
# ---------------------------------------------------------------------------


def state_blocks(states: Sequence) -> list[np.ndarray]:
    """Each state against an empty target, so the block encodes the state alone."""
    return [node_encoder([() for _ in state], MACHINE)(state) for state in states]


class RND:
    def __init__(
        self,
        architecture: PolicyArchitecture,
        seed: int,
        width: int = 64,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        torch.manual_seed(seed)
        self.target = self._net(architecture, width).to(self.device).eval()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.predictor = self._net(architecture, width).to(self.device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=1e-3)
        self.scale = 1.0

    @staticmethod
    def _net(architecture: PolicyArchitecture, width: int) -> nn.Module:
        body = StackPairPolicy(architecture, encoding=MACHINE).eval()
        with torch.no_grad():  # the pooled width, from one empty state
            pooled = body.pooled_features(*collate(state_blocks([((),)]), "cpu"))
        return nn.ModuleDict({"body": body, "out": nn.Linear(pooled.shape[1], width)})

    @staticmethod
    def _forward(net: nn.Module, blocks: Sequence[np.ndarray], device) -> torch.Tensor:
        return net["out"](net["body"].pooled_features(*collate(blocks, device)))

    def raw_errors(self, states: Sequence, batch: int = 512) -> np.ndarray:
        out = []
        with torch.inference_mode():
            for start in range(0, len(states), batch):
                blocks = state_blocks(states[start : start + batch])
                goal = self._forward(self.target, blocks, self.device)
                guess = self._forward(self.predictor, blocks, self.device)
                out.append(((guess - goal) ** 2).mean(dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def train(
        self, states: Sequence, epochs: int = 1, batch: int = 256, seed: int = 0
    ) -> float:
        """Fit the predictor on ``states`` (a phase's visits plus a replay); last-epoch mean loss."""
        rng = np.random.default_rng(seed)
        self.predictor.train()
        mean = 0.0
        for _ in range(epochs):
            order = rng.permutation(len(states))
            total = 0.0
            for start in range(0, len(order), batch):
                blocks = state_blocks([states[i] for i in order[start : start + batch]])
                with torch.no_grad():
                    goal = self._forward(self.target, blocks, self.device)
                self.optimizer.zero_grad()
                loss = (
                    (self._forward(self.predictor, blocks, self.device) - goal) ** 2
                ).mean()
                loss.backward()
                self.optimizer.step()
                total += loss.item() * len(blocks)
            mean = total / max(len(states), 1)
        self.predictor.eval()
        return mean

    def normalise(self, reference: Sequence) -> None:
        """N = raw error / median raw error on ``reference``."""
        errors = self.raw_errors(reference)
        self.scale = float(np.median(errors)) if len(errors) else 1.0
        self.scale = self.scale or 1.0

    def novelty(self, states: Sequence) -> np.ndarray:
        return self.raw_errors(states) / self.scale


def calibrate_threshold(
    seen: np.ndarray, unseen: np.ndarray, quantile: float = 0.9
) -> dict:
    """n0 so that ``quantile`` of well-visited nodes pass; the unvisited pass rate is the sanity number."""
    if len(seen) == 0:
        return {"n0": None, "seen_pass": None, "unseen_pass": None}
    n0 = float(np.quantile(seen, quantile))
    return {
        "n0": n0,
        "seen_pass": float((seen <= n0).mean()),
        "unseen_pass": float((unseen <= n0).mean()) if len(unseen) else None,
        "seen_nodes": len(seen),
        "unseen_nodes": len(unseen),
    }
