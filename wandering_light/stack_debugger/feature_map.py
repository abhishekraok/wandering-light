"""Lay the beam out in the policy's own space, as the rollout debugger does.

The pooled ``(state, target)`` set encoding is what the action head reads, so
distance in it is distance in the space the search moves through. The math is
``rollout_debugger.feature_map``'s; only the encoder call differs, because a
stack node is a product of k example items rather than one byte pair.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from wandering_light.rollout_debugger.feature_map import project_to_plane
from wandering_light.stack_beam import block_encoder

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wandering_light.stack_beam import Product
    from wandering_light.stack_policy import PolicyScorer


def features_of(
    scorer: PolicyScorer, products: Sequence[Product], outputs: Product
) -> np.ndarray:
    if not products:
        return np.zeros((0, 0), dtype=np.float32)
    encode = block_encoder(scorer, outputs)
    blocks = [encode(product) for product in products]
    return scorer.pooled_features(blocks)


def session_map(
    scorer: PolicyScorer,
    inputs: Product,
    outputs: Product,
    layers: Sequence[dict[str, Any]],
    retained_products: Sequence[Sequence[Product]],
    node_features: Sequence[np.ndarray | None] = (),
) -> dict[str, Any]:
    """Lay out the start, every retained node and the target anchor.

    The nodes retained at layer d are exactly the nodes scored at layer d+1,
    so their pooled rows already exist when the recorder kept them
    (``node_features[d+1]``); only a layer the search never scored -- the one
    it fit in -- is encoded here, in one small extra pass.
    """
    anchors = features_of(scorer, [inputs, outputs], outputs)
    rows: list[np.ndarray] = [anchors[0], anchors[1]]
    index: list[dict[str, Any]] = [
        {"kind": "start", "depth": 0, "rank": 0},
        {"kind": "target", "depth": None, "rank": None},
    ]
    for position, (layer, retained) in enumerate(
        zip(layers, retained_products, strict=True)
    ):
        kept = (
            node_features[position + 1] if position + 1 < len(node_features) else None
        )
        if kept is None or len(kept) != len(retained):
            kept = (
                features_of(scorer, list(retained), outputs)
                if retained
                else np.zeros((0, anchors.shape[1]))
            )
        for rank, _product in enumerate(retained):
            rows.append(kept[rank])
            index.append(
                {
                    "kind": "node",
                    "depth": layer["depth"],
                    "rank": rank,
                    "program": layer["retained"][rank]["program"],
                    "log_probability": layer["retained"][rank]["log_probability"],
                }
            )
    features = np.stack(rows).astype(np.float32)
    coordinates, explained = project_to_plane(features)
    anchor = features[1]
    points = [
        {
            **entry,
            "x": round(float(point[0]), 4),
            "y": round(float(point[1]), 4),
            "distance_to_target": round(float(np.linalg.norm(feature - anchor)), 4),
        }
        for entry, point, feature in zip(index, coordinates, features, strict=True)
    ]
    return {
        "points": points,
        "explained_variance": [round(v, 4) for v in explained],
        "dimensions": int(features.shape[1]) if features.size else 0,
    }


def progress_frame(
    scorer: PolicyScorer, inputs: Product, outputs: Product, trail: Sequence[Product]
) -> dict[str, Any]:
    """x is progress along phi(start) -> phi(target); y is the departure from it."""
    features = features_of(scorer, [inputs, outputs, *trail], outputs)
    origin, destination = features[0], features[1]
    axis = destination - origin
    length = float(np.linalg.norm(axis))
    unit = np.zeros_like(axis) if length == 0 else axis / length
    points = []
    for feature in features[2:]:
        offset = feature - origin
        along = float(offset @ unit)
        residual = offset - along * unit
        points.append(
            {
                "x": round(along / length if length else 0.0, 4),
                "y": round(float(np.linalg.norm(residual)) / (length or 1.0), 4),
                "distance_to_target": round(
                    float(np.linalg.norm(feature - destination)), 4
                ),
            }
        )
    return {"points": points, "axis_length": round(length, 4)}
