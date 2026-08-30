"""Project search progress into the policy's own representation.

A 2D plot of a search needs axes. Depth and cumulative log-probability are
honest but say nothing about whether a state is *closer* to the target. The
policy's pooled ``(state, target)`` encoding does: it is what the action head
reads, so distance in it is distance in the space the search actually moves
through.

The anchor is ``phi(target, target)`` -- where a solved state lands. Points are
projected onto the two principal components of the visited states plus that
anchor, so "toward the anchor" reads as progress. PCA is fitted per session:
the axes describe this task's search, and are not comparable across tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wandering_light.rollout_policy import RolloutPolicy
    from wandering_light.typed_list import TypedList


def pooled_features(
    policy: RolloutPolicy, pairs: Sequence[tuple[TypedList, TypedList]]
) -> np.ndarray:
    """Encode each ``(state, target)`` pair exactly as the policy does."""

    from wandering_light.rollout_policy import pair_batch

    if not pairs:
        return np.zeros((0, 0), dtype=np.float32)
    serialized = [(state.to_string(), target.to_string()) for state, target in pairs]
    inputs = pair_batch(serialized, device=policy.device)
    with torch.inference_mode():
        features = policy.model.pooled_features(*inputs)
    return features.detach().cpu().numpy().astype(np.float32)


def project_to_plane(features: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Project rows onto their first two principal components.

    Returns the 2D coordinates and the fraction of variance each axis carries,
    so the UI can say how much of the geometry the picture is showing.
    """

    if features.shape[0] < 2:
        return np.zeros((features.shape[0], 2), dtype=np.float32), [0.0, 0.0]
    centered = features - features.mean(axis=0, keepdims=True)
    # SVD rather than an eigendecomposition of the covariance: the feature
    # count exceeds the point count for short searches, and SVD handles that.
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    axes = components[:2]
    coordinates = centered @ axes.T
    total = float((singular**2).sum())
    explained = [
        float(value**2 / total) if total > 0 else 0.0 for value in singular[:2]
    ]
    while len(explained) < 2:
        explained.append(0.0)
    return coordinates.astype(np.float32), explained


def progress_frame(
    policy: RolloutPolicy,
    start: TypedList,
    target: TypedList,
    states: Sequence[TypedList],
) -> dict[str, Any]:
    """Place states on a start-to-target axis, with their departure from it.

    PCA needs a population to fit and its axes turn as that population grows,
    which makes it wrong for a walk being driven one step at a time -- points
    would move because the fit moved. Two anchors give a frame that does not:
    ``x`` is progress along phi(start, target) -> phi(target, target), where 0
    is the input and 1 is solved, and ``y`` is how far off that line the state
    sits. Both are stateless, so a step needs no cached basis.
    """

    pairs = [(start, target), (target, target), *((state, target) for state in states)]
    features = pooled_features(policy, pairs)
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
    return {
        "points": points,
        "axis_length": round(length, 4),
        "start_distance": round(length, 4),
    }


def session_map(
    policy: RolloutPolicy,
    target: TypedList,
    layers: Sequence[dict[str, Any]],
    states_by_depth: Sequence[Sequence[TypedList]],
    *,
    start: TypedList,
) -> dict[str, Any]:
    """Lay out the start, every retained beam node, and the target anchor.

    ``states_by_depth[i]`` are the materialized states retained at
    ``layers[i]``, in beam order. They are recomputed from the trace's paths by
    the caller, which keeps this function free of any search of its own.
    """

    pairs: list[tuple[TypedList, TypedList]] = [(start, target), (target, target)]
    index: list[dict[str, Any]] = [
        {"kind": "start", "depth": 0, "rank": 0},
        {"kind": "target", "depth": None, "rank": None},
    ]
    for layer, states in zip(layers, states_by_depth, strict=True):
        for rank, (entry, state) in enumerate(
            zip(layer["retained"], states, strict=True)
        ):
            pairs.append((state, target))
            index.append(
                {
                    "kind": "node",
                    "depth": layer["depth"],
                    "rank": rank,
                    "path": entry["path"],
                    "log_probability": entry["log_probability"],
                }
            )

    features = pooled_features(policy, pairs)
    coordinates, explained = project_to_plane(features)
    anchor_plane = coordinates[1]
    anchor_feature = features[1]
    points = [
        {
            **entry,
            "x": round(float(point[0]), 4),
            "y": round(float(point[1]), 4),
            # Distance in the full representation, not the projection: the plot
            # can only show two axes, but "how far from solved" should not be
            # an artifact of which two.
            "distance_to_target": round(
                float(np.linalg.norm(feature - anchor_feature)), 4
            ),
            "plane_distance": round(float(np.linalg.norm(point - anchor_plane)), 4),
        }
        for entry, point, feature in zip(index, coordinates, features, strict=True)
    ]
    return {
        "points": points,
        "explained_variance": [round(value, 4) for value in explained],
        "dimensions": int(features.shape[1]) if features.size else 0,
    }
