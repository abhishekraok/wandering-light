"""Pick a task, run the production rollout on it, and keep what it did.

A session is the unit the UI navigates and the unit that has to be
reproducible: it records the corpus revision, the checkpoint revision, the
search settings and the seed, so the same session can be rebuilt from its own
provenance block.

The witness overlay is read from the corpus record *after* the search has
finished. It is display-only by construction: ``run_search`` never receives it,
so there is no path by which a certified answer could reach the search.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from wandering_light.rollout_debugger.trace import TraceRecorder

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wandering_light.basis_dataset import BasisTaskRecord
    from wandering_light.function_def import FunctionDefSet
    from wandering_light.solver import RolloutSearchResult
    from wandering_light.typed_list import TypedList

DEFAULT_CORPUS_DIR = Path("wandering_light/training/data/deep_corpus_v1")
DEFAULT_SPLIT = "test"
DEFAULT_BEAM_WIDTH = 64
DEFAULT_MAX_DEPTH = 8
DEFAULT_BUDGET = 100_000


@dataclass(frozen=True)
class SearchSettings:
    """The knobs a replay is pinned by, and that a deep link carries."""

    beam_width: int = DEFAULT_BEAM_WIDTH
    max_depth: int = DEFAULT_MAX_DEPTH
    budget: int = DEFAULT_BUDGET

    def to_dict(self) -> dict[str, int]:
        return {
            "beam_width": self.beam_width,
            "max_depth": self.max_depth,
            "budget": self.budget,
        }


@dataclass
class DebugSession:
    """One recorded rollout, with everything needed to reproduce it."""

    task_id: str
    input_state: dict[str, Any]
    target_state: dict[str, Any]
    settings: SearchSettings
    result: RolloutSearchResult
    trace: dict[str, Any]
    provenance: dict[str, Any]
    witness: list[str] = field(default_factory=list)
    certified_distance: int | None = None
    basis: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        solution = [function.name for function in self.result.function_defs]
        return {
            "task_id": self.task_id,
            "input": self.input_state,
            "target": self.target_state,
            "basis": self.basis,
            "settings": self.settings.to_dict(),
            "termination": self.result.termination,
            "executions": self.result.executions,
            "solution": solution,
            "solved": self.result.termination == "solved",
            "witness": self.witness,
            "certified_distance": self.certified_distance,
            "witness_overlay": self._witness_overlay(),
            "provenance": self.provenance,
            "trace": self.trace,
        }

    def _witness_overlay(self) -> list[dict[str, Any]]:
        """Per layer, whether the certified witness prefix is still in the beam.

        Diagnosis only. This is computed from the finished trace; the search
        that produced the trace never saw the witness.
        """

        overlay = []
        for layer in self.trace["layers"]:
            depth = layer["depth"]
            prefix = self.witness[:depth]
            survives = any(entry["path"] == prefix for entry in layer["retained"])
            overlay.append(
                {
                    "depth": depth,
                    "prefix": prefix,
                    "survives": bool(self.witness) and survives,
                }
            )
        return overlay


def select_task(
    records: Sequence[BasisTaskRecord],
    *,
    task_id: str | None = None,
    certified_distance: int | None = None,
    input_type: str | None = None,
    seed: int | None = None,
) -> tuple[BasisTaskRecord, dict[str, Any]]:
    """Load one task by ID, or sample one from a filtered pool with a seed.

    The returned provenance names the filters and the seed rather than only the
    chosen task, so a session says how the task was reached and not just which
    one it was.
    """

    if task_id is not None:
        for record in records:
            if record.task_id == task_id:
                return record, {"mode": "task_id", "task_id": task_id}
        raise KeyError(f"No task {task_id!r} in the loaded split")

    pool = list(records)
    if certified_distance is not None:
        pool = [
            record
            for record in pool
            if record.metadata.get("certified_distance") == certified_distance
        ]
    if input_type is not None:
        pool = [
            record
            for record in pool
            if record.input_value.item_type.__name__ == input_type
        ]
    if not pool:
        raise ValueError(
            "No task matches the requested filters: "
            f"distance={certified_distance}, input_type={input_type}"
        )

    chosen_seed = random.SystemRandom().randrange(2**32) if seed is None else seed
    # Sample from a source-index ordering so the same seed and filters pick the
    # same task regardless of how the split happened to be iterated.
    ordered = sorted(pool, key=lambda record: record.source_index)
    record = random.Random(chosen_seed).choice(ordered)
    return record, {
        "mode": "sampled",
        "seed": chosen_seed,
        "certified_distance": certified_distance,
        "input_type": input_type,
        "pool_size": len(ordered),
        "task_id": record.task_id,
    }


def run_search(
    scorer: Any,
    record: BasisTaskRecord,
    available_functions: FunctionDefSet,
    *,
    settings: SearchSettings,
    capture_logits: bool = False,
) -> tuple[RolloutSearchResult, dict[str, Any]]:
    """Run the production rollout on one task and capture what it did."""

    from wandering_light.solver import RolloutPredictor

    predictor = RolloutPredictor(
        scorer,
        beam_width=settings.beam_width,
        max_depth=settings.max_depth,
        budget=settings.budget,
    )
    recorder = TraceRecorder(capture_logits=capture_logits)
    result = predictor.search_with_cost(
        record.input_value,
        record.output_value,
        available_functions,
        observer=recorder,
    )
    return result, recorder.to_dict()


def build_session(
    scorer: Any,
    record: BasisTaskRecord,
    available_functions: FunctionDefSet,
    *,
    settings: SearchSettings,
    provenance: dict[str, Any],
    capture_logits: bool = False,
) -> DebugSession:
    """Run one task and assemble the session the UI navigates."""

    result, trace = run_search(
        scorer,
        record,
        available_functions,
        settings=settings,
        capture_logits=capture_logits,
    )
    distance = record.metadata.get("certified_distance")
    return DebugSession(
        task_id=record.task_id,
        input_state=_state_payload(record.input_value),
        target_state=_state_payload(record.output_value),
        settings=settings,
        result=result,
        trace=trace,
        provenance=provenance,
        witness=list(record.witness_function_names),
        certified_distance=distance if isinstance(distance, int) else None,
        basis=basis_table(available_functions),
    )


def basis_table(available_functions: FunctionDefSet) -> list[dict[str, str]]:
    """The whole basis with its types, sent once rather than per beam node."""

    return [
        {
            "name": function.name,
            "input_type": function.input_type,
            "output_type": function.output_type,
        }
        for function in available_functions
    ]


def _state_payload(state: TypedList) -> dict[str, Any]:
    return {
        "item_type": state.item_type.__name__,
        "length": len(state.items),
        "value": repr(state.items),
        "serialized": state.to_string(),
    }


def encoding_view(state: TypedList, target: TypedList) -> dict[str, Any]:
    """Show the exact model input for one (state, target) pair.

    This calls the policy's own encoder, so what is displayed is what the model
    was fed -- byte truncation, separator, padding and tensor shapes included.
    """

    from wandering_light.rollout_policy import (
        ITEM_CAP,
        MAX_ITEMS,
        PAD,
        SEP,
        _serialized_pair_items,
        pair_batch,
    )

    serialized_state = state.to_string()
    serialized_target = target.to_string()
    pairs = _serialized_pair_items(serialized_state, serialized_target)
    sequences, counts, item_mask = pair_batch([(serialized_state, serialized_target)])

    source_items = json.loads(serialized_state).get("items", [])
    target_items = json.loads(serialized_target).get("items", [])
    aligned = []
    for index, (left, right) in enumerate(
        zip(source_items[:MAX_ITEMS], target_items[:MAX_ITEMS], strict=True)
    ):
        left_bytes = json.dumps(left, separators=(",", ":")).encode("utf-8", "replace")
        right_bytes = json.dumps(right, separators=(",", ":")).encode(
            "utf-8", "replace"
        )
        aligned.append(
            {
                "index": index,
                "state_text": left_bytes.decode("utf-8", "replace"),
                "target_text": right_bytes.decode("utf-8", "replace"),
                "state_bytes": len(left_bytes),
                "target_bytes": len(right_bytes),
                "state_truncated": len(left_bytes) > ITEM_CAP,
                "target_truncated": len(right_bytes) > ITEM_CAP,
            }
        )

    return {
        "item_cap": ITEM_CAP,
        "max_items": MAX_ITEMS,
        "pad_token": PAD,
        "separator_token": SEP,
        "items_dropped": max(0, len(source_items) - MAX_ITEMS),
        "aligned": aligned,
        "encoded_pairs": [list(pair) for pair in pairs],
        "tensor_shapes": {
            "sequences": list(sequences.shape),
            "counts": list(counts.shape),
            "item_mask": list(item_mask.shape),
        },
        "padding": int((sequences == PAD).sum().item()),
    }
