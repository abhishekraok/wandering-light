"""Independent Stage A reconstruction for #178; synthetic callers only so far.

Uses the established interpreter/encoder primitives, never the diagnostic's
selection, replay, ownership, support, seal or metrics implementations. This is
one part of result verification, not a complete result verifier or trial CLI.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import fields

from wandering_light.stack_machine import (
    Config,
    Mark,
    format_stack,
    parse_stack,
    step_raw,
)
from wandering_light.stack_pbe import OP_BY_NAME, OPS, Task
from wandering_light.stack_policy import encode_stack, flags

STRUCTURAL = {"[", "]", "map", "filter", "fold", "ifte", "dip", "i"}
ASSEMBLY = {"quote", "cons", "cat"}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def same(actual, expected, context):
    require(
        json.dumps(actual, sort_keys=True, allow_nan=False)
        == json.dumps(expected, sort_keys=True, allow_nan=False),
        context,
    )


def check_time(deadline):
    require(deadline is None or time.perf_counter() < deadline, "verification deadline")


def choose(tasks, programs):
    """Separate original-skeleton split and salted ordering implementation."""
    first, identities, groups = {}, {}, defaultdict(set)
    for index, task in enumerate(tasks):
        require(task.split == "train", "non-train original task")
        key = task.skeleton or task.behaviour
        require(identities.setdefault(task.behaviour, key) == key, "inconsistent group")
        first.setdefault(task.behaviour, (index, task))
        if (
            task.behaviour in programs
            and int(hashlib.sha256(key.encode()).hexdigest(), 16) % 10
        ):
            groups[key].add(task.behaviour)

    def order(salt, name):
        return hashlib.sha256((salt + name).encode("utf-8")).hexdigest(), name

    panel = []
    for key in sorted(groups, key=lambda name: order("capped-glue-D1-group:", name))[
        :64
    ]:
        name = min(
            groups[key], key=lambda name: order("capped-glue-D1-behaviour:", name)
        )
        index, task = first[name]
        metadata = {
            field.name: getattr(task, field.name)
            for field in fields(Task)
            if field.name not in {"program", "held_out_inputs", "held_out_outputs"}
        }
        panel.append(
            {
                "original_index": index,
                "original_metadata": metadata,
                "original_program": task.program,
                "capped_program": programs[name].split(),
                "group": key,
                "hidden_pairs_retained": False,
            }
        )
    selection = {
        "train_tasks": len(tasks),
        "train_behaviours": len(first),
        "side_behaviours": len(programs),
        "eligible_groups": len(groups),
        "selected_groups": len(panel),
        "group_cap": 64,
        "side_behaviours_absent_from_train": sorted(set(programs) - set(first)),
    }
    return panel, selection


def move(product, name, deadline):
    """Raw interpreter per example, independently preserving first refusal."""
    stacks, fuel = [], 0
    for example, stack in enumerate(product):
        check_time(deadline)
        result = step_raw(OP_BY_NAME[name], stack, 100000, 0)
        check_time(deadline)
        if isinstance(result, str):
            return None, fuel, example, f"example {example}: {result}"
        stacks.append(result[0])
        fuel += result[1]
    return tuple(stacks), fuel, len(product), None


def reconstruct(item, deadline):
    program, metadata = item["capped_program"], item["original_metadata"]
    require(program and metadata["inputs"], "empty witness or examples")
    require(len(metadata["inputs"]) == len(metadata["outputs"]), "shown pair count")
    require(metadata["basis"] == "mark-28", "wrong witness basis")
    product = tuple(parse_stack(text, basis="mark-28") for text in metadata["inputs"])
    target = tuple(parse_stack(text, basis="mark-28") for text in metadata["outputs"])
    opened, rows, products = [], [], []
    for step, name in enumerate(program):
        require(name in OP_BY_NAME, "unknown witness op")
        require(name != "]" or opened, "unmatched closer")
        depth = len(opened)
        require(
            all(sum(isinstance(v, Mark) for v in stack) == depth for stack in product),
            "Mark depth",
        )
        kind = (
            "open"
            if name == "["
            else "close"
            if name == "]"
            else "body"
            if depth
            else "run"
            if name in STRUCTURAL
            else "direct"
        )
        row = {
            "behaviour": metadata["behaviour"],
            "group": item["group"],
            "step": step,
            "op": name,
            "depth": depth,
            "opener": opened[-1] if opened else None,
            "kind": kind,
            "side": "body" if depth else "top",
            "structural": name in STRUCTURAL,
            "assembly": name in ASSEMBLY,
            "original_index": item["original_index"],
            "product": [format_stack(Config(stack=stack)) for stack in product],
            "target": metadata["outputs"],
        }
        products.append(product)
        product, spent, executed, reason = move(product, name, deadline)
        require(product is not None, f"witness refusal: {reason}")
        row.update(witness_fuel=spent, witness_executions=executed)
        rows.append(row)
        if name == "[":
            opened.append(step)
        elif name == "]":
            opened.pop()
    require(not opened, "unmatched opener")
    require(product == target, "shown witness target")
    require(
        all(not any(isinstance(v, Mark) for v in stack) for stack in product),
        "final Mark",
    )
    boundaries = [i + 1 for i, row in enumerate(rows) if row["kind"] == "run"]
    if not boundaries or boundaries[-1] != len(rows):
        boundaries.append(len(rows))
    start = 0
    for span, stop in enumerate(boundaries):
        for step in range(start, stop):
            row = rows[step]
            n, index = stop - start, step - start
            top_content = row["side"] == "top" and not row["structural"]
            row.update(
                span=span,
                index_in_span=index,
                span_length=n,
                position="only"
                if n == 1
                else "first"
                if index == 0
                else "last"
                if index == n - 1
                else "middle",
                before_opener=top_content
                and any(
                    r["op"] == "[" and r["depth"] == 0 for r in rows[step + 1 : stop]
                ),
                between_closer_and_combinator=top_content
                and rows[stop - 1]["kind"] == "run"
                and any(r["op"] == "]" and r["depth"] == 1 for r in rows[start:step]),
            )
        start = stop
    return rows, products, target


def matched(rows):
    buckets = defaultdict(lambda: {"top": [], "body": []})
    groups = {}
    for row in rows:
        groups[row["behaviour"]] = row["group"]
        if not row["structural"]:
            buckets[row["behaviour"], row["op"], row["position"]][row["side"]].append(
                [row["behaviour"], row["step"]]
            )
    cells = []
    for (name, op, position), sides in sorted(buckets.items()):
        if sides["top"] and sides["body"]:
            require(position == "middle", "non-midspan common support")
            cells.append(
                {
                    "behaviour": name,
                    "group": groups[name],
                    "op": op,
                    "position": position,
                    **sides,
                }
            )
    group_names = sorted({cell["group"] for cell in cells})
    support = {
        "groups": group_names,
        "behaviours": sorted({cell["behaviour"] for cell in cells}),
        "cells": len(cells),
        "passes": len(group_names) >= 20 and len(cells) >= 40,
    }
    keys = {
        tuple(key) for cell in cells for side in ("top", "body") for key in cell[side]
    }
    for row in rows:
        row["matched"] = (row["behaviour"], row["step"]) in keys
    labels = Counter(
        (
            r["side"],
            "structural" if r["structural"] else "content",
            r["position"],
            r["op"],
            r["depth"],
        )
        for r in rows
    )
    counts = {
        "all_rows": len(rows),
        "content_rows": sum(not r["structural"] for r in rows),
        "structural_rows": sum(r["structural"] for r in rows),
        "matched_rows": len(keys),
        "matched_top_rows": sum(r["matched"] and r["side"] == "top" for r in rows),
        "matched_body_rows": sum(r["matched"] and r["side"] == "body" for r in rows),
        "omitted_content_rows": sum(
            not r["structural"] and not r["matched"] for r in rows
        ),
        "by_label": [
            dict(
                zip(("side", "class", "position", "op", "depth"), key, strict=True),
                count=n,
            )
            for key, n in sorted(labels.items())
        ],
    }
    return cells, support, counts


def legality(product, witness, deadline):
    mask, refused = [], []
    for op in OPS:
        result, _, _, reason = move(product, op.name, deadline)
        mask.append(result is not None)
        if result is None:
            require(op.name != witness, "alternative replay refused witness")
            category = (
                "fuel"
                if reason.endswith("out of fuel")
                else "depth"
                if reason.endswith("quotation nested too deep")
                else "ordinary"
            )
            refused.append({"op": op.name, "reason": reason, "class": category})
    counts = {
        name: sum(r["class"] == name for r in refused)
        for name in ("ordinary", "fuel", "depth")
    }
    return {
        "mask": mask,
        "refusals": refused,
        "first_refusal_counts": counts,
        "resource_affected": bool(counts["fuel"] or counts["depth"]),
    }


def exposure(product, target):
    examples = []
    for state, wanted in zip(product, target, strict=True):
        sw, tw = (
            len(encode_stack(state, sys.maxsize)),
            len(encode_stack(wanted, sys.maxsize)),
        )
        examples.append(
            {
                "state_pre_cap_width": sw,
                "target_pre_cap_width": tw,
                "state_retained_width": min(sw, 160),
                "target_retained_width": min(tw, 64),
                "state_truncated": sw > 160,
                "target_truncated": tw > 64,
                "difference_flags": len(flags(state, wanted)),
            }
        )
    layouts = {}
    for name in ("machine", "difference"):
        widths = [
            r["state_retained_width"]
            + 1
            + r["target_retained_width"]
            + (1 + r["difference_flags"] if name == "difference" else 0)
            for r in examples
        ]
        layouts[name] = {"item_widths": widths, "block_width": max(widths)}
    return {
        "state_cap": 160,
        "target_cap": 64,
        "examples": examples,
        "layouts": layouts,
        "any_state_truncated": any(r["state_truncated"] for r in examples),
        "any_target_truncated": any(r["target_truncated"] for r in examples),
        "training_row_multiplicity": None,
        "training_row_multiplicity_status": "unavailable_not_reconstructed",
    }


def verify_stage(stage, tasks, programs, *, deadline=None):
    require(stage["schema"] == "capped-glue-D1-A-stage-a", "stage schema")
    require(
        stage["complete"] is True
        and stage["models_loaded"] == stage["logit_rows"] == 0,
        "stage contains model work",
    )
    panel, selection = choose(tasks, programs)
    same(stage["panel"], panel, "original panel selection")
    same(stage["selection"], selection, "selection totals")
    rows, replay = [], []
    for item in panel:
        check_time(deadline)
        reconstructed, products, target = reconstruct(item, deadline)
        rows.extend(reconstructed)
        replay.extend((product, target) for product in products)
    cells, support, counts = matched(rows)
    same(stage["cells"], cells, "cell membership")
    same(stage["support"], support, "support gate")
    same(stage["counts"], counts, "row counts")
    require(
        stage["kind"] == ("supported" if support["passes"] else "support_only"),
        "support kind",
    )
    copies = defaultdict(list)
    for item in panel:
        copies[" ".join(item["capped_program"])].append(
            item["original_metadata"]["behaviour"]
        )
    same(
        stage["duplicate_capped_programs"],
        [
            {"program": text, "behaviours": sorted(names)}
            for text, names in sorted(copies.items())
            if len(names) > 1
        ],
        "duplicate program texts",
    )
    same(
        stage["limits"],
        {
            "group_cap": 64,
            "call_fuel": 100000,
            "initial_depth": 0,
            "max_quotation_depth": 32,
            "deadline_seconds": 900,
            "batch_size": 32,
            "torch_threads": 1,
        },
        "frozen limits",
    )
    require(len(stage["rows"]) == len(rows), "row coverage")
    refusals = Counter()
    for saved, expected, (product, target) in zip(
        stage["rows"], rows, replay, strict=True
    ):
        check_time(deadline)
        if support["passes"]:
            expected["legality"] = legality(product, expected["op"], deadline)
            expected["encoder_diagnostics"] = exposure(product, target)
            refusals.update(expected["legality"]["first_refusal_counts"])
        same(
            saved, expected, f"replayed row {expected['behaviour']}:{expected['step']}"
        )
    same(stage["alternative_first_refusal_counts"], dict(refusals), "refusal totals")
    return {
        "groups": len(support["groups"]),
        "cells": len(cells),
        "rows": len(rows),
        "support_passes": support["passes"],
        "independent_shown_replay": True,
    }
