"""Independent low-level input/attempt and retention checks for synthetic Gate A.

Does not import the candidate retention/search implementation. Reconstructs
machine attempts, masks, score ordering and typed identity from observation.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from wandering_light.quotation_guard import blocked
from wandering_light.stack_machine import DEFAULT_FUEL_LIMIT, Lst, step_raw, value_key
from wandering_light.stack_pbe import OPS


def result_key(result):
    return {k: v for k, v in asdict(result).items() if k != "seconds"}


def identity(product):
    return tuple(tuple(value_key(v) for v in stack) for stack in product)


def atom_count(product):
    todo = [v for stack in product for v in stack]
    count = 0
    while todo:
        v = todo.pop()
        count += 1
        if isinstance(v, Lst):
            todo.extend(v.items)
    return count


def verify_attempts(capture, result=None):
    """Recompute every attempt, refusal scope and successful-example charging."""
    beams = {}
    successful = {}
    charged_fuel = charged_executions = 0
    for event in capture.events:
        if event[0] == "started":
            _, depth, beam = event
            beams[depth] = beam
        elif event[0] == "attempt":
            _, depth, node, index, moved, reason, fuel, stepped = event
            product = beams[depth][node][0]
            states = []
            own_fuel = 0
            own_reason = None
            for example, stack in enumerate(product):
                outcome = step_raw(OPS[index], stack, DEFAULT_FUEL_LIMIT, 0)
                if isinstance(outcome, str):
                    own_reason = f"example {example}: {outcome}"
                    break
                states.append(outcome[0])
                own_fuel += outcome[1]
            expected = tuple(states) if own_reason is None else None
            assert moved == expected
            assert reason == own_reason
            assert fuel == own_fuel and stepped == len(states)
            charged_fuel += fuel
            charged_executions += stepped
            successful[depth, node, index] = moved is not None
        elif event[0] == "scored":
            _, depth, shape, mask_bytes, _values = event
            mask = np.frombuffer(mask_bytes, dtype=bool).reshape(shape)
            rebuilt = np.array(
                [
                    [successful[depth, node, index] for index in range(len(OPS))]
                    for node in range(shape[0])
                ]
            )
            np.testing.assert_array_equal(mask, rebuilt)
    if result is not None:
        assert result.executions == charged_executions
        assert result.fuel == charged_fuel
        assert result.expansions == sum(
            1 for event in capture.events if event[0] == "attempt" and event[3] == 0
        ) + sum(
            1
            for event in capture.events
            if event[0] == "started"
            and not any(e[0] == "attempt" and e[1] == event[1] for e in capture.events)
        )


def verify_layers(capture, case):
    """Independent plain-table oracle; guards and atom count also recomputed.

    Rebuild the ranking from attempted successors and legal-only scores rather
    than trusting the candidate's persisted ranked list. Unread eligibility is
    never emitted as a verdict, even though the oracle can compute ordinary C.
    """
    starts = {e[1]: e[2] for e in capture.events if e[0] == "started"}
    scored = {}
    for e in capture.events:
        if e[0] == "scored":
            _, d, shape, mask_bytes, values = e
            mask = np.frombuffer(mask_bytes, dtype=bool).reshape(shape)
            scored[d] = dict(
                zip(zip(*np.where(mask), strict=True), values, strict=True)
            )
    for layer in capture.layers:
        beam = starts[layer.depth]
        table = []
        for e in capture.events:
            if e[0] != "attempt" or e[1] != layer.depth or e[4] is None:
                continue
            _, _d, n, i, p, _reason, _fuel, _stepped = e
            table.append(
                (
                    n,
                    i,
                    p,
                    (*beam[n][1], OPS[i].name),
                    beam[n][2] + scored[layer.depth][n, i],
                )
            )
        table.sort(key=lambda row: row[4], reverse=True)
        assert tuple(table) == layer.ranked
        old_seen = {identity(p) for p in layer.old_seen}
        ordinary = []
        filters = {}
        ordinary_seen = set(old_seen)
        for n, i, p, program, score in table:
            reason = (
                blocked(p[0], case.horizon - layer.depth - 1, case.guard)
                if case.guard != "off"
                else None
            )
            if reason is not None:
                filters[n, i] = f"guard:{reason}"
            elif case.cap is not None and atom_count(p) > case.cap:
                filters[n, i] = "guard:size"
            elif identity(p) in ordinary_seen:
                filters[n, i] = "duplicate"
            else:
                ordinary_seen.add(identity(p))
                ordinary.append((n, i, p, program, score))
                if len(ordinary) == case.width:
                    break
        anchor = ordinary[0][4] if ordinary else None
        expected = [
            row
            for row in ordinary
            if case.band is None
            or np.float64(row[4]) >= np.float64(anchor) - np.float64(case.band)
        ]
        retained = tuple((p, pr, score) for _n, _i, p, pr, score in expected)
        parents = tuple((n, i) for n, i, _p, _pr, _s in expected)
        assert layer.retained == retained and layer.parents == parents
        assert layer.anchor == anchor
        read = {}
        rejected = None
        stop = "exhausted"
        count = 0
        # The table scan is independent of candidate code. A failed band test
        # ends before a state could affect same-layer identity or later seen.
        own_seen = set(old_seen)
        for rank, (n, i, p, _pr, score) in enumerate(table):
            reason = (
                blocked(p[0], case.horizon - layer.depth - 1, case.guard)
                if case.guard != "off"
                else None
            )
            if reason is not None:
                read[n, i] = f"guard:{reason}"
            elif case.cap is not None and atom_count(p) > case.cap:
                read[n, i] = "guard:size"
            elif identity(p) in own_seen:
                read[n, i] = "duplicate"
            elif case.band is not None and np.float64(score) < np.float64(
                anchor
            ) - np.float64(case.band):
                read[n, i] = "score_band"
                rejected = (n, i)
                stop = "score_band"
                break
            else:
                read[n, i] = "retained"
                own_seen.add(identity(p))
                count += 1
                if count == case.width:
                    stop = "width" if rank + 1 < len(table) else "exhausted"
                    break
        assert layer.verdicts == read
        assert layer.stop == stop and layer.first_band_rejected == rejected
        assert layer.width_pruned is (stop == "width")
        assert layer.band_pruned is (stop == "score_band")
        assert bool(layer.all_inf_fallback) == (
            case.band is not None and anchor is not None and np.isneginf(anchor)
        )
        # Verify the persistent set for later layers independently.
        later = next((x for x in capture.layers if x.depth == layer.depth + 1), None)
        if later is not None:
            assert {identity(p) for p in later.old_seen} == own_seen
