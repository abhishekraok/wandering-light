"""The recorder watches the beam without changing it (stack-rollout-debugger, #115)."""

import json
import random

from wandering_light.rollout_policy import PolicyArchitecture
from wandering_light.stack_beam import BeamObserver, beam_solve
from wandering_light.stack_debugger.trace import OP_NAMES, TraceRecorder
from wandering_light.stack_pbe import BY_NAME, build_task, draw_task
from wandering_light.stack_policy import PolicyScorer, StackPairPolicy

TINY = PolicyArchitecture(8, 16, 32)


def task_for(name: str, seed: int = 4):
    rng = random.Random(seed)
    return build_task(BY_NAME[name], draw_task(BY_NAME[name], rng, 4, held_out=6))


def scorer_for(seed: int = 0) -> PolicyScorer:
    import torch

    torch.manual_seed(seed)
    return PolicyScorer(StackPairPolicy(TINY), "cpu")


def run_twice(task, **settings):
    scorer = scorer_for()
    plain = beam_solve(task, scorer, **settings)
    recorder = TraceRecorder()
    watched = beam_solve(task, scorer, observer=recorder, **settings)
    return plain, watched, recorder


def test_recording_does_not_change_a_fit():
    plain, watched, recorder = run_twice(
        task_for("add_pair", 6), beam_width=8, max_depth=4
    )
    assert plain.program == watched.program == ("add",)
    assert (
        plain.termination,
        plain.executions,
        plain.fuel,
        plain.expansions,
        plain.depth,
    ) == (
        watched.termination,
        watched.executions,
        watched.fuel,
        watched.expansions,
        watched.depth,
    )
    assert recorder.termination == "fit"
    assert recorder.executions == plain.executions
    payload = recorder.to_dict()
    assert payload["layers"][0]["nodes"][0]["program"] == []
    fits = [
        (op, e)
        for row in payload["layers"][0]["expansions"]
        for op, e in zip(OP_NAMES, row, strict=True)
        if e and e["verdict"] == "fit"
    ]
    assert [op for op, _ in fits] == ["add"]


def test_recording_does_not_change_a_max_depth_run():
    plain, watched, recorder = run_twice(task_for("sum"), beam_width=6, max_depth=3)
    assert plain.program is None and watched.program is None
    assert plain.termination == watched.termination == "max_depth"
    assert plain.executions == watched.executions
    assert plain.fuel == watched.fuel
    payload = recorder.to_dict()
    assert [layer["depth"] for layer in payload["layers"]] == [0, 1, 2]
    for layer in payload["layers"]:
        assert layer["scored"] and layer["ranked"]
        assert len(layer["retained"]) <= 6
        for node_row, score_row in zip(
            layer["expansions"], layer["scores"], strict=True
        ):
            for entry, score in zip(node_row, score_row, strict=True):
                assert entry is not None
                if entry["outcome"] == "refused":
                    assert score is None and entry["successor"] is None
                else:
                    assert score is not None
                    assert entry["verdict"] in {
                        "retained",
                        "duplicate",
                        "cut",
                        "guard:close",
                        "guard:type",
                    }
        # Every retained node names the (node, op) that produced it.
        for retained in layer["retained"]:
            node, op = retained["parent"]
            assert layer["expansions"][node][op]["verdict"] == "retained"
            assert retained["program"] == [
                *layer["nodes"][node]["program"],
                OP_NAMES[op],
            ]
    # The trace is plain JSON: no tuples, no numpy scalars, no infinities.
    json.dumps(payload, allow_nan=False)


def test_captured_ops_are_told_apart_from_executed_ones():
    # Depth 0 is exhaustive at beam 64, so every legal first op is retained.
    _plain, _watched, recorder = run_twice(task_for("sum"), beam_width=64, max_depth=2)
    payload = recorder.to_dict()
    second = payload["layers"][1]
    open_nodes = [
        i for i, node in enumerate(second["nodes"]) if node["program"] == ["["]
    ]
    assert open_nodes, "an open [ is legal at depth 0 and retained at beam 64"
    row = second["expansions"][open_nodes[0]]
    by_op = dict(zip(OP_NAMES, row, strict=True))
    assert by_op["add"]["outcome"] == "captured"
    assert by_op["]"]["outcome"] == "executed"
    assert by_op["["]["outcome"] == "executed"


def test_the_base_observer_is_inert():
    task = task_for("add_pair", 6)
    scorer = scorer_for()
    plain = beam_solve(task, scorer, beam_width=4, max_depth=2)
    watched = beam_solve(
        task, scorer, beam_width=4, max_depth=2, observer=BeamObserver()
    )
    fields = ("program", "termination", "executions", "fuel", "expansions", "depth")
    assert [getattr(plain, f) for f in fields] == [getattr(watched, f) for f in fields]
