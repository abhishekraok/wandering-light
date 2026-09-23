"""The collision recorder reads the beam's duplicate verdicts without changing the search (#164)."""

import random

from experiments.seen_set_collisions import (
    CollisionRecorder,
    closed,
    markdown,
    summarise,
)
from wandering_light.rollout_policy import PolicyArchitecture
from wandering_light.stack_beam import beam_solve
from wandering_light.stack_machine import parse_stack
from wandering_light.stack_pbe import BASIS, BY_NAME, build_task, draw_task
from wandering_light.stack_policy import PolicyScorer, StackPairPolicy

TINY = PolicyArchitecture(8, 16, 32)


def task_for(name: str, seed: int = 4):
    rng = random.Random(seed)
    return build_task(BY_NAME[name], draw_task(BY_NAME[name], rng, 4, held_out=6))


def scorer_for(seed: int = 0) -> PolicyScorer:
    import torch

    torch.manual_seed(seed)
    return PolicyScorer(StackPairPolicy(TINY), "cpu")


def test_recording_does_not_change_the_search():
    task = task_for("sum_of_squares", 3)
    scorer = scorer_for()
    settings = {"beam_width": 8, "max_depth": 6}
    plain = beam_solve(task, scorer, **settings)
    recorder = CollisionRecorder()
    watched = beam_solve(task, scorer, observer=recorder, **settings)
    assert (plain.program, plain.termination, plain.expansions, plain.depth) == (
        watched.program,
        watched.termination,
        watched.expansions,
        watched.depth,
    )
    assert recorder.layers == len(recorder.retained_per_layer)
    assert all(0 < n <= 8 for n in recorder.retained_per_layer)


def test_every_duplicate_is_classified_once():
    recorder = CollisionRecorder()
    beam_solve(
        task_for("sum", 5), scorer_for(1), observer=recorder, beam_width=16, max_depth=8
    )
    c = recorder.counts
    by_kind = sum(c[f"duplicate:{k}"] for k in ("cycle", "same-layer", "earlier-layer"))
    by_state = sum(
        c[f"duplicate:{k}:{s}"]
        for k in ("cycle", "same-layer", "earlier-layer")
        for s in ("open", "closed")
    )
    assert c["duplicate"] == by_kind == by_state
    assert c["retained"] == c["retained:open"] + c["retained:closed"]
    assert c["retained"] == sum(recorder.retained_per_layer)
    closed_cross = (
        c["duplicate:same-layer:closed"] + c["duplicate:earlier-layer:closed"]
    )
    assert sum(v for k, v in c.items() if k.startswith("met:length:")) == closed_cross
    assert (
        sum(v for k, v in c.items() if k.startswith("met:longer-by:")) == closed_cross
    )
    assert c["met:longer-by:0"] == c["duplicate:same-layer:closed"]
    assert c["met:states"] <= closed_cross


def test_a_cycle_is_a_duplicate_of_the_node_s_own_path():
    # ``swap swap`` returns to the root: the first ``swap`` is retained, the
    # second lands on the root product, which is the node's own ancestor. The
    # task must not fit in one op, or the search ends before its second layer.
    task = task_for("sum_of_squares", 2)
    recorder = CollisionRecorder()
    beam_solve(task, scorer_for(2), observer=recorder, beam_width=28, max_depth=2)
    # every root-returning op pair (swap swap, dup drop, over drop, 0 drop ...)
    # is a cycle, never a cross-path meeting of two different routes to the root
    assert recorder.counts["duplicate:cycle"] >= 1
    for sample in recorder.samples:
        assert sample["first"] != sample["second"]


def test_closed_reads_the_first_example():
    open_product = (parse_stack("1 [ dup", basis=BASIS),)
    assert not closed(open_product)
    assert closed((parse_stack("1 2", basis=BASIS),))


def test_summary_and_markdown_hold_together():
    rows = [
        {
            "layers": 2,
            "retained_per_layer": [3, 4],
            "counts": {
                "retained": 7,
                "retained:closed": 5,
                "retained:open": 2,
                "duplicate": 3,
                "duplicate:cycle": 1,
                "duplicate:cycle:open": 1,
                "duplicate:same-layer": 2,
                "duplicate:same-layer:closed": 2,
                "met:length:3": 2,
                "met:longer-by:0": 2,
                "met:states": 1,
                "guard:close": 1,
            },
        }
    ]
    summary = summarise(rows, beam_width=4)
    assert summary["ranked_successors"] == 11
    assert summary["closed_cross_path_per_closed_retained"] == 2 / 5
    assert summary["effective_width"] == 7 / 8
    assert summary["tasks_with_closed_cross_path"] == 1
    assert summary["closed_states_met_per_closed_retained"] == 1 / 5
    assert summary["met_by_length"] == {"3": 2}
    text = markdown(summary, "unit")
    assert "**40.0%**" in text and "| same-layer | 0 | 2 |" in text
    assert "| 3 | 2 | | 0 | 2 |" in text
