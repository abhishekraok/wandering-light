"""Task selection, the witness overlay and play mode (stack-rollout-debugger, #115)."""

import random

import pytest

from wandering_light.rollout_policy import PolicyArchitecture
from wandering_light.stack_debugger.session import (
    SearchSettings,
    build_session,
    cohort_indices,
    encoding_view,
    parse_product,
    select_task,
    token_names,
)
from wandering_light.stack_debugger.walk import inner_frames, play_program
from wandering_light.stack_machine import parse_stack
from wandering_light.stack_pbe import BASIS, BY_NAME, OP_BY_NAME, build_task, draw_task
from wandering_light.stack_policy import VOCAB, PolicyScorer, StackPairPolicy

TINY = PolicyArchitecture(8, 16, 32)


def task_for(name: str, seed: int = 4, examples: int = 3):
    rng = random.Random(seed)
    return build_task(
        BY_NAME[name], draw_task(BY_NAME[name], rng, examples, held_out=6)
    )


@pytest.fixture
def tasks():
    return [
        task_for("add_pair", 6),
        task_for("sum"),
        task_for("min_pair", 2),
        task_for("keep_negative", 3),
    ]


def test_select_by_index_and_by_filter(tasks):
    index, provenance = select_task(tasks, task=2)
    assert index == 2 and provenance["mode"] == "task"
    index, provenance = select_task(tasks, behaviour="sum", seed=1)
    assert tasks[index].behaviour == "sum" and provenance["pool_size"] == 1
    index, _ = select_task(tasks, bucket="1-5", seed=1)
    assert tasks[index].length <= 5
    with pytest.raises(ValueError):
        select_task(tasks, behaviour="no_such_behaviour")
    with pytest.raises(KeyError):
        select_task(tasks, task=99)


def test_cohort_rule_matches_the_campaign_driver():
    """The #101 cohort is restated here; the driver's own function is the reference."""
    from experiments.compute_optimal_mark28 import cohort

    pool = [task_for(name, seed, examples=3) for seed, name in enumerate(BY_NAME)][:150]
    expected = [index for index, _ in cohort(pool)]
    assert cohort_indices(pool) == expected


def test_session_payload_carries_witness_overlay_and_grading(tasks):
    scorer = PolicyScorer(StackPairPolicy(TINY), "cpu")
    session = build_session(
        scorer,
        tasks,
        0,
        settings=SearchSettings(beam_width=8, max_depth=3),
        provenance={"x": 1},
    )
    payload = session.to_dict()
    assert payload["result"]["program"] == ["add"]
    assert payload["result"]["solved"] is True
    assert payload["witnesses"][0] == ["add"]
    first = payload["witness_overlay"][0]["witnesses"][0]
    assert first["node"] == 0 and first["next"] == "add"
    # Depth 0 has one node and the search fits while expanding it, so the
    # layer is never scored: the overlay says so rather than inventing a rank.
    assert first["rank"] is None
    assert payload["basis"][0]["name"] == "0" and len(payload["basis"]) == 28
    assert payload["task"]["bucket"] == "1-5"


def test_witness_rank_is_reported_from_the_node_on_the_witness(tasks):
    scorer = PolicyScorer(StackPairPolicy(TINY), "cpu")
    session = build_session(
        scorer,
        tasks,
        1,
        settings=SearchSettings(beam_width=16, max_depth=2),
        provenance={},
    )
    payload = session.to_dict()
    overlay = payload["witness_overlay"]
    assert overlay[0]["witnesses"][0]["node"] == 0
    assert overlay[0]["witnesses"][0]["rank"] is not None
    assert (
        1 <= overlay[0]["witnesses"][0]["rank"] <= overlay[0]["witnesses"][0]["legal"]
    )


def test_token_names_cover_the_vocabulary():
    names = token_names()
    assert len(names) == VOCAB
    view = encoding_view(parse_product(["[ (1 2) 300 ]"]), parse_product(["[ 3 ]"]))
    item = view["items"][0]
    assert item["state_tokens"] == ["(", "1", "2", ")", "BIG", "·3", "·0", "·0"]
    assert item["target_tokens"] == ["3"] and item["state_overflow"] == 0


def test_play_steps_a_witness_and_shows_inner_work():
    task = task_for("sum")
    played = play_program(task, task.program)
    assert played["solved_shown"] is True
    assert played["refused"] is None
    assert [step["op"] for step in played["steps"]] == [None, *task.program]
    fold = next(step for step in played["steps"] if step["op"] == "fold")
    assert fold["inner"] is not None
    assert fold["inner"]["agrees_with_machine"] is True
    kinds = [frame["kind"] for frame in fold["inner"]["frames"]]
    assert kinds.count("iter") == len(parse_stack(task.inputs[0], basis=BASIS)[0].items)
    assert kinds[-1] == "leave"


def test_play_reports_a_refusal_instead_of_raising():
    task = task_for("sum")
    played = play_program(task, ["add"])
    assert played["refused"] is not None and played["steps"][-1]["refused"]
    assert played["solved_shown"] is False


@pytest.mark.parametrize(
    ("stack", "op"),
    [
        ("[ (1 2 3) (dup mul) ]", "map"),
        ("[ (1 -2 3) (0 lt?) ]", "filter"),
        ("[ (1 2 3) 0 (add) ]", "fold"),
        ("[ 1 2 (over lt?) (swap drop) (drop) ]", "ifte"),
        ("[ 5 (1 add) ]", "i"),
        ("[ 1 5 (1 add) ]", "dip"),
        ("[ 1 2 3 ((add) dip add) ]", "i"),
    ],
)
def test_inner_frames_agree_with_the_machine(stack, op):
    walked = inner_frames(OP_BY_NAME[op], parse_stack(stack, basis=BASIS))
    assert walked["agrees_with_machine"] is True
    assert walked["frames"]


def test_anatomy_gives_each_quotation_a_role_and_examples():
    from wandering_light.stack_debugger.anatomy import anatomy, spans

    task = task_for("sum_negative") if "sum_negative" in BY_NAME else task_for("sum")
    program = [
        "0",
        "[",
        "[",
        "dup",
        "0",
        "lt?",
        "]",
        "[",
        "add",
        "]",
        "[",
        "drop",
        "]",
        "ifte",
        "]",
        "fold",
    ]
    tree = spans(program)
    assert [(n["start"], n["end"], n["depth"], n["parent"]) for n in tree] == [
        (1, 14, 0, None),
        (2, 6, 1, 0),
        (7, 9, 1, 0),
        (10, 12, 1, 0),
    ]
    result = anatomy(task, program)
    nodes = {n["body"]: n for n in result["nodes"]}
    assert nodes["(dup 0 lt?)"]["roles"] == ["ifte condition"]
    assert nodes["(add)"]["roles"] == ["ifte then"]
    assert nodes["(drop)"]["roles"] == ["ifte else"]
    assert nodes["((dup 0 lt?) (add) (drop) ifte)"]["roles"] == ["fold body"]
    condition = nodes["(dup 0 lt?)"]
    assert condition["runs"] == nodes["((dup 0 lt?) (add) (drop) ifte)"]["runs"] > 0
    assert all(
        row["output"] in ("[ true ]", "[ false ]")
        or row["output"].endswith(("true ]", "false ]"))
        for row in condition["examples"]
    )
    assert result["runtime"] == []


def test_anatomy_lists_quotations_built_at_runtime():
    from wandering_light.stack_debugger.anatomy import anatomy

    task = task_for("sum")
    # `quote` builds a one-element quotation the program never wrote down.
    program = ["[", "add", "]", "i"]  # not a solver, just a literal body
    assert anatomy(task, program)["runtime"] == []
    built = anatomy(task, ["uncons", "quote", "i"])
    assert built["nodes"] == []
    assert built["runtime"] and built["runtime"][0]["roles"] == ["i body"]
