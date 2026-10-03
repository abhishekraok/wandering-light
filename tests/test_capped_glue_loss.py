"""Synthetic only: ownership, mid-span support, weighting, and refusal scope."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import capped_glue_loss as glue
from experiments.hindsight_subtargets import replay_states
from experiments.segment_composite import levels, segments
from experiments.staged_transfer import step_kinds
from wandering_light import stack_beam
from wandering_light.stack_machine import Int, Lst, Mark, Sym
from wandering_light.stack_pbe import BASIS, OPS, Task


def task(program, outputs, *, behaviour="fixture", skeleton="original", inputs=("2",)):
    program = tuple(program.split())
    return Task(
        behaviour=behaviour,
        summary="synthetic fixture",
        split="train",
        task_hash_split="train",
        inputs=inputs,
        outputs=tuple(outputs),
        # These malformed sentinels must never be parsed by this diagnostic.
        held_out_inputs=("(",),
        held_out_outputs=(")",),
        program=program,
        length=len(program),
        examples=len(inputs),
        held_out=1,
        fuel=0,
        tags=(),
        skeleton=skeleton,
        basis=BASIS,
    )


def rows(program, outputs, **kwargs):
    t = task(program, outputs, **kwargs)
    return glue.occurrences(t, glue.replay_shown(t))


def test_nested_ownership_matches_levels_and_actual_marks():
    t = task("[ [ dup ] i ] i", ("2 2",))
    replay = glue.replay_shown(t)
    assert replay.states == tuple(replay_states(t))
    assert replay.kinds == tuple(step_kinds(t))
    assert [(owner.depth, owner.opener) for owner in replay.owners] == [
        (0, None),
        (1, 0),
        (2, 1),
        (2, 1),
        (1, 0),
        (1, 0),
        (0, None),
    ]
    # Assert typed nesting, rather than trusting visual bracket syntax.
    assert replay.states[4][0][-1] == Lst((Sym("dup"),))
    assert replay.states[6][0][-1] == Lst((Lst((Sym("dup"),)), Sym("i")))
    top, bodies = levels(t.program, [1.0] * 7, replay.kinds)
    assert top == [2.0] and bodies == [2.0, 3.0]
    assert sum(owner.depth == 0 for owner in replay.owners) == 2


@pytest.mark.parametrize("program", ["]", "[ dup", "[ [ ]", "foreign-op"])
def test_malformed_program_is_invalid(program):
    with pytest.raises(glue.InvalidDiagnosticError):
        glue.lexical_owners(program.split())


def test_mark_cross_check_covers_every_shown_example_and_ignores_nested_marks():
    glue.check_mark_depth(((Lst((Mark(),)),),), glue.Owner(0, None), context="nested")
    with pytest.raises(glue.InvalidDiagnosticError, match="example 1"):
        glue.check_mark_depth(((Mark(),), ()), glue.Owner(1, 0), context="second")


def test_replay_checks_original_target_all_shown_examples_and_train_split():
    t = task("dup", ("2 2", "3 3"), inputs=("2", "3"))
    assert glue.replay_shown(t).states[-1] == ((Int(2), Int(2)), (Int(3), Int(3)))
    with pytest.raises(glue.InvalidDiagnosticError, match="misses shown target"):
        glue.replay_shown(replace(t, outputs=("2 2", "3")))
    with pytest.raises(glue.InvalidDiagnosticError, match="train task"):
        glue.replay_shown(replace(t, split="eval"))
    with pytest.raises(glue.InvalidDiagnosticError, match="example 1"):
        glue.replay_shown(replace(t, inputs=("2", "")))


def test_span_positions_keep_full_body_and_final_tail():
    t = task("dup dup [ dup dup ] i dup", ("2 2 2 2 2 2",))
    replay = glue.replay_shown(t)
    rs = glue.occurrences(t, replay)
    assert [row.position for row in rs] == [
        "first",
        "middle",
        "middle",
        "middle",
        "middle",
        "middle",
        "last",
        "only",
    ]
    assert [row.index_in_span for row in rs] == [0, 1, 2, 3, 4, 5, 6, 0]
    assert [row.span_length for row in rs] == [7] * 7 + [1]
    assert [row.span for row in rs] == [0] * 7 + [1]
    assert segments([1.0] * 8, replay.kinds) == [7.0, 1.0]
    (cell,) = glue.matched_cells(rs)
    assert cell.op == "dup" and cell.position == "middle"
    assert cell.top == (("fixture", 1),)
    assert cell.body == (("fixture", 3), ("fixture", 4))
    assert rs[1].before_opener
    assert not rs[7].before_opener


def test_boundary_glue_and_structural_ops_cannot_enter_primary():
    rs = rows("dup [ dup ] i dup", ("2 2 2 2",))
    assert glue.matched_cells(rs) == ()  # top dup is first or only
    # An i under a Mark is still structural; matching names must not admit it.
    rs = rows("[ [ dup ] i ] i", ("2 2",))
    assert glue.matched_cells(rs) == ()
    assert all(
        row.position == "middle" for row in rs if row.depth and not row.structural
    )


def test_top_between_closer_and_combinator_is_descriptive():
    rs = rows("dup [ dup ] dup drop i", ("2 2 2",))
    assert rs[4].op == "dup" and rs[4].between_closer_and_combinator
    assert rs[5].op == "drop" and rs[5].between_closer_and_combinator
    (cell,) = glue.matched_cells(rs)
    assert cell.top == (("fixture", 4),) and cell.body == (("fixture", 2),)


def test_exact_ops_not_aliases_match_and_duplicate_identity_is_invalid():
    rs = rows("dup 1 [ dup ] i", ("2 2 1 1",))
    assert glue.matched_cells(rs) == ()  # middle literal 1 does not match dup
    with pytest.raises(glue.InvalidDiagnosticError, match="duplicate occurrence"):
        glue.matched_cells(rs + rs)


def test_group_fallback_is_original_behaviour():
    rs = rows("dup dup [ dup ] i", ("2 2 2 2",), skeleton="")
    (cell,) = glue.matched_cells(rs)
    assert cell.group == "fixture"


def test_assembly_flag_does_not_change_matching_or_attribute_runtime_work():
    # The bracketed quote is discarded as data; top quote actually wraps a value.
    t = task("dup quote [ quote ] drop", ("2 (2)",))
    replay = glue.replay_shown(t)
    assert replay.states[-1] == ((Int(2), Lst((Int(2),))),)
    (cell,) = glue.matched_cells(glue.occurrences(t, replay))
    assert cell.op == "quote" and cell.assembly
    assert cell.top == (("fixture", 1),) and cell.body == (("fixture", 3),)
    out = glue.aggregate_cells((cell,), {cell.top[0]: 2.0, cell.body[0]: 1.0})
    assert out["delta"] == 1.0 and out["cells"][0]["assembly"]


def test_inconsistent_original_group_is_invalid():
    rs = rows("dup dup [ dup ] i", ("2 2 2 2",))
    with pytest.raises(
        glue.InvalidDiagnosticError, match="inconsistent original groups"
    ):
        glue.matched_cells((*rs[:-1], replace(rs[-1], group="changed")))


def test_support_counts_independent_groups_not_aliases_or_seed_copies():
    (base,) = glue.matched_cells(rows("dup dup [ dup ] i", ("2 2 2 2",)))
    shared = [replace(base, behaviour=f"alias{i}", group="shared") for i in range(40)]
    assert glue.support(shared)["passes"] is False
    assert glue.support(shared)["groups"] == ["shared"]
    independent = [
        replace(base, behaviour=f"b{i}", group=f"g{i // 2}") for i in range(40)
    ]
    assert glue.support(independent)["passes"] is True
    assert glue.support(independent[:-1])["passes"] is False  # only 39 total cells
    nineteen = [replace(cell, group=f"g{i % 19}") for i, cell in enumerate(independent)]
    assert glue.support(nineteen)["passes"] is False


def test_means_use_every_occurrence_equal_cells_then_equal_behaviours():
    a_dup = glue.MatchedCell(
        "a", "ga", "dup", "middle", (("a", 0),), (("a", 1), ("a", 2))
    )
    a_one = glue.MatchedCell(
        "a", "ga", "1", "middle", (("a", 3), ("a", 4)), (("a", 5),)
    )
    b_dup = glue.MatchedCell("b", "gb", "dup", "middle", (("b", 0),), (("b", 1),))
    losses = {
        ("a", 0): 6,
        ("a", 1): 1,
        ("a", 2): 3,
        ("a", 3): 2,
        ("a", 4): 4,
        ("a", 5): 3,
        ("b", 0): 0,
        ("b", 1): 2,
    }
    out = glue.aggregate_cells((a_dup, a_one, b_dup), losses)
    assert [cell["delta"] for cell in out["cells"]] == [4, 0, -2]
    assert out["behaviours"] == {"a": 2, "b": -2}
    assert out["delta"] == 0 and out["positive_behaviour_fraction"] == 0.5
    # Exposure weighting or treating all cells as tasks would give a different answer.
    assert sum(cell["delta"] for cell in out["cells"]) / 3 != out["delta"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_losses_never_become_a_negative(value):
    (cell,) = glue.matched_cells(rows("dup dup [ dup ] i", ("2 2 2 2",)))
    losses = {cell.top[0]: value, cell.body[0]: 0.0}
    with pytest.raises(glue.InvalidDiagnosticError, match="nonfinite"):
        glue.aggregate_cells((cell,), losses)
    with pytest.raises(glue.InvalidDiagnosticError, match="missing"):
        glue.aggregate_cells((cell,), {})


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("example 2: map: i: out of fuel", "fuel"),
        ("example 1: i: quotation nested too deep", "depth"),
        ("example 0: add: needs 2 value(s), stack has 1", "ordinary"),
        ("example 0: unknown op 'out of fuel'", "ordinary"),
    ],
)
def test_resource_reason_classification_preserves_wrapper_scope(reason, expected):
    assert glue.refusal_class(reason) == expected


def test_legal_refusal_records_only_first_refusing_example():
    out = glue.legal_alternatives(((Int(2),), (), ()), "0")
    (refusal,) = [row for row in out["refusals"] if row["op"] == "dup"]
    assert refusal["reason"].startswith("example 1:")
    assert refusal["class"] == "ordinary"
    assert len(out["mask"]) == 28
    assert sum(out["first_refusal_counts"].values()) == len(out["refusals"])
    assert not out["resource_affected"]


def test_body_mask_accepts_all_ops_without_executing_them():
    out = glue.legal_alternatives(((Mark(),),), "dup")
    assert all(out["mask"]) and len(out["mask"]) == len(OPS)
    assert out["refusals"] == [] and not out["resource_affected"]


def test_real_prefixed_depth_refusal_is_masked_and_flagged():
    quotation = Lst((Sym("dup"),))
    for _ in range(34):
        quotation = Lst((quotation, Sym("i")))
    out = glue.legal_alternatives(((Int(2), quotation),), "0")
    (refusal,) = [row for row in out["refusals"] if row["op"] == "i"]
    assert refusal["class"] == "depth" and refusal["reason"].startswith("example 0:")
    assert out["resource_affected"] and out["first_refusal_counts"]["depth"] >= 1


def test_fuel_refusal_of_alternative_is_masked_but_witness_is_invalid(monkeypatch):
    # A deliberately low fixture allocation exercises the actual reason wrapper.
    monkeypatch.setattr(stack_beam, "DEFAULT_FUEL_LIMIT", 1)
    product = ((Int(2), Lst((Sym("dup"), Sym("dup")))),)
    out = glue.legal_alternatives(product, "0")
    (refusal,) = [row for row in out["refusals"] if row["op"] == "i"]
    assert refusal["class"] == "fuel" and out["resource_affected"]
    with pytest.raises(glue.InvalidDiagnosticError, match="witness i refused"):
        glue.legal_alternatives(product, "i")
    t = task("[ dup dup ] i", ("2 2 2",))
    with pytest.raises(glue.InvalidDiagnosticError, match="witness step 4 i"):
        glue.replay_shown(t)


def test_cooperative_deadline_refuses_before_work():
    t = task("dup", ("2 2",))
    with pytest.raises(glue.InvalidDiagnosticError, match="deadline"):
        glue.replay_shown(t, deadline=0)
    with pytest.raises(glue.InvalidDiagnosticError, match="deadline"):
        glue.legal_alternatives(((Int(2),),), "dup", deadline=0)
