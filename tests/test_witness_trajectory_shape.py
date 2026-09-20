from types import SimpleNamespace

from experiments.witness_trajectory_shape import BASIS, describe, replay, run_lengths
from wandering_light.stack_machine import parse_stack


def _kinds(program: str, stack: str) -> str:
    steps = replay(program.split(), parse_stack(stack, basis=BASIS))
    return "".join(step["kind"] for step in steps)


def test_everything_under_a_mark_is_a_write_including_literals_and_nested_closes():
    # `0` is a literal, which is not captured as a symbol but still lands in the body.
    assert _kinds("[ 0 [ add ] i ] i", "[ 5 ]") == "WWWWWWWX"


def test_top_level_ops_are_direct_and_combinators_execute():
    assert _kinds("0 [ swap drop ] fold", "[ (5 -1) ]") == "DWWWWX"


def test_execution_records_body_size_and_hidden_fuel():
    steps = replay(
        ["0", "[", "add", "]", "fold"], parse_stack("[ (1 2 3) ]", basis=BASIS)
    )
    assert steps[-1]["kind"] == "X"
    assert steps[-1]["body"] == 1
    assert steps[-1]["fuel"] > 1
    assert steps[-1]["stack"] == parse_stack("[ 6 ]", basis=BASIS)


def test_run_lengths():
    assert run_lengths("DWWWWX") == "D1 W4 X1"


def test_ifte_body_counts_all_three_quotations():
    program = [
        "sub",
        "[",
        "dup",
        "0",
        "lt?",
        "]",
        "[",
        "0",
        "swap",
        "sub",
        "]",
        "[",
        "]",
        "ifte",
    ]
    steps = replay(program, parse_stack("[ -3 5 ]", basis=BASIS))
    assert steps[-1]["body"] == 6


def test_peak_size_includes_the_initial_state():
    task = SimpleNamespace(
        behaviour="add_pair", split="train", inputs=("[ 100 -100 ]",), program=("add",)
    )
    assert describe(task)["peak_size_ratio"] == 1.0
