from experiments.staged_witnesses.check import measure, within_rules
from experiments.witness_trajectory_shape import BASIS
from wandering_light.stack_machine import parse_stack

ROOT = parse_stack("[ (1 2 3 4) ]", basis=BASIS)
FUSED = [
    "[",
    "dup",
    "mul",
    "dup",
    "add",
    "1",
    "add",
    "1",
    "add",
    "1",
    "add",
    "1",
    "add",
    "]",
    "map",
]
STAGED = [
    "[",
    "dup",
    "mul",
    "]",
    "map",
    "[",
    "dup",
    "add",
    "]",
    "map",
    "[",
    "1",
    "add",
    "1",
    "add",
    "1",
    "add",
    "1",
    "add",
    "]",
    "map",
]


def test_fused_polynomial_breaks_the_quotation_and_write_run_rules():
    m = measure(tuple(FUSED), ROOT)
    assert (m["executions"], m["largest_quotation"], m["longest_write_run"]) == (
        1,
        12,
        14,
    )
    assert len(within_rules(m)) == 2


def test_staged_polynomial_is_within_the_rules_with_three_closed_stages():
    m = measure(tuple(STAGED), ROOT)
    assert m["executions"] == 3
    assert m["post_execution_states"] == 3
    assert within_rules(m) == []


def test_a_parked_quotation_is_counted_until_it_runs():
    # `[ add ]` waits through `swap` and `cons` before `map` consumes it.
    m = measure(("1", "[", "add", "]", "swap", "cons", "map"), ROOT)
    assert m["longest_park"] == 2
