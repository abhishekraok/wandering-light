"""The pure helpers of `experiments.cycle_deficit`: no corpus, no policy."""

from experiments.cycle_deficit import cycle_spans, library


def spans(program: str) -> list[str]:
    tokens = tuple(program.split())
    return [" ".join(tokens[a:b]) for a, b in cycle_spans(tokens)]


def test_a_cycle_is_adjacent_quotations_then_a_combinator():
    assert spans("0 [ add ] fold") == ["[ add ] fold"]
    assert spans("[ ] [ drop 1 ] [ drop 0 ] ifte") == ["[ ] [ drop 1 ] [ drop 0 ] ifte"]


def test_nested_cycles_belong_to_the_outer_one():
    assert spans("[ 0 [ add ] fold ] map") == ["[ 0 [ add ] fold ] map"]


def test_an_op_between_the_quotation_and_its_combinator_breaks_the_cycle():
    assert spans("[ lt? ] swap cons filter") == []
    assert spans("[ a ] quote [ b ] map") == ["[ b ] map"]


def test_library_ranks_by_ops_saved_not_frequency():
    short = ("[", "add", "]", "fold")
    long = (
        "[",
        "[",
        "lt?",
        "]",
        "[",
        "swap",
        "]",
        "[",
        "]",
        "ifte",
        "drop",
        "]",
        "fold",
    )
    programs = [short, short, short, long, long]
    assert library(programs, 1) == [long]
    assert library(programs, 2) == [long, short]
