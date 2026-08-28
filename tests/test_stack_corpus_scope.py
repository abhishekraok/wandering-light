"""Pin the corpus-composition figures quoted in `docs/STATUS.md`.

The Stack BFS baseline row carries a scope caveat built from these numbers. If
the corpus changes and they drift, the caveat becomes wrong silently, so it
fails here instead.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from stack_corpus_scope import (
    ARITHMETIC,
    COMBINATOR,
    PREDICATE,
    RUNS_CODE,
    carries_quotation,
    parenthesised,
    resolve,
    uses,
)


@pytest.fixture(scope="module")
def tasks():
    import gzip
    import json

    corpus = resolve([])
    if not corpus.exists():
        pytest.skip(f"corpus not present at {corpus}")
    with gzip.open(corpus, "rt") as handle:
        return [json.loads(line) for line in handle]


def test_corpus_size(tasks):
    assert len(tasks) == 1857


def test_no_program_builds_a_quotation_from_symbols(tasks):
    """Structural: `executable-26` has no symbol pushes, so this must be zero."""
    from wandering_light.stack_machine import CORE_OPS

    names = {op.name for op in CORE_OPS}
    for task in tasks:
        assert set(task["program"]) <= names


def test_published_composition_figures(tasks):
    total = len(tasks)
    quoted = [t for t in tasks if carries_quotation(t)]
    executed = [t for t in quoted if uses(t, RUNS_CODE)]

    assert sum(uses(t, RUNS_CODE) for t in tasks) == 219  # 11.8%
    assert sum(uses(t, COMBINATOR) for t in tasks) == 105  # 5.7%
    assert (
        sum(not uses(t, ARITHMETIC | PREDICATE | COMBINATOR) for t in tasks) == 1026
    )  # 55.3%
    assert len(quoted) == 680  # 36.6%
    assert len(executed) == 158  # 23.2% of those handed one
    assert len(quoted) - len(executed) == 522  # 76.8%
    assert total == 1857


def test_quotation_detector_ignores_bare_values():
    """`true`, `false` and `nil` are data; counting them gives 786, not 680."""
    assert not carries_quotation({"inputs": ["[ (1 0 1) (true nil) ]"]})
    assert carries_quotation({"inputs": ["[ (2 5) (dup mul) ]"]})


def test_parenthesised_handles_nesting():
    assert parenthesised("[ ((1 add) 2 5) ]") == ["1 add", "(1 add) 2 5"]
