"""The `beam_solve` speedups are semantics-preserving (solved-program-cache, #151).

`0635935` replaced `key_of(product)` with the product itself in two places: the
hit test (`key_of(moved) == target` became `moved == outputs`) and the visited
set. Both rest on one claim -- **structural `==` on the closed `Value` union
induces exactly `value_key`'s partition** -- and every wall-clock and coverage
cell measured after that commit is wrong if the claim is wrong, silently,
because a divergence would change which nodes are pruned rather than raise.

`910cfa6` then turned the cyclic collector off for the duration of a search,
which cannot change semantics but is asserted here to be restored afterwards.

These tests pin the claim rather than the speed.
"""

from __future__ import annotations

import gc
import random

import numpy as np

from wandering_light.stack_beam import (
    beam_solve,
    key_of,
    step_all_quiet,
)
from wandering_light.stack_machine import (
    Bool,
    Int,
    Lst,
    Mark,
    Sym,
    parse_stack,
    value_key,
)
from wandering_light.stack_pbe import BASIS, BY_NAME, OPS, build_task, draw_task
from wandering_light.stack_policy import OP_NAMES

# The pairs a careless implementation gets wrong: `True == 1` in plain Python,
# a symbol whose text is a digit, a symbol spelled like a mark, an empty
# quotation against a nil-ish symbol, and each of those nested one level down.
ADVERSARIAL = (
    Int(0),
    Bool(False),
    Int(1),
    Bool(True),
    Sym("0"),
    Sym("1"),
    Sym("MARK"),
    Mark(),
    Lst(()),
    Sym("nil"),
    Int(2**70),
    Int(-(2**70)),
    Lst((Int(1),)),
    Lst((Bool(True),)),
    Lst((Sym("1"),)),
    Lst((Lst(()),)),
)


def test_equality_and_value_key_agree_on_adversarial_pairs():
    """No two values compare equal unless their keys do, and vice versa."""
    for left in ADVERSARIAL:
        for right in ADVERSARIAL:
            assert (left == right) is (value_key(left) == value_key(right)), (
                f"{left!r} vs {right!r}"
            )


def test_equal_values_hash_alike():
    """The visited set is a set, so `__eq__` and `__hash__` must agree."""
    for left in ADVERSARIAL:
        for right in ADVERSARIAL:
            if left == right:
                assert hash(left) == hash(right), f"{left!r} vs {right!r}"


def test_the_two_partitions_of_reachable_products_have_the_same_size():
    """On products the search actually walks, `==` and `key_of` partition alike.

    Equal cardinality is the load-bearing check: the visited set dedups by
    `==` where it used to dedup by `key_of`, so a coarser or finer partition
    would prune a different set of nodes.
    """
    rng = random.Random(151)
    products = []
    for name in ("negate_each", "reverse", "min_pair", "keep_below", "drop_last"):
        task = build_task(BY_NAME[name], draw_task(BY_NAME[name], rng, 4, held_out=6))
        root = tuple(parse_stack(text, basis=BASIS) for text in task.inputs)
        products.append(root)
        # Most ops refuse most states, so walk breadth-first from the root
        # rather than following one chain: a chain dies at the first refusal
        # and collects almost nothing.
        frontier = [root]
        for _ in range(4):
            nxt = []
            for product in frontier:
                for op in OPS:
                    moved, _spent, _stepped = step_all_quiet(product, op)
                    if moved is not None:
                        products.append(moved)
                        nxt.append(moved)
            rng.shuffle(nxt)
            frontier = nxt[:12]
    assert len(products) > 200, len(products)

    by_key = {key_of(p) for p in products}
    # `Product` is a tuple of tuples of frozen values, so it is hashable and a
    # set of products is exactly the `==` partition.
    by_eq = set(products)
    assert len(by_eq) == len(by_key)

    for left in products[:120]:
        for right in products[:120]:
            assert (left == right) is (key_of(left) == key_of(right))


class _Deterministic:
    """A scorer with no model: seeded logits, reproducible across processes.

    `beam_solve` only needs `log_probs(blocks, legal)`, so a checkpoint is not
    required to compare two search implementations step for step.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed

    def log_probs(self, blocks, legal):
        rng = np.random.default_rng(self.seed + len(blocks))
        raw = rng.standard_normal((len(blocks), len(OP_NAMES)))
        masked = np.where(legal, raw, -np.inf)
        shifted = masked - masked.max(axis=1, keepdims=True)
        exp = np.exp(np.where(np.isfinite(shifted), shifted, -np.inf))
        with np.errstate(divide="ignore"):  # an illegal op is -inf by design
            return np.log(exp / exp.sum(axis=1, keepdims=True))


def test_the_collector_is_restored_after_a_search():
    task = build_task(
        BY_NAME["negate_each"],
        draw_task(BY_NAME["negate_each"], random.Random(3), 4, 6),
    )
    was_enabled = gc.isenabled()
    beam_solve(task, _Deterministic(), beam_width=4, max_depth=3)
    assert gc.isenabled() is was_enabled


def test_a_search_is_reproducible_under_a_fixed_scorer():
    """Guards the differential harness itself: the search is a pure function."""
    rng = random.Random(9)
    task = build_task(BY_NAME["reverse"], draw_task(BY_NAME["reverse"], rng, 4, 6))
    first = beam_solve(task, _Deterministic(5), beam_width=8, max_depth=6)
    second = beam_solve(task, _Deterministic(5), beam_width=8, max_depth=6)
    for field in (
        "program",
        "termination",
        "executions",
        "fuel",
        "expansions",
        "depth",
    ):
        assert getattr(first, field) == getattr(second, field), field
