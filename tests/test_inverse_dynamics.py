"""inverse-dynamics-probe (#166): triples, features and the three encoders."""

import torch

import experiments.inverse_dynamics as idp
from wandering_light.stack_beam import step_all_quiet
from wandering_light.stack_machine import Int, Lst
from wandering_light.stack_policy import OP_NAMES

PRODUCT = (
    (Lst((Int(1), Int(2))), Int(3), Int(4)),
    (Lst((Int(5),)), Int(6), Int(7)),
)


def test_every_label_set_is_exactly_the_moves_that_reach_the_target():
    rows = idp.successors(PRODUCT)
    assert rows
    by_name = {op.name: op for op in idp.MOVES}
    for row in rows:
        assert row["op"] in row["set"]
        for index, op in enumerate(OP_NAMES):
            if op not in by_name:
                continue
            reaches = step_all_quiet(PRODUCT, by_name[op])[0] == row["t"]
            assert reaches == (index in row["set"])


def test_structural_ops_are_never_moves():
    assert all(op.name not in ("[", "]") for op in idp.MOVES)


def test_flags_compare_from_the_top():
    s, t = (Int(1), Int(2), Int(3)), (Int(1), Int(3), Int(2))
    got = (
        [(f - idp.FLAG_BASE) // 4 for f in idp.flags(s, t)],
        [(f - idp.FLAG_BASE) % 4 for f in idp.flags(s, t)],
    )
    assert got == ([0, 1, 2], [idp.CHANGED, idp.CHANGED, idp.EQUAL])
    shorter = [(f - idp.FLAG_BASE) % 4 for f in idp.flags(s, (Int(1), Int(2)))]
    assert shorter == [idp.CHANGED, idp.CHANGED, idp.REMOVED]
    longer = [(f - idp.FLAG_BASE) % 4 for f in idp.flags((Int(1),), (Int(1), Int(2)))]
    assert longer == [idp.CHANGED, idp.ADDED]


def test_each_arm_scores_the_28_ops():
    rows = idp.successors(PRODUCT)[:3]
    for arm in idp.ARMS:
        model = idp.build(arm)
        out = idp.logits(model, idp.featurise(rows, arm), arm, "cpu")
        assert out.shape == (3, len(OP_NAMES))


def test_aligned_block_holds_the_top_items_of_both_stacks():
    row = idp.successors(PRODUCT)[0]
    block = idp.aligned_block(row)
    assert block["values"].shape[0] == 2 * idp.TOP * len(PRODUCT)
    assert block["depth"].tolist() == [
        [len(s), len(t)] for s, t in zip(row["s"], row["t"], strict=True)
    ]


def test_set_nll_is_zero_when_all_mass_is_on_the_set():
    out = torch.full((1, len(OP_NAMES)), -1e4)
    out[0, 3] = out[0, 5] = 0.0
    assert idp.set_nll(out, [(3, 5)]).item() < 1e-6
    assert idp.set_nll(out, [(4,)]).item() > 100
