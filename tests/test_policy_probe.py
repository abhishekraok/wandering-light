"""The probe's conclusion rests on its ranking metric. Pin that."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from policy_probe import evaluate, rank_of_first_optimal


def test_uniform_scores_rank_worst_not_best():
    """The tie-break must be pessimistic.

    An earlier version sorted optimal actions first among ties, which made a
    uniform predictor score 100% top-1. Any predictor that carries no
    information must land at the bottom.
    """
    flat = dict.fromkeys("abcdef", 0.0)
    assert rank_of_first_optimal(flat, {"d"}) == 6
    assert rank_of_first_optimal(flat, {"a", "f"}) == 5


def test_rank_follows_the_score_order():
    scores = {"a": 0.1, "b": 0.9, "c": 0.0}
    assert rank_of_first_optimal(scores, {"b"}) == 1
    assert rank_of_first_optimal(scores, {"a"}) == 2
    assert rank_of_first_optimal(scores, {"c"}) == 3


def test_any_optimal_action_counts():
    """Tasks can have several optimal first actions; hitting one is enough."""
    scores = {"a": 0.3, "b": 0.2, "c": 0.1}
    assert rank_of_first_optimal(scores, {"b", "c"}) == 2


def test_missing_optimal_action_ranks_past_the_end():
    scores = {"a": 1.0, "b": 0.5}
    assert rank_of_first_optimal(scores, {"zzz"}) == 3


FIRST = "optimal_first_action_names"


def _task(distance, optimal, field=FIRST):
    return {"metadata": {"certified_distance": distance, field: optimal}}


def test_evaluate_reports_top1_and_distance_breakdown():
    tasks = [_task(1, ["a"]), _task(2, ["b"]), _task(2, ["a"])]
    perfect = evaluate(lambda t: {"a": 1.0, "b": 0.0}, tasks, "x", FIRST)
    assert perfect["top1"] == pytest.approx(2 / 3)
    assert perfect["by_distance"][1] == pytest.approx(1.0)
    assert perfect["by_distance"][2] == pytest.approx(1.5)


def test_evaluate_is_not_fooled_by_a_flat_predictor():
    tasks = [_task(3, ["a"]) for _ in range(4)]
    flat = evaluate(lambda t: {"a": 0.0, "b": 0.0, "c": 0.0}, tasks, "flat", FIRST)
    assert flat["top1"] == 0.0
    assert flat["mean_rank"] == 3.0


def test_last_action_masks_on_output_type_not_input():
    """The two ends are not symmetric.

    A first action must consume the task's input type; a last action must
    produce its output type. Masking a last-action probe on input type would
    compare predictors against a mask that does not apply to them.
    """
    from policy_probe import action_field

    names_field, complete_flag = action_field(last=True)
    assert names_field == "optimal_last_action_names"
    assert complete_flag == "optimal_last_actions_complete"
    assert action_field(last=False) == (
        "optimal_first_action_names",
        "optimal_first_actions_complete",
    )


def test_type_mask_uses_the_field_it_is_given():
    from policy_probe import make_type_mask

    type_of = {"a": "builtins.int", "b": "builtins.str"}
    task = {"metadata": {"input_type": "builtins.int", "output_type": "builtins.str"}}
    assert make_type_mask(["a", "b"], type_of, "input_type")(task) == {
        "a": 1.0,
        "b": 0.0,
    }
    assert make_type_mask(["a", "b"], type_of, "output_type")(task) == {
        "a": 0.0,
        "b": 1.0,
    }


# ---------------------------------------------------------------------------
# The pair encoder
# ---------------------------------------------------------------------------

import json  # noqa: E402

from policy_probe import (  # noqa: E402
    ITEM_CAP,
    MAX_LEN,
    PAD,
    SEP,
    build_encoder,
    encode_flat,
    encode_pairs,
    pair_batch,
    truncation_rates,
)


def _pair_task(inputs, outputs, in_type="builtins.int", out_type="builtins.str"):
    return {
        "input": json.dumps({"type": in_type, "items": inputs}),
        "output": json.dumps({"type": out_type, "items": outputs}),
    }


def test_encode_pairs_aligns_the_ith_input_with_the_ith_output():
    """Every basis function is elementwise, so `items[i]` on the two sides are
    the same example. Pairing them by anything but index would be pairing an
    input with some other input's answer."""
    pairs, in_type, out_type = encode_pairs(
        _pair_task([1, 22, 333], ["a", "bb", "ccc"])
    )
    assert (in_type, out_type) == ("builtins.int", "builtins.str")
    assert len(pairs) == 3
    assert pairs[0] == [*b"1", SEP, *b'"a"']
    assert pairs[2] == [*b"333", SEP, *b'"ccc"']


def test_the_pair_encoder_keeps_a_tail_the_flat_one_truncates():
    """The confound this replaces: 46.0% of deep-corpus-v1 tasks overrun the
    flat encoder's 256 bytes, so on those the model never sees the end of the
    target it is supposed to be predicting a route to."""
    task = _pair_task(list(range(12)), [f"example-{i:03d}" for i in range(12)])
    flat = bytes(b for b in encode_flat(task) if b)
    assert len(f"{task['input']}\x00{task['output']}".encode()) > MAX_LEN
    assert b"example-011" not in flat  # the last example is cut off entirely

    pairs, _, _ = encode_pairs(task)
    assert len(pairs) == 12
    last = pairs[-1]
    assert bytes(last[last.index(SEP) + 1 :]) == b'"example-011"'


def test_the_item_cap_truncates_only_the_item_that_overruns_it():
    pairs, _, _ = encode_pairs(_pair_task([1, 2], ["x" * 200, "y"]))
    long_side = pairs[0][pairs[0].index(SEP) + 1 :]
    assert len(long_side) == ITEM_CAP
    assert pairs[1] == [*b"2", SEP, *b'"y"']


def test_truncation_rates_report_both_budgets():
    short = _pair_task([1], ["a"])
    long = _pair_task(list(range(12)), ["z" * 200] * 12)
    rates = truncation_rates([short, long])
    assert rates["flat_tasks_over_budget"] == 0.5
    # 12 overlong outputs out of 2 + 24 serialized items
    assert rates["pairs_items_over_budget"] == pytest.approx(12 / 26)
    assert rates["pairs_items_dropped"] == 0.0


def test_pair_batch_marks_exactly_the_real_items():
    """The cross-item pool divides by the item count, so a padded cell counted
    as real would silently shrink every task batched beside a longer one."""
    tasks = [_pair_task([1, 2, 3], ["a", "b", "c"]), _pair_task([1], ["a"])]
    x, counts, item_mask, tags = pair_batch(
        tasks, {"builtins.int": 1, "builtins.str": 2}
    )
    assert counts.tolist() == [3, 1]
    assert item_mask.shape == (2, 3)
    assert item_mask.tolist() == [[True, True, True], [True, False, False]]
    assert x.shape[0] == 4  # only real items reach the trunk
    assert tags.tolist() == [[1, 2], [1, 2]]


def test_unknown_types_fall_back_to_the_reserved_row():
    _, _, _, tags = pair_batch([_pair_task([1], ["a"], in_type="builtins.mystery")], {})
    assert tags.tolist() == [[0, 0]]


def _logits(model, tasks):
    import torch

    with torch.no_grad():
        return model(*pair_batch(tasks, {"builtins.int": 1, "builtins.str": 2}))


def test_a_task_encodes_the_same_alone_as_in_a_mixed_batch():
    """The failure mode that would silently corrupt every number here.

    Items vary 3-16 per task and bytes vary 2-800 per item, so a batch is
    padded on both axes. If padding leaked into either pool, a model would
    train on one representation and be scored on another, and nothing in the
    output would say so.
    """
    import torch

    torch.manual_seed(0)
    model = build_encoder("pairs", n_names=5, conv=True, n_types=2).eval()
    small = _pair_task([1, 2, 3], ["a", "b", "c"])
    large = _pair_task(list(range(9)), ["long-string-" * 4] * 9)

    alone = _logits(model, [small])[0]
    batched = _logits(model, [small, large])[0]
    assert torch.allclose(alone, batched, atol=1e-5)


def test_the_cross_item_pool_ignores_item_order():
    """Mean and max are the pool precisely because the examples are a set: the
    program acts on each item independently, so their order carries nothing."""
    import torch

    torch.manual_seed(0)
    model = build_encoder("pairs", n_names=5, conv=True, n_types=2).eval()
    forward = _logits(model, [_pair_task([1, 22, 333], ["a", "bb", "ccc"])])
    reversed_ = _logits(model, [_pair_task([333, 22, 1], ["ccc", "bb", "a"])])
    assert torch.allclose(forward, reversed_, atol=1e-5)


def test_padding_is_not_a_byte_value():
    """Byte 0 is the flat encoder's pad *and* its separator. The pair encoder
    needs a pad it can mask on, so it must live outside 0-255."""
    assert PAD == 256 and SEP == 257
    pairs, _, _ = encode_pairs(_pair_task([0], ["\x00"]))
    assert PAD not in pairs[0]


def test_the_two_encoders_stay_comparable_in_size():
    """The question is whether the model can see more, not whether it is
    bigger, so the pair encoder is held near the published model's size."""
    flat = build_encoder("flat", n_names=118, conv=True, n_types=12)
    pairs = build_encoder("pairs", n_names=118, conv=True, n_types=12)
    flat_n = sum(p.numel() for p in flat.parameters())
    pairs_n = sum(p.numel() for p in pairs.parameters())
    assert flat_n == 90582  # the published model, unchanged
    assert pairs_n / flat_n < 1.2


# ---------------------------------------------------------------------------
# Bits per step
# ---------------------------------------------------------------------------

from policy_probe import bits, distribution, evaluate, optimal_mass  # noqa: E402, F811

TYPE_OF = {"a": "builtins.int", "b": "builtins.int", "c": "builtins.str"}


def _typed(optimal, in_type="builtins.int"):
    return {
        "metadata": {
            "certified_distance": 1,
            "input_type": in_type,
            FIRST: optimal,
        }
    }


def test_the_type_mask_scores_exactly_the_mask_width():
    """The anchor, and a free implementation check: a predictor that is uniform
    over the compatible ops and knows nothing else must cost log2(width) bits,
    which is the same quantity the report already gives as substrate branching.
    If this drifts, the column is not subtractable from that."""
    from policy_probe import make_type_mask

    predict = make_type_mask(list(TYPE_OF), TYPE_OF, "input_type")
    row = evaluate(
        predict,
        [_typed(["a"]), _typed(["c"], in_type="builtins.str")],
        "type mask",
        FIRST,
        names=list(TYPE_OF),
        type_of=TYPE_OF,
        match_key="input_type",
    )
    # two compatible ops on the first task, one on the second
    assert row["bits"]["0"]["mean"] == pytest.approx((1.0 + 0.0) / 2)
    assert row["mask_width"] == pytest.approx(1.5)


def test_mass_is_summed_over_the_whole_optimal_set():
    """A policy that prefers an equally short route is not wrong, so the score
    is on the set, not on one witness op."""
    scores = {"a": 0.25, "b": 0.25, "c": 0.5}
    one, _, _ = optimal_mass(scores, {"a"}, ["a", "b", "c"], "renorm")
    both, hits, width = optimal_mass(scores, {"a", "b"}, ["a", "b", "c"], "renorm")
    assert (one, both) == pytest.approx((0.25, 0.5))
    assert (hits, width) == (2, 3)
    assert bits(both, hits, width, 0.0) == pytest.approx(1.0)


def test_eps_interpolation_bounds_an_op_the_predictor_never_saw():
    """A search consumes a policy as an ordering with the whole compatible set
    still reachable, so zero mass is a modelling artefact, not an infinite
    cost. At eps = 0 it is kept as the calibration diagnostic."""
    assert bits(0.0, 1, 4, 0.0) == float("inf")
    assert bits(0.0, 1, 4, 0.1) == pytest.approx(-math.log2(0.1 * 0.25))
    # a confident and correct predictor is barely moved by the floor
    assert bits(1.0, 1, 4, 0.01) == pytest.approx(-math.log2(0.99 + 0.0025))


def test_probability_rungs_are_not_flattened_into_uniform():
    """Softmaxing a rung that already returns probabilities in [0, 0.1] would
    collapse it to nearly uniform, and marginal and conditional would both
    report the mask width -- the column would say nothing."""
    scores = {"a": 0.09, "b": 0.01}
    assert distribution(scores, ["a", "b"], "renorm") == pytest.approx([0.9, 0.1])
    flattened = distribution(scores, ["a", "b"], "softmax")
    assert flattened[0] == pytest.approx(0.52, abs=0.01)


def test_a_predictor_with_no_signal_falls_back_to_uniform():
    assert distribution(dict.fromkeys("abc", 0.0), ["a", "b", "c"], "renorm") == (
        pytest.approx([1 / 3, 1 / 3, 1 / 3])
    )


def test_unreachable_fraction_is_reported_beside_the_mean():
    """One unreachable optimal op would otherwise swallow the whole column."""
    from policy_probe import summarise_bits

    out = summarise_bits([(0.5, 1, 2), (0.0, 1, 2)])
    assert out["0"]["unreachable"] == pytest.approx(0.5)
    assert out["0"]["mean"] == pytest.approx(1.0)  # over the finite tasks only
    assert out["0.1"]["unreachable"] == 0.0


def test_the_pool_is_the_mean_and_max_of_the_items_it_claims_to_hold():
    """Recompute the cross-item pool the long way and require the same answer.

    `test_a_task_encodes_the_same_alone_as_in_a_mixed_batch` shows the pool is
    *consistent*; it would still pass if every task pooled its neighbour's
    items. This pins the value: split the trunk's output by item count with a
    plain loop, take the mean and the max by hand, and push that through the
    head. An off-by-one in the padding path is in range and would not raise --
    it would just return a clean, plausible, wrong number.
    """
    import torch

    torch.manual_seed(0)
    model = build_encoder("pairs", n_names=5, conv=True, n_types=2).eval()
    vocab = {"builtins.int": 1, "builtins.str": 2}
    tasks = [
        _pair_task([1, 2, 3], ["a", "b", "c"]),
        _pair_task(list(range(9)), ["long-string-" * 4] * 9),
        _pair_task([7, 8], ["yy", "zz"]),
    ]
    x, counts, item_mask, tags = pair_batch(tasks, vocab)

    with torch.no_grad():
        items = model.trunk(x, x != PAD)
        start, mean, peak = 0, [], []
        for n in counts.tolist():
            group = items[start : start + n]
            mean.append(group.mean(dim=0))
            peak.append(group.max(dim=0).values)
            start += n
        assert start == items.shape[0]  # every item was claimed exactly once
        by_hand = model.head(
            torch.cat(
                [torch.stack(mean), torch.stack(peak), model.tags(tags).flatten(1)],
                dim=1,
            )
        )
        assert torch.allclose(by_hand, model(x, counts, item_mask, tags), atol=1e-5)


def test_the_wide_arm_is_the_same_model_with_a_bigger_window():
    """The control that separates the two things the pair encoder changes.

    If `wide` differed from `flat` in any parameter, a gain over `flat` could
    be capacity rather than visibility, and the arm would answer nothing.
    """
    from policy_probe import WIDE_LEN

    flat = build_encoder("flat", n_names=118, conv=True, n_types=12)
    wide = build_encoder("wide", n_names=118, conv=True, n_types=12)
    assert [p.shape for p in flat.parameters()] == [p.shape for p in wide.parameters()]

    task = _pair_task(list(range(12)), [f"example-{i:03d}" for i in range(12)])
    assert len(encode_flat(task, MAX_LEN)) == MAX_LEN
    assert len(encode_flat(task, WIDE_LEN)) == WIDE_LEN
    # the tail the published budget cuts is inside the wider one
    assert b"example-011" in bytes(b for b in encode_flat(task, WIDE_LEN) if b)
