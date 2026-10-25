from dataclasses import replace

import pytest
import torch
from torch.nn import functional

from wandering_light.program_embedding import (
    EmbeddingConfig,
    ProgramEmbedding,
    ProgramIndex,
    symmetric_multi_positive_contrastive_loss,
)


@pytest.fixture
def model():
    torch.manual_seed(7)
    return ProgramEmbedding(EmbeddingConfig(hidden_dim=16, byte_dim=8, dim=12))


def test_program_embeddings_preserve_order_and_padding(model):
    single = model.encode_programs([("dup", "drop")])
    batched = model.encode_programs(
        ["dup drop", "longer program with many tokens", "drop dup"]
    )
    torch.testing.assert_close(single[0], batched[0], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(batched[0], batched[2])
    torch.testing.assert_close(batched.norm(dim=1), torch.ones(3))
    torch.testing.assert_close(model.encode_programs([""]), model.encode_programs([()]))
    assert model.encode_programs([]).shape == (0, 12)
    assert model.encode_io([]).shape == (0, 12)


def test_io_pooling_is_permutation_and_batch_invariant(model):
    pairs = [("[1 2]", "[3]"), ("[2 3]", "[5]"), ("[]", "[0]")]
    standalone = model.encode_io([pairs])
    batched = model.encode_io([list(reversed(pairs)), [("a much longer input", "x")]])
    torch.testing.assert_close(standalone[0], batched[0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(batched.norm(dim=1), torch.ones(2))
    assert not torch.allclose(
        model.encode_io([[("a", "b\x00c")]]),
        model.encode_io([[("a\x00b", "c")]]),
    )
    # Full-set duplication leaves the empirical example distribution unchanged.
    torch.testing.assert_close(standalone, model.encode_io([pairs * 2]))


def test_gradients_reach_both_encoders_and_shared_projection(model):
    programs = model.encode_programs(["dup", "drop"])
    examples = model.encode_io([[("[1]", "[1 1]")], [("[1]", "[]")]])
    symmetric_multi_positive_contrastive_loss(programs, examples).backward()
    for name in ("program_encoder", "io_encoder", "projection", "byte_embedding"):
        assert any(
            parameter.grad.abs().sum() > 0
            for parameter in getattr(model, name).parameters()
        )


@pytest.mark.parametrize(
    "field",
    [
        "dim",
        "hidden_dim",
        "byte_dim",
        "num_layers",
        "max_program_bytes",
        "max_example_bytes",
    ],
)
def test_config_rejects_nonpositive_sizes(field):
    with pytest.raises(ValueError, match="positive integer"):
        EmbeddingConfig(**{field: 0})


def test_lengths_are_bytes_and_never_truncated(model):
    model.config = replace(model.config, max_program_bytes=3, max_example_bytes=3)
    model.encode_programs(["abc"])
    with pytest.raises(ValueError, match=r"4 bytes.*no truncation"):
        model.encode_programs(["éé"])
    with pytest.raises(ValueError, match=r"4 bytes.*no truncation"):
        model.encode_io([[("é", "é")]])
    with pytest.raises(ValueError, match="at least one pair"):
        model.encode_io([[]])
    with pytest.raises(ValueError, match="input and output"):
        model.encode_io([[("",)]])
    with pytest.raises(TypeError, match="tuple of tokens"):
        model.encode_programs([["dup"]])
    with pytest.raises(TypeError, match="strings"):
        model.encode_programs([("dup", 1)])
    with pytest.raises(TypeError, match="strings"):
        model.encode_io([[(0, "x")]])


def test_multi_positive_loss_matches_supervised_formula_and_has_gradients():
    torch.manual_seed(8)
    left, right = [torch.randn(3, 5, requires_grad=True) for _ in range(2)]
    labels = ["same", "same", "other"]
    mask = torch.tensor(
        [[True, True, False], [True, True, False], [False, False, True]]
    )
    logits = (
        functional.normalize(left, dim=1) @ functional.normalize(right, dim=1).T / 0.3
    )
    expected = (
        sum(
            -(functional.log_softmax(scores, dim=1) * positives)
            .sum(dim=1)
            .div(positives.sum(dim=1))
            .mean()
            for scores, positives in [(logits, mask), (logits.T, mask.T)]
        )
        / 2
    )
    actual = symmetric_multi_positive_contrastive_loss(
        left, right, labels, temperature=0.3
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        actual,
        symmetric_multi_positive_contrastive_loss(
            left, right, torch.tensor([1, 1, 2]), torch.eye(3, dtype=torch.bool), 0.3
        ),
    )
    actual.backward()
    assert left.grad.abs().sum() > 0
    assert right.grad.abs().sum() > 0


def test_rectangular_positive_mask_and_input_validation():
    left, right = torch.randn(2, 4), torch.randn(3, 4)
    mask = torch.tensor([[True, True, False], [False, False, True]])
    assert torch.isfinite(
        symmetric_multi_positive_contrastive_loss(left, right, positive_mask=mask)
    )
    for bad in (
        torch.zeros(2, 3, dtype=torch.bool),
        torch.ones(2, 3),
        torch.ones(2, 2, dtype=torch.bool),
    ):
        with pytest.raises(ValueError):
            symmetric_multi_positive_contrastive_loss(left, right, positive_mask=bad)
    with pytest.raises(ValueError, match="Unaligned"):
        symmetric_multi_positive_contrastive_loss(left, right)
    with pytest.raises(ValueError, match="nonempty"):
        symmetric_multi_positive_contrastive_loss(left[:0], right)
    for temperature in (0, -1, float("nan")):
        with pytest.raises(ValueError, match="temperature"):
            symmetric_multi_positive_contrastive_loss(
                left, left, temperature=temperature
            )


def test_weights_only_round_trip_and_index(model, tmp_path):
    path = tmp_path / "embedding.pt"
    metadata = {
        "commit": "test",
        "dirty": True,
        "seed": 7,
        "data_digests": {},
        "wandb_run.url": None,
    }
    model.save(path, metadata)
    payload = torch.load(path, weights_only=True)
    assert set(payload) == {"format_version", "config", "metadata", "state_dict"}
    restored = ProgramEmbedding.load(path)
    assert restored.config == model.config
    assert restored.metadata == metadata
    torch.testing.assert_close(
        model.encode_programs(["dup drop"]), restored.encode_programs(["dup drop"])
    )
    index = ProgramIndex(model)
    with pytest.raises(ValueError, match="before searching"):
        index.search_program("dup")
    with pytest.raises(ValueError, match="empty"):
        index.fit([])
    assert index.fit(["dup", "drop", "dup drop"], batch_size=2) is index
    indices, scores = index.search_program("dup drop", k=10)
    assert indices[0].item() == 2
    assert scores[0].item() == pytest.approx(1, abs=1e-6)
    assert len(indices) == 3
    assert model.training
    indices, scores = index.search_io([("[1]", "[1]")], k=2)
    assert indices.shape == scores.shape == (2,)
    assert torch.isfinite(scores).all()
    with pytest.raises(ValueError, match="positive integer"):
        index.search_program("dup", k=0)


def test_program_whitespace_and_unknown_symbol_boundaries(model):
    encoded = model.encode_programs(["dup drop", "  dup\t\n  drop  ", ("dup", "drop")])
    torch.testing.assert_close(encoded[0], encoded[1])
    torch.testing.assert_close(encoded[0], encoded[2])
    unknown = model.encode_programs(["unknown_one", "unknown_two", "ab c", "a bc"])
    assert not torch.allclose(unknown[0], unknown[1])
    assert not torch.allclose(unknown[2], unknown[3])


def test_long_program_retains_prefix_and_padding_invariance(model):
    suffix = " dup drop" * 200
    programs = ["swap" + suffix, "over" + suffix]
    encoded = model.encode_programs(programs)
    assert torch.linalg.vector_norm(encoded[0] - encoded[1]) > 1e-4
    padded = model.encode_programs(["dup", *programs, "drop"])
    torch.testing.assert_close(encoded, padded[1:3], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        encoded[:1], model.encode_programs(programs[:1]), atol=1e-6, rtol=1e-5
    )


def test_default_model_stays_compact():
    model = ProgramEmbedding()
    assert sum(parameter.numel() for parameter in model.parameters()) < 1_000_000
