import json

import numpy as np

from experiments.pbe_scale.diversity_semantic import (
    THRESHOLDS,
    encode_corpus,
    exact_neighbors,
)


def test_exact_neighbors_match_direct_equality_and_exclude_self():
    rng = np.random.default_rng(123)
    matrix = rng.integers(0, 4, (29, 21)).astype(np.float32)
    matrix[1] = matrix[0]
    matrix[2] = matrix[0]
    matrix[2, 0] = 8
    nearest, mismatches, edges = exact_neighbors(matrix, block_size=7)
    direct = (matrix[:, None, :] != matrix[None, :, :]).sum(axis=-1)
    np.fill_diagonal(direct, matrix.shape[1] + 1)
    assert (nearest != np.arange(len(matrix))).all()
    np.testing.assert_array_equal(mismatches, direct.min(axis=1))
    np.testing.assert_array_equal(nearest, direct.argmin(axis=1))
    for threshold in THRESHOLDS:
        expected = np.triu(1 - direct / matrix.shape[1] >= threshold, k=1).sum()
        assert edges[threshold] == expected


def test_saved_probe_alignment_ignores_display_order_and_exclusions(tmp_path):
    def row(i, inputs, outputs, hidden, hidden_outputs):
        return {
            "candidate_digest": str(i),
            "shared_count": 2,
            "candidate": {
                "id": str(i),
                "input_kind": "list",
                "family": "test",
                "description": "test",
                "program": "dup",
            },
            "task": {
                "inputs": inputs,
                "outputs": outputs,
                "held_out_inputs": hidden,
                "held_out_outputs": hidden_outputs,
            },
        }

    rows = [
        row(0, ["shared-b"], ["0"], ["shared-a", "fresh-0"], ["1", "9"]),
        row(1, ["fresh-1", "shared-a"], ["8", "1"], ["shared-b"], ["2"]),
        row(2, ["excluded"], ["9"], [], []),
    ]
    path = tmp_path / "accepted.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    matrices, metadata, probes, _ = encode_corpus(path, {"2"})
    assert probes["list"] == ["shared-a", "shared-b"]
    assert [r["id"] for r in metadata["list"]] == ["0", "1"]
    assert (matrices["list"][0] == matrices["list"][1]).tolist() == [True, False]


def test_list_length_parser_uses_longest_saved_list():
    from experiments.pbe_scale.diversity_semantic import max_list_length

    assert max_list_length("[ nil ]") == 0
    assert max_list_length("[ (0 -1 2 3) -10 ]") == 4
    assert max_list_length("[ (0) (1 2 3 4 5) ]") == 5
    assert max_list_length("[ -10 10 ]") == 0
