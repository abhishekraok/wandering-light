"""The depth probe rests on how it peels a witness. Pin that.

The scorer and every predictor are imported unchanged from `policy_probe`,
which `tests/test_policy_probe.py` already pins. What is new here is the
construction of the sub-tasks, and it is where the comparison can silently go
wrong -- peeling the wrong end, or masking a peeled task on the field that
applies to the other one.
"""

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from policy_probe_depth import FIRST, LAST, peel, type_name


def _corpus(tmp_path, witnesses):
    """A corpus shaped like `deep-corpus-v1`, run through the real basis."""
    from value_probe import basis_pieces

    from wandering_light.typed_list import TypedList

    _, by_name, executor = basis_pieces()
    with gzip.open(tmp_path / "discovery.jsonl.gz", "wt") as handle:
        for names in witnesses:
            state = TypedList([2, 3], item_type=int)
            source = state.to_string()
            for name in names:
                state = executor.execute(by_name[name], state)
            handle.write(
                json.dumps(
                    {
                        "input": source,
                        "output": state.to_string(),
                        "witness_function_names": names,
                        "metadata": {
                            "certified_distance": len(names),
                            "input_type": "builtins.int",
                            "output_type": "builtins.int",
                        },
                    }
                )
                + "\n"
            )
    return tmp_path


def _peel(tmp_path, witnesses, depth, min_distance=1):
    from value_probe import basis_pieces

    _, by_name, executor = basis_pieces()
    corpus = _corpus(tmp_path, witnesses)
    return peel(corpus, "discovery", by_name, executor, depth, min_distance)


def test_depth_zero_reproduces_the_original_task(tmp_path):
    """`k = 0` has to be the published probe, or the comparison has no anchor."""
    front, back = _peel(tmp_path, [["inc", "double", "neg"]], depth=0)
    original_input = front[0]["input"]

    assert front[0]["input"] == back[0]["input"] == original_input
    assert front[0]["output"] == back[0]["output"]
    assert front[0]["metadata"][FIRST] == ["inc"]
    assert back[0]["metadata"][LAST] == ["neg"]
    assert front[0]["metadata"]["certified_distance"] == 3
    assert back[0]["metadata"]["certified_distance"] == 3


def test_the_two_ends_peel_opposite_ends(tmp_path):
    """Front peels the input, back peels the target. Peeling the input for
    both would make the back probe predict from a shortened program while
    still masking on the original output type."""
    from wandering_light.typed_list import TypedList

    front, back = _peel(tmp_path, [["inc", "double", "neg"]], depth=1)

    # front moved its input forward one step and kept the target
    assert TypedList.from_str(front[0]["input"]) == TypedList([3, 4], item_type=int)
    assert TypedList.from_str(front[0]["output"]) == TypedList([-6, -8], item_type=int)
    assert front[0]["metadata"][FIRST] == ["double"]

    # back kept its input and pulled the target back one step
    assert TypedList.from_str(back[0]["input"]) == TypedList([2, 3], item_type=int)
    assert TypedList.from_str(back[0]["output"]) == TypedList([6, 8], item_type=int)
    assert back[0]["metadata"][LAST] == ["double"]

    assert front[0]["metadata"]["certified_distance"] == 2
    assert back[0]["metadata"]["certified_distance"] == 2


def test_each_end_declares_the_type_its_own_mask_reads(tmp_path):
    """A first action must consume the sub-task's input type and a last action
    must produce its output type, so peeling has to move the corresponding
    field. Leaving the original type in place would mask against a state the
    sub-task no longer starts or ends on."""
    # int -> str -> str -> int, so both peeled ends land on a str
    front, back = _peel(tmp_path, [["int_to_str", "upper", "length"]], depth=1)

    # the front sub-task now starts on `s_1`, a str, not on the original int
    assert front[0]["metadata"]["input_type"] == "builtins.str"
    assert front[0]["metadata"]["output_type"] == "builtins.int"
    # the back sub-task now ends on `s_2`, a str, not on the original int
    assert back[0]["metadata"]["input_type"] == "builtins.int"
    assert back[0]["metadata"]["output_type"] == "builtins.str"
    assert front[0]["metadata"][FIRST] == back[0]["metadata"][LAST] == ["upper"]


def test_type_name_matches_the_corpus_and_basis_spelling(tmp_path):
    from wandering_light.typed_list import TypedList

    assert type_name(TypedList([1], item_type=int)) == "builtins.int"
    assert type_name(TypedList([[1]], item_type=list)) == "builtins.list"


def test_min_distance_holds_the_task_population_fixed(tmp_path):
    """Short tasks cannot be peeled, so they must be dropped at every depth --
    including `k = 0`, where they still could be. Otherwise the k=0 row is
    scored on an easier population than k=1 and the depth comparison is
    confounded with task length."""
    witnesses = [["inc"], ["inc", "double"], ["inc", "double", "neg"]]
    front_all, _ = _peel(tmp_path, witnesses, depth=0, min_distance=1)
    front_deep, _ = _peel(tmp_path, witnesses, depth=0, min_distance=2)
    assert len(front_all) == 3
    assert len(front_deep) == 2
    assert all(t["metadata"]["certified_distance"] >= 2 for t in front_deep)


def test_labels_come_from_the_witness_at_the_right_offset(tmp_path):
    """Off-by-one here would score every predictor against the wrong op."""
    witness = ["inc", "double", "neg", "square"]
    front0, back0 = _peel(tmp_path, [witness], depth=0)
    front1, back1 = _peel(tmp_path, [witness], depth=1)
    assert front0[0]["metadata"][FIRST] == ["inc"]
    assert front1[0]["metadata"][FIRST] == ["double"]
    assert back0[0]["metadata"][LAST] == ["square"]
    assert back1[0]["metadata"][LAST] == ["neg"]
