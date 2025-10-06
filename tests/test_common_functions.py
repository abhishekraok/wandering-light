import pytest

from wandering_light.common_functions import EXPECTED_OUTPUTS, SAMPLE_INPUTS, basic_fns
from wandering_light.executor import Executor
from wandering_light.typed_list import TypedList


@pytest.fixture(scope="module")
def executor():
    return Executor(basic_fns)


def _make_test(fn):
    def test_fn(executor):
        input_list = SAMPLE_INPUTS.get(fn.input_type)
        assert input_list is not None, f"No samples for type {fn.input_type}"
        expected_output_list = EXPECTED_OUTPUTS[fn.name]
        tl_in = TypedList(input_list, item_type=fn.input_type_cls())
        tl_out = executor.execute(fn, tl_in)
        assert tl_out.item_type == fn.output_type_cls()
        assert tl_out.items == expected_output_list, (
            f"Function {fn.name} failed: got {tl_out.items}, "
            f"expected {expected_output_list}"
        )

    test_fn.__name__ = f"test_{fn.name}"
    return test_fn


# Dynamically create a test for each function
for fn in basic_fns:
    globals()[f"test_{fn.name}"] = _make_test(fn)


def test_list_min_non_numeric(executor):
    fn = {f.name: f for f in basic_fns}
    tl_in = TypedList([{}, {"a": 1}], dict)
    tl_items = executor.execute(fn["dict_items"], tl_in)
    tl_out = executor.execute(fn["list_min"], tl_items)
    assert tl_out.items == [0, 0]


def test_list_max_non_numeric(executor):
    fn = {f.name: f for f in basic_fns}
    tl_in = TypedList([{}, {"a": 1}], dict)
    tl_items = executor.execute(fn["dict_items"], tl_in)
    tl_out = executor.execute(fn["list_max"], tl_items)
    assert tl_out.items == [0, 0]


def test_list_sum_non_numeric(executor):
    fn = {f.name: f for f in basic_fns}
    tl_in = TypedList(["ab", "cd"], str)
    tl_chars = executor.execute(fn["str_to_list"], tl_in)
    tl_out = executor.execute(fn["list_sum"], tl_chars)
    assert tl_out.items == [0, 0]
