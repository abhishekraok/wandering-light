import pytest

from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDef, FunctionDefList
from wandering_light.typed_list import TypedList


@pytest.fixture
def setup_exec():
    available_functions = FunctionDefList([])
    ex = Executor(available_functions)
    return available_functions, ex


def test_increment_function_def(setup_exec):
    available_functions, ex = setup_exec
    code = "return x + 1"
    fn = FunctionDef(
        name="increment",
        input_type="builtins.int",
        output_type="builtins.int",
        code=code,
    )
    available_functions.append(fn)
    inp = TypedList([1, 2, 3])
    out = ex.execute(fn, inp)
    assert out == TypedList([2, 3, 4])
    # usage_count incremented
    assert fn.usage_count == 1
