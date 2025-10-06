from wandering_light.function_def import FunctionDef


def test_resolve_builtin_type():
    fn = FunctionDef(
        name="foo",
        input_type="builtins.int",
        output_type="builtins.int",
        code="def foo(x): return x + 1",
    )
    assert fn.input_type_cls() is int
    assert fn.output_type_cls() is int
