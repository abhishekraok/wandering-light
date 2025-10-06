from wandering_light.common_functions import basic_fns
from wandering_light.function_def import FunctionDef, FunctionDefList, FunctionDefSet


class TestFunctionDefList:
    """Test suite for FunctionDefList class (preserves duplicates)."""

    def test_basic_functionality(self):
        """Test basic FunctionDefList creation and representation."""
        # Test with empty list
        empty_list = FunctionDefList()
        assert len(empty_list) == 0
        assert str(empty_list) == "FunctionDefList([])"

        # Test with actual functions
        first_five = FunctionDefList(list(basic_fns)[:5])
        assert len(first_five) == 5
        assert "inc" in str(first_five)
        assert "dec" in str(first_five)

        # Test with duplicates - should preserve them
        with_duplicates = FunctionDefList([basic_fns[0], basic_fns[1], basic_fns[0]])
        assert len(with_duplicates) == 3  # Preserves duplicates

    def test_list_operations(self):
        """Test list-like operations on FunctionDefList."""
        func_list = FunctionDefList()

        # Test append
        test_func = FunctionDef(
            name="test",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x",
        )
        func_list.append(test_func)
        assert len(func_list) == 1
        assert func_list[0] == test_func

        # Test append duplicate - should be preserved
        func_list.append(test_func)
        assert len(func_list) == 2  # Duplicates preserved

        # Test extend
        more_funcs = [
            FunctionDef(
                name="func1",
                input_type="builtins.int",
                output_type="builtins.int",
                code="return x",
            ),
            FunctionDef(
                name="func2",
                input_type="builtins.int",
                output_type="builtins.int",
                code="return x",
            ),
        ]
        func_list.extend(more_funcs)
        assert len(func_list) == 4

        # Test iteration
        names = [func.name for func in func_list]
        assert names == ["test", "test", "func1", "func2"]

    def test_to_string_method(self):
        """Test the to_string method that converts FunctionDefList to comma-separated string."""
        # Test with empty list
        empty_list = FunctionDefList()
        assert empty_list.to_string() == ""

        # Test with single function
        single_func = FunctionDefList(list(basic_fns)[:1])
        assert single_func.to_string() == "inc"

        # Test with multiple functions
        multi_func = FunctionDefList(list(basic_fns)[:4])
        expected = "inc, dec, double, half"
        assert multi_func.to_string() == expected

        # Test with duplicates
        with_duplicates = FunctionDefList([basic_fns[0], basic_fns[1], basic_fns[0]])
        expected_with_dups = "inc, dec, inc"
        assert with_duplicates.to_string() == expected_with_dups

    def test_compact_representation(self):
        """Test that the representation is compact and readable."""
        # Test with various function counts
        single = FunctionDefList(list(basic_fns)[:1])
        assert str(single) == "FunctionDefList([inc])"

        triple = FunctionDefList(list(basic_fns)[:3])
        assert str(triple) == "FunctionDefList([inc, dec, double])"

        # Test with duplicates
        with_duplicates = FunctionDefList([basic_fns[0], basic_fns[1], basic_fns[0]])
        assert str(with_duplicates) == "FunctionDefList([inc, dec, inc])"


class TestFunctionDefSet:
    """Test suite for FunctionDefSet class (deduplicates)."""

    def test_basic_functionality(self):
        """Test basic FunctionDefSet creation and representation."""
        # Test with empty set
        empty_set = FunctionDefSet()
        assert len(empty_set) == 0
        assert str(empty_set) == "FunctionDefSet([])"

        # Test with actual functions
        first_five = FunctionDefSet(list(basic_fns)[:5])
        assert len(first_five) == 5
        assert "inc" in str(first_five)
        assert "dec" in str(first_five)

        # Test with duplicates - should deduplicate them
        with_duplicates = FunctionDefSet([basic_fns[0], basic_fns[1], basic_fns[0]])
        assert len(with_duplicates) == 2  # Deduplicates

    def test_set_operations(self):
        """Test set-like operations on FunctionDefSet."""
        func_set = FunctionDefSet()

        # Test append
        test_func = FunctionDef(
            name="test",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x",
        )
        func_set.append(test_func)
        assert len(func_set) == 1
        assert func_set[0] == test_func

        # Test append duplicate - should be ignored
        func_set.append(test_func)
        assert len(func_set) == 1  # Duplicate ignored

        # Test extend with duplicates
        more_funcs = [
            FunctionDef(
                name="func1",
                input_type="builtins.int",
                output_type="builtins.int",
                code="return x",
            ),
            test_func,  # Duplicate
            FunctionDef(
                name="func2",
                input_type="builtins.int",
                output_type="builtins.int",
                code="return x",
            ),
        ]
        func_set.extend(more_funcs)
        assert len(func_set) == 3  # Only unique functions

        # Test iteration
        names = [func.name for func in func_set]
        assert names == ["test", "func1", "func2"]

    def test_parse_string_functionality(self):
        """Test parse_string method that returns FunctionDefList."""
        # Create a set of available functions
        available_functions = FunctionDefSet(list(basic_fns)[:5])

        # Test parsing empty string
        result = available_functions.parse_string("")
        assert len(result) == 0
        assert isinstance(result, FunctionDefList)

        # Test parsing single function
        result = available_functions.parse_string("inc")
        assert len(result) == 1
        assert result[0].name == "inc"
        assert isinstance(result, FunctionDefList)

        # Test parsing multiple functions
        result = available_functions.parse_string("inc, dec, double, half")
        assert len(result) == 4
        expected_names = ["inc", "dec", "double", "half"]
        actual_names = [func.name for func in result]
        assert actual_names == expected_names

        # Test parsing with duplicates - should preserve them
        result = available_functions.parse_string("inc, dec, inc, double, inc")
        assert len(result) == 5  # Preserves duplicates
        expected_names = ["inc", "dec", "inc", "double", "inc"]
        actual_names = [func.name for func in result]
        assert actual_names == expected_names

        # Test parsing with spaces
        result = available_functions.parse_string("inc, dec,   double,half  ")
        assert len(result) == 4
        expected_names = ["inc", "dec", "double", "half"]
        actual_names = [func.name for func in result]
        assert actual_names == expected_names

        # Test unknown function - should return empty list
        result = available_functions.parse_string("inc,unknown_func,dec")
        assert len(result) == 0
        assert isinstance(result, FunctionDefList)

    def test_to_string_method(self):
        """Test the to_string method that converts FunctionDefSet to comma-separated string."""
        # Test with empty set
        empty_set = FunctionDefSet()
        assert empty_set.to_string() == ""

        # Test with single function
        single_func = FunctionDefSet(list(basic_fns)[:1])
        assert single_func.to_string() == "inc"

        # Test with multiple functions
        multi_func = FunctionDefSet(list(basic_fns)[:4])
        expected = "inc, dec, double, half"
        assert multi_func.to_string() == expected

    def test_round_trip_with_duplicates(self):
        """Test that to_string -> parse_string -> to_string preserves duplicates."""
        available_functions = FunctionDefSet(list(basic_fns)[:5])

        # Create a string with duplicates
        original_string = "inc, dec, inc, double, inc"

        # Parse it (should return FunctionDefList with duplicates)
        parsed_list = available_functions.parse_string(original_string)

        # Convert back to string
        result_string = parsed_list.to_string()

        # Should preserve duplicates
        assert result_string == original_string

    def test_compact_representation(self):
        """Test that the representation is compact and readable."""
        # Test with various function counts
        single = FunctionDefSet(list(basic_fns)[:1])
        assert str(single) == "FunctionDefSet([inc])"

        triple = FunctionDefSet(list(basic_fns)[:3])
        assert str(triple) == "FunctionDefSet([inc, dec, double])"

        # Test that long function names don't break formatting
        long_name_func = FunctionDef(
            name="very_long_function_name_that_should_still_work",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x",
        )
        long_set = FunctionDefSet([long_name_func])
        expected = "FunctionDefSet([very_long_function_name_that_should_still_work])"
        assert str(long_set) == expected
