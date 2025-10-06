"""
Test suite for the split between FunctionDefSet and FunctionDefList.
Verifies that FunctionDefSet deduplicates while FunctionDefList preserves duplicates.
"""

from wandering_light.common_functions import basic_fns
from wandering_light.function_def import FunctionDef, FunctionDefList, FunctionDefSet


class TestFunctionDefSplit:
    """Test suite for FunctionDefSet vs FunctionDefList functionality."""

    def test_function_def_set_deduplicates(self):
        """Test that FunctionDefSet properly deduplicates functions."""
        # Create some test functions
        func1 = FunctionDef(
            name="test_func",
            input_type="builtins.int",
            output_type="builtins.str",
            code="return str(x)",
        )
        func2 = FunctionDef(
            name="another_func",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x + 1",
        )

        # Test deduplication
        func_set = FunctionDefSet()
        func_set.append(func1)
        func_set.append(func2)
        func_set.append(func1)  # Duplicate - should be ignored

        assert len(func_set) == 2
        assert func_set[0].name == "test_func"
        assert func_set[1].name == "another_func"

    def test_function_def_list_preserves_duplicates(self):
        """Test that FunctionDefList preserves duplicate functions."""
        # Create some test functions
        func1 = FunctionDef(
            name="test_func",
            input_type="builtins.int",
            output_type="builtins.str",
            code="return str(x)",
        )
        func2 = FunctionDef(
            name="another_func",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x + 1",
        )

        # Test that duplicates are preserved
        func_list = FunctionDefList()
        func_list.append(func1)
        func_list.append(func2)
        func_list.append(func1)  # Duplicate - should be preserved

        assert len(func_list) == 3
        assert func_list[0].name == "test_func"
        assert func_list[1].name == "another_func"
        assert func_list[2].name == "test_func"  # Duplicate preserved

    def test_parse_string_with_duplicates(self):
        """Test that parse_string correctly handles duplicate function names."""
        # Create a set of available functions
        available_functions = FunctionDefSet(list(basic_fns)[:5])  # First 5 functions

        # Test parsing a string with duplicate function names
        test_string = "inc,dec,inc,double,inc"  # 'inc' appears 3 times

        # Parse using FunctionDefSet.parse_string (returns FunctionDefList)
        result = available_functions.parse_string(test_string)

        # Verify that duplicates are preserved in the result
        assert len(result) == 5
        assert result[0].name == "inc"
        assert result[1].name == "dec"
        assert result[2].name == "inc"  # First duplicate
        assert result[3].name == "double"
        assert result[4].name == "inc"  # Second duplicate

        # Verify the result is a FunctionDefList (not FunctionDefSet)
        assert isinstance(result, FunctionDefList)

    def test_parse_string_unknown_function(self):
        """Test that parse_string returns empty list for unknown functions."""
        available_functions = FunctionDefSet(list(basic_fns)[:3])

        # Test with unknown function
        result = available_functions.parse_string("inc,unknown_func,dec")
        assert len(result) == 0
        assert isinstance(result, FunctionDefList)

    def test_to_string_methods(self):
        """Test that to_string methods work correctly for both classes."""
        # Test FunctionDefSet
        func_set = FunctionDefSet(list(basic_fns)[:3])
        set_string = func_set.to_string()
        expected_names = [f.name for f in list(basic_fns)[:3]]
        assert set_string == ", ".join(expected_names)

        # Test FunctionDefList with duplicates
        func_list = FunctionDefList([basic_fns[0], basic_fns[1], basic_fns[0]])
        list_string = func_list.to_string()
        expected_with_dups = (
            f"{basic_fns[0].name}, {basic_fns[1].name}, {basic_fns[0].name}"
        )
        assert list_string == expected_with_dups

    def test_round_trip_with_duplicates(self):
        """Test that to_string -> parse_string -> to_string preserves duplicates."""
        available_functions = FunctionDefSet(list(basic_fns)[:5])

        # Create a string with duplicates
        original_string = "inc, dec, inc, double, inc"

        # Parse it
        parsed_list = available_functions.parse_string(original_string)

        # Convert back to string
        result_string = parsed_list.to_string()

        # Should preserve duplicates
        assert result_string == original_string

    def test_function_def_set_representation(self):
        """Test that FunctionDefSet has correct string representation."""
        func_set = FunctionDefSet(list(basic_fns)[:2])
        repr_str = repr(func_set)
        assert "FunctionDefSet" in repr_str
        assert "inc" in repr_str
        assert "dec" in repr_str

    def test_function_def_list_representation(self):
        """Test that FunctionDefList has correct string representation."""
        func_list = FunctionDefList([basic_fns[0], basic_fns[0]])  # Duplicates
        repr_str = repr(func_list)
        assert "FunctionDefList" in repr_str
        assert repr_str.count("inc") == 2  # Should show both occurrences

    def test_empty_collections(self):
        """Test behavior with empty collections."""
        # Empty FunctionDefSet
        empty_set = FunctionDefSet()
        assert len(empty_set) == 0
        assert empty_set.to_string() == ""
        assert repr(empty_set) == "FunctionDefSet([])"

        # Empty FunctionDefList
        empty_list = FunctionDefList()
        assert len(empty_list) == 0
        assert empty_list.to_string() == ""
        assert repr(empty_list) == "FunctionDefList([])"

        # Parse empty string
        available_functions = FunctionDefSet(list(basic_fns)[:3])
        result = available_functions.parse_string("")
        assert len(result) == 0
        assert isinstance(result, FunctionDefList)

    def test_extend_methods(self):
        """Test extend methods for both classes."""
        # Test FunctionDefSet extend (should deduplicate)
        func_set = FunctionDefSet()
        func_set.extend([basic_fns[0], basic_fns[1], basic_fns[0]])  # Duplicate
        assert len(func_set) == 2  # Deduplicated

        # Test FunctionDefList extend (should preserve duplicates)
        func_list = FunctionDefList()
        func_list.extend([basic_fns[0], basic_fns[1], basic_fns[0]])  # Duplicate
        assert len(func_list) == 3  # Preserves duplicates

    def test_critical_bug_fix(self):
        """
        Test the specific bug that was causing 90% success rate instead of 100%.
        This is the core issue: repeated function names in ground truth sequences.
        """
        # Create available functions - need all functions to include the test cases
        available_functions = FunctionDefSet(list(basic_fns))

        # Test cases that were failing before the fix
        test_cases = [
            "f_abs_sqrt,f_abs_sqrt,float_to_str",  # Spec 40 from the failures
            "set_is_empty,bool_not,bool_not",  # Spec 41 from the failures
            "range_max,int_popcount,int_popcount",  # Spec 51 from the failures
            "f_abs_sqrt,f_square,f_square",  # Spec 57 from the failures
            "bool_not,bool_not,bool_to_int",  # Variation with multiple duplicates
        ]

        for test_case in test_cases:
            # This should now work correctly (preserving duplicates)
            result = available_functions.parse_string(test_case)

            # Count expected vs actual functions
            expected_functions = test_case.split(",")
            expected_count = len(expected_functions)
            actual_count = len(result)

            # The key assertion: duplicates should be preserved
            assert actual_count == expected_count, (
                f"Failed for '{test_case}': expected {expected_count} functions, got {actual_count}"
            )

            # Verify the actual function names match
            actual_names = [f.name for f in result]
            expected_names = [name.strip() for name in expected_functions]
            assert actual_names == expected_names, (
                f"Function names don't match for '{test_case}': expected {expected_names}, got {actual_names}"
            )


if __name__ == "__main__":
    test = TestFunctionDefSplit()
    test.test_critical_bug_fix()
    print("✅ Critical bug fix test passed!")
