"""
Generated trajectory specs data
Total specs: 500
"""

from function_def import FunctionDef
from trajectory import TrajectorySpec, TrajectorySpecList
from typed_list import TypedList

# Function definitions
list_median_func = FunctionDef(
    name="list_median",
    input_type="builtins.list",
    output_type="builtins.float",
    code="""import statistics; return float(statistics.median(x)) if x else 0.0""",
    usage_count=0,
    metadata={},
)

tuple_length_func = FunctionDef(
    name="tuple_length",
    input_type="builtins.tuple",
    output_type="builtins.int",
    code="""return len(x)""",
    usage_count=0,
    metadata={},
)

inc_func = FunctionDef(
    name="inc",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return x + 1""",
    usage_count=0,
    metadata={},
)

is_negative_func = FunctionDef(
    name="is_negative",
    input_type="builtins.int",
    output_type="builtins.bool",
    code="""return x < 0""",
    usage_count=0,
    metadata={},
)

range_length_func = FunctionDef(
    name="range_length",
    input_type="builtins.range",
    output_type="builtins.int",
    code="""return len(x)""",
    usage_count=0,
    metadata={},
)

bool_to_str_func = FunctionDef(
    name="bool_to_str",
    input_type="builtins.bool",
    output_type="builtins.str",
    code="""return str(x)""",
    usage_count=0,
    metadata={},
)

bytes_reverse_func = FunctionDef(
    name="bytes_reverse",
    input_type="builtins.bytes",
    output_type="builtins.bytes",
    code="""return x[::-1]""",
    usage_count=0,
    metadata={},
)

f_sin_func = FunctionDef(
    name="f_sin",
    input_type="builtins.float",
    output_type="builtins.float",
    code="""import math; return math.sin(x)""",
    usage_count=0,
    metadata={},
)

tuple_reverse_func = FunctionDef(
    name="tuple_reverse",
    input_type="builtins.tuple",
    output_type="builtins.tuple",
    code="""return x[::-1]""",
    usage_count=0,
    metadata={},
)

set_to_list_func = FunctionDef(
    name="set_to_list",
    input_type="builtins.set",
    output_type="builtins.list",
    code="""return list(x)""",
    usage_count=0,
    metadata={},
)

tuple_to_index_dict_func = FunctionDef(
    name="tuple_to_index_dict",
    input_type="builtins.tuple",
    output_type="builtins.dict",
    code="""return {i:v for i,v in enumerate(x)}""",
    usage_count=0,
    metadata={},
)

complex_abs_func = FunctionDef(
    name="complex_abs",
    input_type="builtins.complex",
    output_type="builtins.float",
    code="""return abs(x)""",
    usage_count=0,
    metadata={},
)

bytearray_reverse_func = FunctionDef(
    name="bytearray_reverse",
    input_type="builtins.bytearray",
    output_type="builtins.bytearray",
    code="""return bytearray(x[::-1])""",
    usage_count=0,
    metadata={},
)

str_hash_func = FunctionDef(
    name="str_hash",
    input_type="builtins.str",
    output_type="builtins.int",
    code="""return hash(x)""",
    usage_count=0,
    metadata={},
)

range_list_func = FunctionDef(
    name="range_list",
    input_type="builtins.range",
    output_type="builtins.list",
    code="""return list(x)""",
    usage_count=0,
    metadata={},
)

complex_conjugate_func = FunctionDef(
    name="complex_conjugate",
    input_type="builtins.complex",
    output_type="builtins.complex",
    code="""return x.conjugate()""",
    usage_count=0,
    metadata={},
)

dict_items_func = FunctionDef(
    name="dict_items",
    input_type="builtins.dict",
    output_type="builtins.list",
    code="""return list(x.items())""",
    usage_count=0,
    metadata={},
)

f_round_func = FunctionDef(
    name="f_round",
    input_type="builtins.float",
    output_type="builtins.int",
    code="""return round(x)""",
    usage_count=0,
    metadata={},
)

half_func = FunctionDef(
    name="half",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return x // 2""",
    usage_count=0,
    metadata={},
)

range_max_func = FunctionDef(
    name="range_max",
    input_type="builtins.range",
    output_type="builtins.int",
    code="""return x[-1] if x else 0""",
    usage_count=0,
    metadata={},
)

mod2_func = FunctionDef(
    name="mod2",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return x % 2""",
    usage_count=0,
    metadata={},
)

upper_func = FunctionDef(
    name="upper",
    input_type="builtins.str",
    output_type="builtins.str",
    code="""return x.upper()""",
    usage_count=0,
    metadata={},
)

identity_int_func = FunctionDef(
    name="identity_int",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return x""",
    usage_count=0,
    metadata={},
)

range_sum_func = FunctionDef(
    name="range_sum",
    input_type="builtins.range",
    output_type="builtins.int",
    code="""return sum(x)""",
    usage_count=0,
    metadata={},
)

int_is_power_of_two_func = FunctionDef(
    name="int_is_power_of_two",
    input_type="builtins.int",
    output_type="builtins.bool",
    code="""return x > 0 and (x & (x - 1)) == 0""",
    usage_count=0,
    metadata={},
)

list_max_func = FunctionDef(
    name="list_max",
    input_type="builtins.list",
    output_type="builtins.int",
    code="""return max(x) if x else 0""",
    usage_count=0,
    metadata={},
)

startswith_a_func = FunctionDef(
    name="startswith_a",
    input_type="builtins.str",
    output_type="builtins.bool",
    code="""return x.startswith('a')""",
    usage_count=0,
    metadata={},
)

bool_to_float_func = FunctionDef(
    name="bool_to_float",
    input_type="builtins.bool",
    output_type="builtins.float",
    code="""return 1.0 if x else 0.0""",
    usage_count=0,
    metadata={},
)

neg_func = FunctionDef(
    name="neg",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return -x""",
    usage_count=0,
    metadata={},
)

dict_has_duplicate_values_func = FunctionDef(
    name="dict_has_duplicate_values",
    input_type="builtins.dict",
    output_type="builtins.bool",
    code="""vals=list(x.values()); return len(vals)!=len(set(vals))""",
    usage_count=0,
    metadata={},
)

f_abs_sqrt_func = FunctionDef(
    name="f_abs_sqrt",
    input_type="builtins.float",
    output_type="builtins.float",
    code="""return abs(x) ** 0.5""",
    usage_count=0,
    metadata={},
)

is_positive_func = FunctionDef(
    name="is_positive",
    input_type="builtins.int",
    output_type="builtins.bool",
    code="""return x > 0""",
    usage_count=0,
    metadata={},
)

repeat_func = FunctionDef(
    name="repeat",
    input_type="builtins.str",
    output_type="builtins.str",
    code="""return x * 2""",
    usage_count=0,
    metadata={},
)

complex_phase_func = FunctionDef(
    name="complex_phase",
    input_type="builtins.complex",
    output_type="builtins.float",
    code="""import math; return math.atan2(x.imag, x.real)""",
    usage_count=0,
    metadata={},
)

bytearray_to_bytes_func = FunctionDef(
    name="bytearray_to_bytes",
    input_type="builtins.bytearray",
    output_type="builtins.bytes",
    code="""return bytes(x)""",
    usage_count=0,
    metadata={},
)

bytes_upper_func = FunctionDef(
    name="bytes_upper",
    input_type="builtins.bytes",
    output_type="builtins.bytes",
    code="""return x.upper()""",
    usage_count=0,
    metadata={},
)

f_square_func = FunctionDef(
    name="f_square",
    input_type="builtins.float",
    output_type="builtins.float",
    code="""return x * x""",
    usage_count=0,
    metadata={},
)

tuple_to_list_func = FunctionDef(
    name="tuple_to_list",
    input_type="builtins.tuple",
    output_type="builtins.list",
    code="""return list(x)""",
    usage_count=0,
    metadata={},
)

str_is_palindrome_func = FunctionDef(
    name="str_is_palindrome",
    input_type="builtins.str",
    output_type="builtins.bool",
    code="""s=x.lower(); return s==s[::-1]""",
    usage_count=0,
    metadata={},
)

abs_func = FunctionDef(
    name="abs",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return abs(x)""",
    usage_count=0,
    metadata={},
)

bool_not_func = FunctionDef(
    name="bool_not",
    input_type="builtins.bool",
    output_type="builtins.bool",
    code="""return not x""",
    usage_count=0,
    metadata={},
)

dict_keys_func = FunctionDef(
    name="dict_keys",
    input_type="builtins.dict",
    output_type="builtins.list",
    code="""return list(x.keys())""",
    usage_count=0,
    metadata={},
)

is_title_func = FunctionDef(
    name="is_title",
    input_type="builtins.str",
    output_type="builtins.bool",
    code="""return x.istitle()""",
    usage_count=0,
    metadata={},
)

set_is_empty_func = FunctionDef(
    name="set_is_empty",
    input_type="builtins.set",
    output_type="builtins.bool",
    code="""return len(x) == 0""",
    usage_count=0,
    metadata={},
)

list_reverse_func = FunctionDef(
    name="list_reverse",
    input_type="builtins.list",
    output_type="builtins.list",
    code="""return x[::-1]""",
    usage_count=0,
    metadata={},
)

tuple_is_empty_func = FunctionDef(
    name="tuple_is_empty",
    input_type="builtins.tuple",
    output_type="builtins.bool",
    code="""return len(x) == 0""",
    usage_count=0,
    metadata={},
)

dict_freeze_func = FunctionDef(
    name="dict_freeze",
    input_type="builtins.dict",
    output_type="builtins.tuple",
    code="""return tuple(sorted(x.items()))""",
    usage_count=0,
    metadata={},
)

dict_length_func = FunctionDef(
    name="dict_length",
    input_type="builtins.dict",
    output_type="builtins.int",
    code="""return len(x)""",
    usage_count=0,
    metadata={},
)

dict_keyset_func = FunctionDef(
    name="dict_keyset",
    input_type="builtins.dict",
    output_type="builtins.set",
    code="""return set(x.keys())""",
    usage_count=0,
    metadata={},
)

str_count_vowels_func = FunctionDef(
    name="str_count_vowels",
    input_type="builtins.str",
    output_type="builtins.int",
    code="""return sum(1 for c in x.lower() if c in 'aeiou')""",
    usage_count=0,
    metadata={},
)

bool_identity_func = FunctionDef(
    name="bool_identity",
    input_type="builtins.bool",
    output_type="builtins.bool",
    code="""return x""",
    usage_count=0,
    metadata={},
)

f_abs_func = FunctionDef(
    name="f_abs",
    input_type="builtins.float",
    output_type="builtins.float",
    code="""return abs(x)""",
    usage_count=0,
    metadata={},
)

f_fraction_func = FunctionDef(
    name="f_fraction",
    input_type="builtins.float",
    output_type="builtins.float",
    code="""return x - int(x)""",
    usage_count=0,
    metadata={},
)

first_char_func = FunctionDef(
    name="first_char",
    input_type="builtins.str",
    output_type="builtins.str",
    code="""return x[0] if x else ''""",
    usage_count=0,
    metadata={},
)

length_func = FunctionDef(
    name="length",
    input_type="builtins.str",
    output_type="builtins.int",
    code="""return len(x)""",
    usage_count=0,
    metadata={},
)

dict_flip_func = FunctionDef(
    name="dict_flip",
    input_type="builtins.dict",
    output_type="builtins.dict",
    code="""return {v:k for k,v in x.items()}""",
    usage_count=0,
    metadata={},
)

dict_values_func = FunctionDef(
    name="dict_values",
    input_type="builtins.dict",
    output_type="builtins.list",
    code="""return list(x.values())""",
    usage_count=0,
    metadata={},
)

double_func = FunctionDef(
    name="double",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return x * 2""",
    usage_count=0,
    metadata={},
)

list_min_func = FunctionDef(
    name="list_min",
    input_type="builtins.list",
    output_type="builtins.int",
    code="""return min(x) if x else 0""",
    usage_count=0,
    metadata={},
)

complex_imag_func = FunctionDef(
    name="complex_imag",
    input_type="builtins.complex",
    output_type="builtins.float",
    code="""return x.imag""",
    usage_count=0,
    metadata={},
)

complex_real_func = FunctionDef(
    name="complex_real",
    input_type="builtins.complex",
    output_type="builtins.float",
    code="""return x.real""",
    usage_count=0,
    metadata={},
)

bytearray_length_func = FunctionDef(
    name="bytearray_length",
    input_type="builtins.bytearray",
    output_type="builtins.int",
    code="""return len(x)""",
    usage_count=0,
    metadata={},
)

strip_func = FunctionDef(
    name="strip",
    input_type="builtins.str",
    output_type="builtins.str",
    code="""return x.strip()""",
    usage_count=0,
    metadata={},
)

str_remove_digits_func = FunctionDef(
    name="str_remove_digits",
    input_type="builtins.str",
    output_type="builtins.str",
    code="""return ''.join(c for c in x if not c.isdigit())""",
    usage_count=0,
    metadata={},
)

is_alpha_func = FunctionDef(
    name="is_alpha",
    input_type="builtins.str",
    output_type="builtins.bool",
    code="""return x.isalpha()""",
    usage_count=0,
    metadata={},
)

is_even_func = FunctionDef(
    name="is_even",
    input_type="builtins.int",
    output_type="builtins.bool",
    code="""return x % 2 == 0""",
    usage_count=0,
    metadata={},
)

int_popcount_func = FunctionDef(
    name="int_popcount",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return x.bit_count()""",
    usage_count=0,
    metadata={},
)

list_length_func = FunctionDef(
    name="list_length",
    input_type="builtins.list",
    output_type="builtins.int",
    code="""return len(x)""",
    usage_count=0,
    metadata={},
)

int_to_bool_func = FunctionDef(
    name="int_to_bool",
    input_type="builtins.int",
    output_type="builtins.bool",
    code="""return bool(x)""",
    usage_count=0,
    metadata={},
)

swapcase_func = FunctionDef(
    name="swapcase",
    input_type="builtins.str",
    output_type="builtins.str",
    code="""return x.swapcase()""",
    usage_count=0,
    metadata={},
)

list_is_empty_func = FunctionDef(
    name="list_is_empty",
    input_type="builtins.list",
    output_type="builtins.bool",
    code="""return len(x) == 0""",
    usage_count=0,
    metadata={},
)

f_exp_func = FunctionDef(
    name="f_exp",
    input_type="builtins.float",
    output_type="builtins.float",
    code="""import math; return math.exp(x)""",
    usage_count=0,
    metadata={},
)

reverse_func = FunctionDef(
    name="reverse",
    input_type="builtins.str",
    output_type="builtins.str",
    code="""return x[::-1]""",
    usage_count=0,
    metadata={},
)

list_sum_func = FunctionDef(
    name="list_sum",
    input_type="builtins.list",
    output_type="builtins.int",
    code="""return sum(x)""",
    usage_count=0,
    metadata={},
)

list_tail_func = FunctionDef(
    name="list_tail",
    input_type="builtins.list",
    output_type="builtins.list",
    code="""return x[1:]""",
    usage_count=0,
    metadata={},
)

int_to_str_func = FunctionDef(
    name="int_to_str",
    input_type="builtins.int",
    output_type="builtins.str",
    code="""return str(x)""",
    usage_count=0,
    metadata={},
)

bytes_length_func = FunctionDef(
    name="bytes_length",
    input_type="builtins.bytes",
    output_type="builtins.int",
    code="""return len(x)""",
    usage_count=0,
    metadata={},
)

f_floor_func = FunctionDef(
    name="f_floor",
    input_type="builtins.float",
    output_type="builtins.int",
    code="""return __import__('math').floor(x)""",
    usage_count=0,
    metadata={},
)

f_ceil_func = FunctionDef(
    name="f_ceil",
    input_type="builtins.float",
    output_type="builtins.int",
    code="""return __import__('math').ceil(x)""",
    usage_count=0,
    metadata={},
)

bytes_to_hex_func = FunctionDef(
    name="bytes_to_hex",
    input_type="builtins.bytes",
    output_type="builtins.str",
    code="""return x.hex()""",
    usage_count=0,
    metadata={},
)

set_size_func = FunctionDef(
    name="set_size",
    input_type="builtins.set",
    output_type="builtins.int",
    code="""return len(x)""",
    usage_count=0,
    metadata={},
)

dec_func = FunctionDef(
    name="dec",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return x - 1""",
    usage_count=0,
    metadata={},
)

set_hash_func = FunctionDef(
    name="set_hash",
    input_type="builtins.set",
    output_type="builtins.int",
    code="""return hash(frozenset(x))""",
    usage_count=0,
    metadata={},
)

int_bit_length_func = FunctionDef(
    name="int_bit_length",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return x.bit_length()""",
    usage_count=0,
    metadata={},
)

int_to_float_func = FunctionDef(
    name="int_to_float",
    input_type="builtins.int",
    output_type="builtins.float",
    code="""return float(x)""",
    usage_count=0,
    metadata={},
)

f_log10_func = FunctionDef(
    name="f_log10",
    input_type="builtins.float",
    output_type="builtins.float",
    code="""import math; return math.log10(x) if x > 0 else 0.0""",
    usage_count=0,
    metadata={},
)

square_func = FunctionDef(
    name="square",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return x * x""",
    usage_count=0,
    metadata={},
)

f_trunc_func = FunctionDef(
    name="f_trunc",
    input_type="builtins.float",
    output_type="builtins.int",
    code="""return int(x)""",
    usage_count=0,
    metadata={},
)

bool_to_int_func = FunctionDef(
    name="bool_to_int",
    input_type="builtins.bool",
    output_type="builtins.int",
    code="""return int(x)""",
    usage_count=0,
    metadata={},
)

float_to_str_func = FunctionDef(
    name="float_to_str",
    input_type="builtins.float",
    output_type="builtins.str",
    code="""return str(x)""",
    usage_count=0,
    metadata={},
)

dict_is_empty_func = FunctionDef(
    name="dict_is_empty",
    input_type="builtins.dict",
    output_type="builtins.bool",
    code="""return len(x) == 0""",
    usage_count=0,
    metadata={},
)

str_to_list_func = FunctionDef(
    name="str_to_list",
    input_type="builtins.str",
    output_type="builtins.list",
    code="""return list(x)""",
    usage_count=0,
    metadata={},
)

list_sorted_func = FunctionDef(
    name="list_sorted",
    input_type="builtins.list",
    output_type="builtins.list",
    code="""return sorted(x)""",
    usage_count=0,
    metadata={},
)

f_reciprocal_func = FunctionDef(
    name="f_reciprocal",
    input_type="builtins.float",
    output_type="builtins.float",
    code="""return float('inf') if x == 0 else 1.0 / x""",
    usage_count=0,
    metadata={},
)

count_a_func = FunctionDef(
    name="count_a",
    input_type="builtins.str",
    output_type="builtins.int",
    code="""return x.count('a')""",
    usage_count=0,
    metadata={},
)

title_func = FunctionDef(
    name="title",
    input_type="builtins.str",
    output_type="builtins.str",
    code="""return x.title()""",
    usage_count=0,
    metadata={},
)

is_upper_func = FunctionDef(
    name="is_upper",
    input_type="builtins.str",
    output_type="builtins.bool",
    code="""return x.isupper()""",
    usage_count=0,
    metadata={},
)

tuple_count_none_func = FunctionDef(
    name="tuple_count_none",
    input_type="builtins.tuple",
    output_type="builtins.int",
    code="""return x.count(None)""",
    usage_count=0,
    metadata={},
)

f_is_integer_func = FunctionDef(
    name="f_is_integer",
    input_type="builtins.float",
    output_type="builtins.bool",
    code="""return x.is_integer()""",
    usage_count=0,
    metadata={},
)

last_char_func = FunctionDef(
    name="last_char",
    input_type="builtins.str",
    output_type="builtins.str",
    code="""return x[-1] if x else ''""",
    usage_count=0,
    metadata={},
)

is_odd_func = FunctionDef(
    name="is_odd",
    input_type="builtins.int",
    output_type="builtins.bool",
    code="""return x % 2 == 1""",
    usage_count=0,
    metadata={},
)

bytes_is_ascii_func = FunctionDef(
    name="bytes_is_ascii",
    input_type="builtins.bytes",
    output_type="builtins.bool",
    code="""return all(b < 128 for b in x)""",
    usage_count=0,
    metadata={},
)

f_mod1_func = FunctionDef(
    name="f_mod1",
    input_type="builtins.float",
    output_type="builtins.float",
    code="""return x % 1.0""",
    usage_count=0,
    metadata={},
)

list_unique_func = FunctionDef(
    name="list_unique",
    input_type="builtins.list",
    output_type="builtins.list",
    code="""return list(dict.fromkeys(x))""",
    usage_count=0,
    metadata={},
)

contains_space_func = FunctionDef(
    name="contains_space",
    input_type="builtins.str",
    output_type="builtins.bool",
    code="""return ' ' in x""",
    usage_count=0,
    metadata={},
)

endswith_z_func = FunctionDef(
    name="endswith_z",
    input_type="builtins.str",
    output_type="builtins.bool",
    code="""return x.endswith('z')""",
    usage_count=0,
    metadata={},
)

sign_func = FunctionDef(
    name="sign",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return 1 if x > 0 else (-1 if x < 0 else 0)""",
    usage_count=0,
    metadata={},
)

str_reverse_words_func = FunctionDef(
    name="str_reverse_words",
    input_type="builtins.str",
    output_type="builtins.str",
    code="""return ' '.join(x.split()[::-1])""",
    usage_count=0,
    metadata={},
)

lower_func = FunctionDef(
    name="lower",
    input_type="builtins.str",
    output_type="builtins.str",
    code="""return x.lower()""",
    usage_count=0,
    metadata={},
)

is_lower_func = FunctionDef(
    name="is_lower",
    input_type="builtins.str",
    output_type="builtins.bool",
    code="""return x.islower()""",
    usage_count=0,
    metadata={},
)

int_clip_0_100_func = FunctionDef(
    name="int_clip_0_100",
    input_type="builtins.int",
    output_type="builtins.int",
    code="""return 0 if x < 0 else (100 if x > 100 else x)""",
    usage_count=0,
    metadata={},
)

is_numeric_func = FunctionDef(
    name="is_numeric",
    input_type="builtins.str",
    output_type="builtins.bool",
    code="""return x.isnumeric()""",
    usage_count=0,
    metadata={},
)

is_digit_func = FunctionDef(
    name="is_digit",
    input_type="builtins.str",
    output_type="builtins.bool",
    code="""return x.isdigit()""",
    usage_count=0,
    metadata={},
)

f_frac_percent_func = FunctionDef(
    name="f_frac_percent",
    input_type="builtins.float",
    output_type="builtins.int",
    code="""return int((x - int(x)) * 100)""",
    usage_count=0,
    metadata={},
)

is_space_func = FunctionDef(
    name="is_space",
    input_type="builtins.str",
    output_type="builtins.bool",
    code="""return x.isspace()""",
    usage_count=0,
    metadata={},
)

capitalize_func = FunctionDef(
    name="capitalize",
    input_type="builtins.str",
    output_type="builtins.str",
    code="""return x.capitalize()""",
    usage_count=0,
    metadata={},
)

bytes_is_empty_func = FunctionDef(
    name="bytes_is_empty",
    input_type="builtins.bytes",
    output_type="builtins.bool",
    code="""return len(x) == 0""",
    usage_count=0,
    metadata={},
)

duplicate_func = FunctionDef(
    name="duplicate",
    input_type="builtins.str",
    output_type="builtins.str",
    code="""return x + x""",
    usage_count=0,
    metadata={},
)

# Trajectory specifications
_specs = [
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList([list_median_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList([tuple_length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList([inc_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList([is_negative_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList([bool_to_str_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [0, 1]}, {"__bytes__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList([bytes_reverse_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList([f_sin_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList([tuple_reverse_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList([set_to_list_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList([tuple_to_index_dict_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_abs_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList([bytearray_reverse_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList([str_hash_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_list_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList([f_sin_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList([list_median_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_conjugate_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList([dict_items_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList([f_round_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList([half_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_max_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList([tuple_length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList([mod2_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList([upper_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList([identity_int_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_sum_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList([set_to_list_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList([int_is_power_of_two_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_abs_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList([list_max_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList([startswith_a_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList([bool_to_float_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList([bool_to_str_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList([neg_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList([dict_has_duplicate_values_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList([f_abs_sqrt_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList([is_positive_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList([tuple_to_index_dict_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList([repeat_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList([tuple_to_index_dict_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_phase_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList([bytearray_to_bytes_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_max_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList([set_to_list_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [0, 1]}, {"__bytes__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList([bytes_upper_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList([f_square_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_list_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_max_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList([tuple_to_list_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList([dict_items_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList([bytearray_reverse_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList([str_is_palindrome_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_phase_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList([bytearray_to_bytes_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList([tuple_reverse_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList([abs_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList([bool_not_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList([dict_keys_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList([is_title_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList([set_is_empty_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [0, 1]}, {"__bytes__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList([bytes_reverse_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList([list_reverse_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_list_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList([tuple_is_empty_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList([dict_freeze_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [0, 1]}, {"__bytes__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList([bytes_upper_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList([dict_length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_phase_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_phase_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_list_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList([dict_keyset_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList([bytearray_to_bytes_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_phase_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList([str_count_vowels_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList([tuple_to_list_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList([bool_identity_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList([f_abs_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_list_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList([f_fraction_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList([first_char_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList([length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList([tuple_is_empty_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList([dict_flip_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList([dict_values_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList([is_positive_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList([set_to_list_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList([double_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList([tuple_to_list_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList([f_abs_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList([list_min_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_imag_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_real_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList([bool_not_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList([bytearray_length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList([bytearray_length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList([strip_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList([tuple_reverse_func, tuple_is_empty_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList([tuple_length_func, neg_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [0, 1]}, {"__bytes__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList([bytes_reverse_func, bytes_reverse_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList([str_remove_digits_func, is_alpha_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList([int_is_power_of_two_func, bool_not_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_real_func, f_fraction_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_length_func, is_even_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_max_func, int_is_power_of_two_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [0, 1]}, {"__bytes__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList([bytes_reverse_func, bytes_upper_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList([neg_func, int_popcount_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [bytearray_reverse_func, bytearray_to_bytes_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList([list_length_func, int_to_bool_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList([bytearray_length_func, mod2_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList([tuple_is_empty_func, bool_to_str_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList([upper_func, swapcase_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_length_func, abs_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList([tuple_is_empty_func, bool_to_str_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList([list_is_empty_func, bool_to_float_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList([f_exp_func, f_abs_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_abs_func, f_abs_sqrt_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList([reverse_func, str_is_palindrome_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList([list_sum_func, double_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_real_func, f_exp_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList([list_tail_func, list_reverse_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList([tuple_length_func, int_to_str_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_phase_func, f_round_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [bytearray_reverse_func, bytearray_to_bytes_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList([bytearray_to_bytes_func, bytes_length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_sum_func, inc_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_phase_func, f_floor_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList([f_ceil_func, half_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList([bytearray_length_func, mod2_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList([dict_items_func, list_length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [0, 1]}, {"__bytes__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList([bytes_to_hex_func, length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList([tuple_is_empty_func, bool_to_float_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList([set_size_func, double_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList([str_count_vowels_func, dec_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList([int_to_str_func, swapcase_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList([set_hash_func, int_to_str_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList([bytearray_reverse_func, bytearray_reverse_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList([str_hash_func, half_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList([dict_keys_func, list_min_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList([str_remove_digits_func, reverse_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList([int_bit_length_func, int_to_float_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList([f_log10_func, f_square_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList([bytearray_length_func, neg_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList([str_count_vowels_func, square_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList([dict_flip_func, dict_items_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList([dict_flip_func, dict_keyset_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList([tuple_is_empty_func, bool_to_str_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList([f_exp_func, f_trunc_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList([bool_to_int_func, int_bit_length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList([list_length_func, identity_int_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList([dict_length_func, mod2_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList([bool_not_func, bool_to_float_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList([bytearray_length_func, is_even_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_imag_func, float_to_str_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_sum_func, int_bit_length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList([dict_is_empty_func, bool_identity_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList([dict_values_func, list_sum_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList([bool_identity_func, bool_not_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList([str_to_list_func, list_tail_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_real_func, f_ceil_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList([bytearray_to_bytes_func, bytes_reverse_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList([set_size_func, dec_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList([is_negative_func, bool_to_int_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList([is_even_func, bool_to_str_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList([dict_freeze_func, tuple_is_empty_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList([list_sorted_func, list_length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_list_func, list_min_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList([set_is_empty_func, bool_identity_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_abs_func, f_reciprocal_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList([tuple_is_empty_func, bool_to_float_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList([length_func, dec_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [0, 1]}, {"__bytes__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList([bytes_upper_func, bytes_upper_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList([bool_to_float_func, f_abs_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList([set_hash_func, is_positive_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_conjugate_func, complex_conjugate_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [bytearray_length_func, int_is_power_of_two_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList([bytearray_length_func, double_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList([identity_int_func, half_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList([count_a_func, square_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList([bool_not_func, bool_to_float_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList([bytearray_to_bytes_func, bytes_to_hex_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_sum_func, is_even_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList([int_to_bool_func, bool_to_float_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList([dict_flip_func, dict_keyset_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_list_func, list_tail_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList([bool_to_int_func, abs_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [0, 1]}, {"__bytes__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList([bytes_to_hex_func, title_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList([is_upper_func, bool_to_float_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList([float_to_str_func, first_char_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList([bytearray_reverse_func, bytearray_reverse_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList([bytearray_to_bytes_func, bytes_to_hex_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList([list_tail_func, list_length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList([bytearray_length_func, abs_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList([tuple_reverse_func, tuple_count_none_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_phase_func, f_is_integer_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList([f_is_integer_func, bool_to_str_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList([float_to_str_func, last_char_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [0, 1]}, {"__bytes__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList([bytes_to_hex_func, upper_func, last_char_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [0, 1]}, {"__bytes__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList([bytes_length_func, abs_func, neg_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [range_length_func, int_is_power_of_two_func, bool_to_int_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList(
            [list_sorted_func, list_tail_func, list_reverse_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [complex_phase_func, f_trunc_func, int_is_power_of_two_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList(
            [int_bit_length_func, identity_int_func, is_negative_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList([is_odd_func, bool_identity_func, bool_not_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [bytearray_length_func, is_positive_func, bool_to_float_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [complex_imag_func, f_is_integer_func, bool_identity_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [tuple_is_empty_func, bool_not_func, bool_identity_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList([list_max_func, half_func, neg_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList(
            [set_is_empty_func, bool_to_str_func, strip_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [tuple_count_none_func, abs_func, int_to_float_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [tuple_is_empty_func, bool_to_str_func, str_to_list_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [0, 1]}, {"__bytes__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [bytes_is_ascii_func, bool_to_float_func, f_reciprocal_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [tuple_count_none_func, is_negative_func, bool_to_float_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList([f_fraction_func, f_mod1_func, f_sin_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList(
            [list_sorted_func, list_unique_func, list_tail_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [bytearray_to_bytes_func, bytes_to_hex_func, repeat_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList(
            [set_to_list_func, list_length_func, square_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [tuple_to_list_func, list_median_func, f_trunc_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [0, 1]}, {"__bytes__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [bytes_to_hex_func, contains_space_func, bool_to_int_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList([length_func, double_func, mod2_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [0, 1]}, {"__bytes__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytes_to_hex_func,
                str_count_vowels_func,
                int_is_power_of_two_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [range_list_func, list_sorted_func, list_reverse_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [tuple_to_index_dict_func, dict_keyset_func, set_hash_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_has_duplicate_values_func,
                bool_identity_func,
                bool_identity_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList(
            [list_is_empty_func, bool_to_str_func, endswith_z_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList(
            [set_to_list_func, list_is_empty_func, bool_to_int_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [range_length_func, is_even_func, bool_identity_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [complex_real_func, f_is_integer_func, bool_to_str_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [0, 1]}, {"__bytes__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [bytes_length_func, is_negative_func, bool_not_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [tuple_reverse_func, tuple_to_list_func, list_median_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [dict_values_func, list_length_func, int_to_bool_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [bytearray_length_func, is_even_func, bool_to_int_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList([bytearray_length_func, abs_func, sign_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList(
            [str_reverse_words_func, lower_func, swapcase_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList([list_sorted_func, list_length_func, half_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList(
            [set_to_list_func, list_sum_func, is_positive_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList([dict_length_func, dec_func, half_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList(
            [str_remove_digits_func, reverse_func, is_lower_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList(
            [list_is_empty_func, bool_identity_func, bool_not_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList(
            [bool_to_str_func, reverse_func, startswith_a_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [0, 1]}, {"__bytes__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [bytes_reverse_func, bytes_length_func, inc_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [bytearray_length_func, int_clip_0_100_func, inc_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_reverse_func,
                bytearray_reverse_func,
                bytearray_reverse_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList(
            [int_to_str_func, str_hash_func, int_popcount_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList([complex_phase_func, f_abs_func, f_log10_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList(
            [identity_int_func, int_clip_0_100_func, int_popcount_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [0, 1]}, {"__bytes__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [bytes_to_hex_func, str_reverse_words_func, str_to_list_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_list_func, list_length_func, neg_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList(
            [is_negative_func, bool_to_str_func, is_upper_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [bytearray_to_bytes_func, bytes_to_hex_func, contains_space_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList(
            [bool_to_int_func, double_func, int_to_float_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList([f_sin_func, float_to_str_func, is_numeric_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList([title_func, is_digit_func, bool_to_int_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList([set_hash_func, neg_func, sign_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList([f_mod1_func, f_sin_func, f_square_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [complex_conjugate_func, complex_imag_func, f_fraction_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList([f_round_func, int_to_str_func, reverse_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [dict_keyset_func, set_size_func, int_to_float_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [complex_imag_func, f_mod1_func, f_is_integer_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList(
            [set_is_empty_func, bool_to_float_func, f_frac_percent_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList([f_log10_func, f_exp_func, f_fraction_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList([list_sum_func, inc_func, int_bit_length_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList([f_log10_func, float_to_str_func, is_space_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [tuple_to_index_dict_func, dict_length_func, neg_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [dict_keyset_func, set_is_empty_func, bool_not_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_to_bytes_func,
                bytes_is_ascii_func,
                bool_identity_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList([range_sum_func, int_bit_length_func, dec_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList(
            [f_ceil_func, int_is_power_of_two_func, bool_not_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList(
            [
                int_clip_0_100_func,
                int_is_power_of_two_func,
                bool_to_float_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [tuple_to_list_func, list_max_func, int_to_float_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [complex_imag_func, f_reciprocal_func, f_square_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList([dict_keyset_func, set_size_func, abs_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList([f_log10_func, f_exp_func, f_is_integer_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList(
            [set_hash_func, int_to_bool_func, bool_identity_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList([f_log10_func, f_floor_func, square_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList(
            [is_lower_func, bool_to_str_func, is_numeric_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [complex_conjugate_func, complex_imag_func, f_trunc_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [range_list_func, list_reverse_func, list_tail_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList(
            [list_max_func, int_to_bool_func, bool_to_str_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList(
            [bool_identity_func, bool_identity_func, bool_to_float_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [dict_length_func, int_bit_length_func, mod2_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_has_duplicate_values_func,
                bool_to_int_func,
                int_to_float_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList([bool_to_float_func, f_exp_func, f_trunc_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList(
            [set_size_func, int_is_power_of_two_func, bool_to_float_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList([dict_values_func, list_median_func, f_sin_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList([abs_func, is_positive_func, bool_to_str_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList(
            [list_median_func, float_to_str_func, str_is_palindrome_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_reverse_func,
                bytearray_reverse_func,
                bytearray_length_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [0, 1]}, {"__bytes__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [bytes_is_ascii_func, bool_not_func, bool_identity_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList(
            [set_to_list_func, list_is_empty_func, bool_to_str_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList(
            [f_mod1_func, f_frac_percent_func, int_to_bool_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList([dict_length_func, dec_func, mod2_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList([f_trunc_func, int_to_str_func, lower_func]),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList(
            [bool_to_float_func, f_abs_sqrt_func, f_mod1_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList(
            [bool_to_int_func, double_func, int_to_float_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList(
            [int_to_float_func, f_abs_sqrt_func, f_trunc_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [tuple_reverse_func, tuple_reverse_func, tuple_is_empty_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_is_empty_func,
                bool_to_float_func,
                f_mod1_func,
                f_round_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                range_length_func,
                is_positive_func,
                bool_to_float_func,
                f_round_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList(
            [
                int_is_power_of_two_func,
                bool_identity_func,
                bool_to_float_func,
                f_floor_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList(
            [list_max_func, identity_int_func, is_even_func, bool_not_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                set_to_list_func,
                list_median_func,
                f_abs_func,
                float_to_str_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_keys_func,
                list_is_empty_func,
                bool_to_float_func,
                f_fraction_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [range_sum_func, inc_func, is_odd_func, bool_to_int_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList(
            [
                list_tail_func,
                list_median_func,
                f_frac_percent_func,
                int_clip_0_100_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_keyset_func,
                set_hash_func,
                double_func,
                int_clip_0_100_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList(
            [
                int_is_power_of_two_func,
                bool_to_int_func,
                int_to_str_func,
                is_upper_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList(
            [
                startswith_a_func,
                bool_to_str_func,
                capitalize_func,
                endswith_z_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                complex_real_func,
                f_frac_percent_func,
                int_to_str_func,
                title_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_reverse_func,
                bytearray_to_bytes_func,
                bytes_upper_func,
                bytes_to_hex_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_reverse_func,
                bytearray_to_bytes_func,
                bytes_is_ascii_func,
                bool_to_float_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                range_sum_func,
                is_even_func,
                bool_to_float_func,
                f_frac_percent_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_to_bytes_func,
                bytes_reverse_func,
                bytes_reverse_func,
                bytes_upper_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList(
            [
                bool_to_str_func,
                is_alpha_func,
                bool_identity_func,
                bool_to_str_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [dict_freeze_func, tuple_count_none_func, sign_func, mod2_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList(
            [
                int_is_power_of_two_func,
                bool_identity_func,
                bool_to_float_func,
                f_abs_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList(
            [
                int_clip_0_100_func,
                int_popcount_func,
                int_is_power_of_two_func,
                bool_to_str_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList(
            [first_char_func, repeat_func, swapcase_func, str_hash_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_flip_func,
                dict_items_func,
                list_median_func,
                f_frac_percent_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList(
            [f_exp_func, f_floor_func, int_to_str_func, reverse_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList(
            [list_tail_func, list_min_func, is_even_func, bool_identity_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_to_bytes_func,
                bytes_upper_func,
                bytes_upper_func,
                bytes_reverse_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_reverse_func,
                tuple_reverse_func,
                tuple_reverse_func,
                tuple_to_list_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                complex_real_func,
                f_fraction_func,
                float_to_str_func,
                str_reverse_words_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [tuple_count_none_func, sign_func, abs_func, abs_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList(
            [int_is_power_of_two_func, bool_to_int_func, neg_func, neg_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList(
            [
                list_is_empty_func,
                bool_to_float_func,
                f_square_func,
                f_frac_percent_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList(
            [
                lower_func,
                is_digit_func,
                bool_to_float_func,
                f_is_integer_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [dict_items_func, list_tail_func, list_tail_func, list_max_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                range_list_func,
                list_sorted_func,
                list_length_func,
                identity_int_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList(
            [mod2_func, inc_func, is_odd_func, bool_not_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                set_to_list_func,
                list_sorted_func,
                list_sorted_func,
                list_length_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_is_empty_func,
                bool_to_float_func,
                f_round_func,
                is_negative_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_to_bytes_func,
                bytes_to_hex_func,
                count_a_func,
                square_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_flip_func,
                dict_flip_func,
                dict_flip_func,
                dict_values_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList(
            [
                bool_not_func,
                bool_identity_func,
                bool_not_func,
                bool_identity_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList(
            [list_sorted_func, list_length_func, inc_func, int_to_str_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList(
            [
                list_unique_func,
                list_min_func,
                is_negative_func,
                bool_not_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_flip_func,
                dict_keyset_func,
                set_is_empty_func,
                bool_to_int_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                set_to_list_func,
                list_unique_func,
                list_unique_func,
                list_min_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList(
            [str_remove_digits_func, length_func, double_func, square_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [complex_real_func, f_exp_func, f_mod1_func, f_trunc_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList(
            [first_char_func, strip_func, is_space_func, bool_identity_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_values_func,
                list_max_func,
                is_odd_func,
                bool_to_float_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_reverse_func,
                tuple_length_func,
                is_negative_func,
                bool_not_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                complex_conjugate_func,
                complex_phase_func,
                f_ceil_func,
                int_to_str_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [dict_length_func, int_bit_length_func, half_func, half_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList(
            [set_size_func, double_func, identity_int_func, is_odd_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_length_func,
                int_to_float_func,
                f_mod1_func,
                f_abs_sqrt_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_to_list_func,
                list_length_func,
                int_clip_0_100_func,
                half_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [dict_freeze_func, tuple_length_func, abs_func, int_to_bool_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_to_index_dict_func,
                dict_items_func,
                list_min_func,
                int_to_bool_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_is_empty_func,
                bool_identity_func,
                bool_identity_func,
                bool_to_str_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList(
            [bool_to_int_func, int_to_float_func, f_floor_func, dec_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList(
            [bool_to_float_func, f_abs_func, f_exp_func, f_abs_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_is_empty_func,
                bool_to_str_func,
                is_lower_func,
                bool_to_float_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList(
            [set_hash_func, int_to_str_func, title_func, count_a_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList(
            [
                str_is_palindrome_func,
                bool_not_func,
                bool_not_func,
                bool_to_float_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [range_length_func, sign_func, inc_func, is_negative_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList(
            [repeat_func, title_func, lower_func, title_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList(
            [
                bool_to_float_func,
                f_trunc_func,
                int_popcount_func,
                int_bit_length_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [range_sum_func, sign_func, int_bit_length_func, abs_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                range_length_func,
                identity_int_func,
                int_to_str_func,
                is_alpha_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                range_max_func,
                int_to_float_func,
                f_is_integer_func,
                bool_not_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList(
            [
                bool_identity_func,
                bool_to_str_func,
                repeat_func,
                is_upper_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [dict_length_func, abs_func, dec_func, int_to_str_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [0, 1]}, {"__bytes__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytes_is_empty_func,
                bool_to_str_func,
                str_reverse_words_func,
                is_numeric_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList(
            [
                upper_func,
                str_count_vowels_func,
                int_to_float_func,
                f_fraction_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                set_hash_func,
                is_odd_func,
                bool_identity_func,
                bool_identity_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList(
            [list_length_func, abs_func, is_odd_func, bool_not_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_to_list_func,
                list_tail_func,
                list_length_func,
                is_negative_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList(
            [is_even_func, bool_to_float_func, f_round_func, sign_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList(
            [f_fraction_func, f_mod1_func, f_sin_func, f_is_integer_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                complex_abs_func,
                f_floor_func,
                square_func,
                int_bit_length_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_values_func,
                list_unique_func,
                list_reverse_func,
                list_is_empty_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList(
            [duplicate_func, endswith_z_func, bool_not_func, bool_not_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [tuple_count_none_func, dec_func, neg_func, sign_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList(
            [list_sum_func, abs_func, int_to_bool_func, bool_to_str_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_to_list_func,
                list_length_func,
                is_even_func,
                bool_to_float_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList(
            [
                list_sorted_func,
                list_tail_func,
                list_length_func,
                int_is_power_of_two_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList(
            [list_length_func, sign_func, inc_func, int_to_bool_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList(
            [
                first_char_func,
                duplicate_func,
                str_count_vowels_func,
                abs_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                range_length_func,
                is_negative_func,
                bool_to_str_func,
                is_digit_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                range_length_func,
                double_func,
                int_is_power_of_two_func,
                bool_to_str_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_to_bytes_func,
                bytes_upper_func,
                bytes_to_hex_func,
                duplicate_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                range_list_func,
                list_reverse_func,
                list_reverse_func,
                list_reverse_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_is_empty_func,
                bool_to_str_func,
                upper_func,
                str_hash_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [range_max_func, dec_func, double_func, is_odd_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                range_sum_func,
                is_negative_func,
                bool_not_func,
                bool_to_int_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [dict_items_func, list_length_func, int_to_str_func, title_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_length_func,
                half_func,
                int_is_power_of_two_func,
                bool_to_int_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList(
            [bool_not_func, bool_to_float_func, f_abs_func, f_fraction_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList(
            [
                list_unique_func,
                list_max_func,
                half_func,
                int_is_power_of_two_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_length_func,
                int_clip_0_100_func,
                int_to_float_func,
                f_is_integer_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_is_empty_func,
                bool_to_float_func,
                f_reciprocal_func,
                f_square_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [bytearray_length_func, abs_func, int_to_float_func, f_abs_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList(
            [set_hash_func, is_odd_func, bool_to_str_func, str_hash_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                range_max_func,
                identity_int_func,
                mod2_func,
                sign_func,
                sign_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                set_to_list_func,
                list_length_func,
                int_bit_length_func,
                is_even_func,
                bool_to_int_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                range_max_func,
                sign_func,
                int_to_str_func,
                title_func,
                str_remove_digits_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList(
            [
                neg_func,
                is_odd_func,
                bool_to_float_func,
                f_ceil_func,
                sign_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [bytearray_length_func, abs_func, inc_func, inc_func, dec_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList(
            [
                f_mod1_func,
                f_sin_func,
                f_round_func,
                identity_int_func,
                square_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList(
            [
                f_abs_func,
                f_round_func,
                is_negative_func,
                bool_to_int_func,
                neg_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList(
            [
                int_is_power_of_two_func,
                bool_to_float_func,
                f_exp_func,
                f_exp_func,
                f_floor_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_has_duplicate_values_func,
                bool_not_func,
                bool_to_float_func,
                f_frac_percent_func,
                int_to_float_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_is_empty_func,
                bool_to_float_func,
                f_exp_func,
                f_exp_func,
                f_trunc_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                set_hash_func,
                is_positive_func,
                bool_identity_func,
                bool_identity_func,
                bool_to_float_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [0, 1]}, {"__bytes__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytes_is_empty_func,
                bool_to_float_func,
                f_mod1_func,
                f_log10_func,
                f_log10_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                range_sum_func,
                int_popcount_func,
                square_func,
                int_to_float_func,
                f_trunc_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [0, 1]}, {"__bytes__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytes_to_hex_func,
                capitalize_func,
                count_a_func,
                half_func,
                inc_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                complex_real_func,
                f_fraction_func,
                f_log10_func,
                f_floor_func,
                is_odd_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                set_size_func,
                half_func,
                int_to_float_func,
                float_to_str_func,
                str_is_palindrome_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_reverse_func,
                bytearray_length_func,
                sign_func,
                is_odd_func,
                bool_to_int_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList(
            [
                float_to_str_func,
                str_to_list_func,
                list_max_func,
                is_negative_func,
                bool_to_int_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                complex_abs_func,
                f_log10_func,
                f_exp_func,
                f_floor_func,
                is_positive_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList(
            [
                is_digit_func,
                bool_not_func,
                bool_identity_func,
                bool_identity_func,
                bool_to_float_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList(
            [
                f_ceil_func,
                is_positive_func,
                bool_to_float_func,
                f_frac_percent_func,
                abs_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_reverse_func,
                tuple_length_func,
                mod2_func,
                mod2_func,
                half_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_reverse_func,
                tuple_is_empty_func,
                bool_to_str_func,
                repeat_func,
                contains_space_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList(
            [
                neg_func,
                square_func,
                is_odd_func,
                bool_to_float_func,
                f_square_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_to_index_dict_func,
                dict_length_func,
                is_positive_func,
                bool_to_float_func,
                f_is_integer_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_items_func,
                list_unique_func,
                list_max_func,
                inc_func,
                mod2_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList(
            [set_hash_func, mod2_func, sign_func, inc_func, mod2_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_length_func,
                half_func,
                is_negative_func,
                bool_to_str_func,
                reverse_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_length_func,
                is_odd_func,
                bool_not_func,
                bool_to_int_func,
                dec_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList(
            [
                list_sum_func,
                int_is_power_of_two_func,
                bool_to_str_func,
                repeat_func,
                upper_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                set_to_list_func,
                list_max_func,
                int_to_float_func,
                f_is_integer_func,
                bool_to_int_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_to_list_func,
                list_min_func,
                int_to_str_func,
                reverse_func,
                str_to_list_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList(
            [
                str_hash_func,
                identity_int_func,
                int_is_power_of_two_func,
                bool_not_func,
                bool_to_str_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                range_max_func,
                is_odd_func,
                bool_to_float_func,
                f_log10_func,
                f_fraction_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList(
            [
                f_floor_func,
                int_to_str_func,
                is_space_func,
                bool_not_func,
                bool_to_int_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList(
            [
                list_unique_func,
                list_length_func,
                square_func,
                is_even_func,
                bool_to_str_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                complex_imag_func,
                f_trunc_func,
                double_func,
                is_negative_func,
                bool_to_float_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                set_is_empty_func,
                bool_not_func,
                bool_to_float_func,
                f_square_func,
                f_trunc_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_is_empty_func,
                bool_not_func,
                bool_identity_func,
                bool_to_int_func,
                mod2_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [0, 1]}, {"__bytes__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytes_reverse_func,
                bytes_is_empty_func,
                bool_to_float_func,
                f_abs_func,
                f_abs_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_freeze_func,
                tuple_is_empty_func,
                bool_to_str_func,
                first_char_func,
                startswith_a_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList(
            [
                double_func,
                dec_func,
                abs_func,
                int_to_bool_func,
                bool_not_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_count_none_func,
                abs_func,
                mod2_func,
                square_func,
                is_even_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_keys_func,
                list_sorted_func,
                list_sum_func,
                is_odd_func,
                bool_to_str_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_reverse_func,
                bytearray_reverse_func,
                bytearray_to_bytes_func,
                bytes_upper_func,
                bytes_upper_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList(
            [
                is_even_func,
                bool_to_float_func,
                f_reciprocal_func,
                f_ceil_func,
                double_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList(
            [list_max_func, square_func, sign_func, neg_func, half_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                set_hash_func,
                dec_func,
                neg_func,
                is_odd_func,
                bool_to_str_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList(
            [
                f_log10_func,
                f_log10_func,
                f_floor_func,
                int_to_str_func,
                duplicate_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList(
            [
                bool_identity_func,
                bool_to_int_func,
                double_func,
                int_clip_0_100_func,
                is_negative_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_to_bytes_func,
                bytes_to_hex_func,
                upper_func,
                is_space_func,
                bool_to_float_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList(
            [
                f_square_func,
                f_fraction_func,
                f_log10_func,
                f_abs_sqrt_func,
                f_ceil_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList(
            [
                list_sum_func,
                double_func,
                int_to_bool_func,
                bool_to_int_func,
                inc_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList(
            [
                title_func,
                lower_func,
                strip_func,
                last_char_func,
                startswith_a_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                complex_conjugate_func,
                complex_abs_func,
                f_is_integer_func,
                bool_to_int_func,
                int_to_str_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList(
            [
                identity_int_func,
                int_clip_0_100_func,
                int_to_str_func,
                length_func,
                inc_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                complex_phase_func,
                f_abs_sqrt_func,
                f_mod1_func,
                f_abs_func,
                f_fraction_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                complex_real_func,
                f_log10_func,
                f_trunc_func,
                int_bit_length_func,
                int_is_power_of_two_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_keyset_func,
                set_is_empty_func,
                bool_identity_func,
                bool_to_str_func,
                is_title_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList(
            [
                f_exp_func,
                f_mod1_func,
                f_fraction_func,
                f_trunc_func,
                int_to_bool_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_items_func,
                list_sorted_func,
                list_tail_func,
                list_unique_func,
                list_length_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_length_func,
                int_to_str_func,
                str_to_list_func,
                list_sum_func,
                mod2_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList(
            [
                double_func,
                is_negative_func,
                bool_to_str_func,
                is_title_func,
                bool_to_float_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                complex_conjugate_func,
                complex_real_func,
                f_round_func,
                int_to_bool_func,
                bool_not_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_length_func,
                int_to_str_func,
                is_alpha_func,
                bool_identity_func,
                bool_to_str_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_to_bytes_func,
                bytes_upper_func,
                bytes_reverse_func,
                bytes_is_ascii_func,
                bool_identity_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_to_bytes_func,
                bytes_upper_func,
                bytes_is_empty_func,
                bool_to_float_func,
                f_ceil_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                range_length_func,
                int_popcount_func,
                inc_func,
                inc_func,
                int_is_power_of_two_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList(
            [
                int_clip_0_100_func,
                half_func,
                inc_func,
                neg_func,
                int_to_str_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList(
            [
                f_log10_func,
                f_reciprocal_func,
                f_fraction_func,
                f_reciprocal_func,
                f_trunc_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_reverse_func,
                tuple_reverse_func,
                tuple_reverse_func,
                tuple_count_none_func,
                int_to_bool_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [range_max_func, dec_func, abs_func, inc_func, is_odd_func]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_to_index_dict_func,
                dict_is_empty_func,
                bool_not_func,
                bool_to_str_func,
                str_remove_digits_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                set_is_empty_func,
                bool_to_float_func,
                f_floor_func,
                int_to_float_func,
                f_floor_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList(
            [
                bool_identity_func,
                bool_to_str_func,
                is_upper_func,
                bool_to_float_func,
                f_log10_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_reverse_func,
                bytearray_to_bytes_func,
                bytes_is_empty_func,
                bool_not_func,
                bool_to_float_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList(
            [
                bool_to_str_func,
                str_remove_digits_func,
                capitalize_func,
                str_is_palindrome_func,
                bool_not_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.int", "items": [0, 1, 2, 3, 4, 5, -1, -2]}"""
        ),
        function_defs=FunctionDefList(
            [
                int_is_power_of_two_func,
                bool_identity_func,
                bool_to_int_func,
                int_to_bool_func,
                bool_to_int_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList(
            [
                bool_to_int_func,
                int_clip_0_100_func,
                int_to_bool_func,
                bool_to_float_func,
                f_sin_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList(
            [
                str_hash_func,
                square_func,
                int_to_bool_func,
                bool_to_str_func,
                repeat_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList(
            [
                list_reverse_func,
                list_sorted_func,
                list_sum_func,
                abs_func,
                int_bit_length_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytes", "items": [{"__bytes__": []}, {"__bytes__": [0, 1]}, {"__bytes__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytes_upper_func,
                bytes_reverse_func,
                bytes_length_func,
                int_bit_length_func,
                int_to_str_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_reverse_func,
                tuple_reverse_func,
                tuple_to_list_func,
                list_median_func,
                f_square_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.str", "items": ["", "a", "Ab", "hello world", "123", "123.45"]}"""
        ),
        function_defs=FunctionDefList(
            [
                swapcase_func,
                is_alpha_func,
                bool_to_str_func,
                last_char_func,
                is_numeric_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.range", "items": [{"__range__": [0, 0, 1]}, {"__range__": [0, 3, 1]}, {"__range__": [1, 4, 1]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                range_length_func,
                double_func,
                neg_func,
                int_is_power_of_two_func,
                bool_identity_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                set_to_list_func,
                list_reverse_func,
                list_tail_func,
                list_reverse_func,
                list_unique_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList(
            [
                f_square_func,
                f_reciprocal_func,
                f_exp_func,
                f_abs_sqrt_func,
                f_round_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList(
            [
                list_tail_func,
                list_reverse_func,
                list_unique_func,
                list_sum_func,
                inc_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_length_func,
                is_even_func,
                bool_identity_func,
                bool_to_str_func,
                title_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bool", "items": [true, false]}"""
        ),
        function_defs=FunctionDefList(
            [
                bool_to_float_func,
                float_to_str_func,
                lower_func,
                reverse_func,
                endswith_z_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_to_bytes_func,
                bytes_length_func,
                int_to_float_func,
                f_abs_sqrt_func,
                float_to_str_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.float", "items": [0.5, 2.5, 4.0, -0.5, -2.5]}"""
        ),
        function_defs=FunctionDefList(
            [
                f_exp_func,
                float_to_str_func,
                contains_space_func,
                bool_to_str_func,
                is_title_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.list", "items": [[], [3, 1, 2], [1, 2, 3]]}"""
        ),
        function_defs=FunctionDefList(
            [
                list_median_func,
                f_is_integer_func,
                bool_not_func,
                bool_identity_func,
                bool_to_int_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                complex_real_func,
                f_floor_func,
                inc_func,
                int_to_float_func,
                f_floor_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.set", "items": [{"__set__": []}, {"__set__": [1]}, {"__set__": [2]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                set_size_func,
                int_to_float_func,
                f_square_func,
                float_to_str_func,
                str_is_palindrome_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.dict", "items": [{}, {"a": 1}, {"x": 1, "y": 2}]}"""
        ),
        function_defs=FunctionDefList(
            [
                dict_flip_func,
                dict_flip_func,
                dict_is_empty_func,
                bool_to_int_func,
                int_popcount_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.bytearray", "items": [{"__bytearray__": []}, {"__bytearray__": [0, 1]}, {"__bytearray__": [97, 98, 99]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                bytearray_reverse_func,
                bytearray_length_func,
                int_to_bool_func,
                bool_to_float_func,
                f_exp_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_to_list_func,
                list_is_empty_func,
                bool_to_str_func,
                strip_func,
                strip_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.tuple", "items": [{"__tuple__": []}, {"__tuple__": [1, 2, 3]}, {"__tuple__": ["a", "b", "c"]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                tuple_length_func,
                is_even_func,
                bool_to_float_func,
                f_trunc_func,
                int_to_str_func,
            ]
        ),
    ),
    TrajectorySpec(
        input_list=TypedList.from_str(
            """{"type": "builtins.complex", "items": [{"__complex__": [1.0, 2.0]}, {"__complex__": [-1.0, 0.0]}, {"__complex__": [0.0, -3.0]}]}"""
        ),
        function_defs=FunctionDefList(
            [
                complex_real_func,
                f_fraction_func,
                f_abs_func,
                f_trunc_func,
                is_negative_func,
            ]
        ),
    ),
]

eval_trajectory_specs = TrajectorySpecList(_specs)
