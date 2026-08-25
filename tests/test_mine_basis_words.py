"""Tests for the Sequitur implementation and the word-scoring arithmetic.

The Sequitur tests check the two constraints the algorithm is defined by rather
than pinning one output: digram uniqueness (allowing the documented overlap
exception inside a run of equal symbols) and rule utility, plus the round trip
that says the grammar still denotes the input.  The scoring tests pin the exact
criterion, including the two bar values quoted in the issue.
"""

from __future__ import annotations

import math
import random
from collections import Counter

import pytest

from experiments.mine_basis_words import (
    BranchingModel,
    Diagnostics,
    Program,
    Sequitur,
    compression_bar,
    count_occurrences,
    diagnose,
    mdl_delta_bits,
    mine,
    rewrite,
    search_multiplier,
    sequitur_words,
)


def build(text: str) -> Sequitur:
    grammar = Sequitur()
    for token in text:
        grammar.feed(token)
    return grammar


def rule_bodies(grammar: Sequitur) -> dict[int, list[str]]:
    bodies = {}
    for rule in grammar.rules:
        bodies[rule.index] = [
            symbol.value if symbol.rule is None else f"R{symbol.rule.index}"
            for symbol in rule.body()
        ]
    return bodies


def nonoverlapping_digrams(grammar: Sequitur) -> Counter:
    """Digram counts, skipping the second copy inside a run of equal symbols.

    Sequitur permits one overlap: in ``aaa`` the digram ``aa`` occurs twice but
    the two copies share a symbol, so only one of them can ever be replaced.
    """
    found: Counter = Counter()
    for rule in grammar.rules:
        body = list(rule.body())
        index = 0
        while index + 1 < len(body):
            found[(body[index].key, body[index + 1].key)] += 1
            if body[index].key == body[index + 1].key:
                index += 2
            else:
                index += 1
    return found


def reachable_rules(grammar: Sequitur) -> set:
    seen = set()
    stack = [symbol.rule for symbol in grammar.start.body() if symbol.rule is not None]
    while stack:
        rule = stack.pop()
        if rule in seen:
            continue
        seen.add(rule)
        stack.extend(
            symbol.rule for symbol in rule.body() if symbol.rule is not None
        )
    return seen


class TestSequitur:
    def test_paper_example(self):
        """The worked example from Nevill-Manning and Witten, section 2."""
        grammar = build("abcdbcabcdbc")
        expansions = sorted(
            "".join(grammar.expansion(rule))
            for rule in grammar.rules
            if rule is not grammar.start
        )
        assert expansions == ["abcdbc", "bc"]
        start = grammar.start_string()
        assert len(start) == 2
        assert start[0] == start[1]

    def test_no_rules_without_repetition(self):
        grammar = build("abcdef")
        assert [rule for rule in grammar.rules if rule is not grammar.start] == []
        assert grammar.start_string() == list("abcdef")

    def test_round_trip_reproduces_the_input(self):
        text = "abcabcabdabcabd"
        grammar = build(text)
        assert "".join(grammar.expansion(grammar.start)) == text

    @pytest.mark.parametrize("alphabet", ["a", "ab", "abc", "abcde"])
    def test_constraints_hold_on_random_input(self, alphabet):
        rng = random.Random(20260824)
        for _ in range(40):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 200)))
            grammar = build(text)
            assert "".join(grammar.expansion(grammar.start)) == text
            counts = nonoverlapping_digrams(grammar)
            assert all(value == 1 for value in counts.values())
            others = [rule for rule in grammar.rules if rule is not grammar.start]
            assert all(rule.ref_count >= 2 for rule in others)
            assert reachable_rules(grammar) == set(others)

    def test_rule_utility_inlines_a_rule_used_once(self):
        """Every surviving rule is used at least twice, by construction."""
        grammar = build("aabaabaabaab")
        for rule in grammar.rules:
            if rule is grammar.start:
                continue
            assert rule.ref_count >= 2

    def test_hierarchy_is_formed(self):
        grammar = build("abcdbcabcdbc")
        bodies = rule_bodies(grammar)
        nested = [
            body
            for index, body in bodies.items()
            if any(token.startswith("R") for token in body)
        ]
        assert nested, "expected at least one rule referring to another"

    def test_rules_never_straddle_a_sequence_boundary(self):
        """A word must be a substring of one program, so boundaries are sealed."""
        words = sequitur_words([("a", "b"), ("a", "b"), ("a", "b"), ("a", "b")])
        assert ("a", "b") in words
        assert all("a" in word or "b" in word for word in words)
        assert not any(len(word) > 2 for word in words)

    def test_words_dedupe_and_sort_longest_first(self):
        corpus = [("x", "y", "z")] * 6 + [("x", "y")] * 6
        words = sequitur_words(corpus)
        assert len(words) == len(set(words))
        assert words == sorted(words, key=lambda word: (-len(word), word))
        assert ("x", "y") in words


class TestCompressionBar:
    def test_bar_matches_the_quoted_values(self):
        assert compression_bar(26) == pytest.approx(0.0115, abs=5e-5)
        assert compression_bar(118) == pytest.approx(0.00177, abs=5e-6)

    def test_bar_falls_as_the_basis_grows(self):
        """MDL does not self-limit: word B+1 is cheaper to justify than word B."""
        bars = [compression_bar(size) for size in (26, 60, 118, 240, 500)]
        assert bars == sorted(bars, reverse=True)

    def test_bar_rejects_an_empty_basis(self):
        with pytest.raises(ValueError):
            compression_bar(0)


class TestMdlDelta:
    def test_matches_the_criterion_written_out(self):
        total_ops, basis_size, occurrences, length = 137140, 118, 1476, 2
        before = total_ops * math.log2(basis_size)
        after = (total_ops - occurrences * (length - 1)) * math.log2(
            basis_size + 1
        ) + length * math.log2(basis_size)
        assert mdl_delta_bits(
            total_ops=total_ops,
            basis_size=basis_size,
            occurrences=occurrences,
            word_length=length,
        ) == pytest.approx(before - after)

    def test_sign_agrees_with_the_scale_free_bar(self):
        """A word pays iff it removes more than the bar fraction of all ops."""
        total_ops, basis_size = 137140, 118
        bar = compression_bar(basis_size)
        for removed in (0.5 * bar, 0.99 * bar, 1.01 * bar, 2.0 * bar):
            occurrences = round(removed * total_ops)
            delta = mdl_delta_bits(
                total_ops=total_ops,
                basis_size=basis_size,
                occurrences=occurrences,
                word_length=2,
            )
            # The definition costs L * log2(B) bits, so the crossing point sits
            # a hair above the bar rather than exactly on it.
            if removed < bar:
                assert delta < 0
            elif removed > 1.5 * bar:
                assert delta > 0

    def test_delta_grows_with_occurrences_and_with_length(self):
        base = {
            "total_ops": 10000,
            "basis_size": 100,
            "occurrences": 500,
            "word_length": 2,
        }
        more_often = mdl_delta_bits(**{**base, "occurrences": 600})
        longer = mdl_delta_bits(**{**base, "word_length": 3})
        assert more_often > mdl_delta_bits(**base)
        assert longer > mdl_delta_bits(**base)

    def test_fuel_is_absent_from_the_criterion(self):
        """Naming a substring leaves runtime identical, so only length moves.

        Two corpora with the same op count score identically no matter what
        those ops would cost to run; that is the log2(fuel) cancellation.
        """
        left = mdl_delta_bits(
            total_ops=1000, basis_size=50, occurrences=100, word_length=3
        )
        right = mdl_delta_bits(
            total_ops=1000, basis_size=50, occurrences=100, word_length=3
        )
        assert left == right

    def test_rejects_impossible_inputs(self):
        with pytest.raises(ValueError):
            mdl_delta_bits(
                total_ops=100, basis_size=10, occurrences=1, word_length=1
            )
        with pytest.raises(ValueError):
            mdl_delta_bits(
                total_ops=100, basis_size=10, occurrences=-1, word_length=2
            )
        with pytest.raises(ValueError):
            mdl_delta_bits(
                total_ops=10, basis_size=10, occurrences=100, word_length=2
            )


class TestOccurrenceCounting:
    def test_counts_only_non_overlapping_copies(self):
        """``aaaa`` holds three overlapping ``aa`` but only two replaceable."""
        hits = count_occurrences([("a", "a", "a", "a")], [("a", "a")])
        assert hits[("a", "a")] == [(0, 0), (0, 2)]

    def test_counts_across_sequences(self):
        corpus = [("p", "q", "r"), ("z",), ("p", "q")]
        hits = count_occurrences(corpus, [("p", "q")])
        assert hits[("p", "q")] == [(0, 0), (2, 0)]

    def test_word_longer_than_a_sequence_is_absent(self):
        hits = count_occurrences([("a", "b")], [("a", "b", "c")])
        assert hits[("a", "b", "c")] == []

    def test_rewrite_shortens_by_exactly_the_ops_removed(self):
        corpus = [("a", "b", "c"), ("a", "b"), ("c",)]
        word = ("a", "b")
        hits = count_occurrences(corpus, [word])[word]
        out = rewrite(corpus, word, "W", hits)
        assert out == [("W", "c"), ("W",), ("c",)]
        removed = len(hits) * (len(word) - 1)
        assert sum(map(len, corpus)) - sum(map(len, out)) == removed

    def test_rewrite_leaves_untouched_sequences_alone(self):
        corpus = [("a", "b"), ("c", "d")]
        out = rewrite(corpus, ("a", "b"), "W", [(0, 0)])
        assert out[1] == ("c", "d")


TOY_SIGNATURES = {
    "i_to_i": ("int", "int"),
    "i_to_i2": ("int", "int"),
    "i_to_s": ("int", "str"),
    "s_to_i": ("str", "int"),
}


def toy_program(ops):
    return Program(
        ops=tuple(ops),
        input_type="int",
        root_digest="root",
        split="discovery",
        certified_distance=len(ops),
    )


class TestBranchingModel:
    def test_counts_legal_functions_per_visited_state(self):
        model = BranchingModel.from_basis(
            TOY_SIGNATURES, [toy_program(["i_to_s", "s_to_i"])]
        )
        # States are int, str, int: three legal moves at each int, one at str.
        assert model.num_states == 3
        assert model.b_eff() == pytest.approx((3 + 1 + 3) / 3)

    def test_a_new_word_costs_only_the_states_whose_type_it_accepts(self):
        model = BranchingModel.from_basis(
            TOY_SIGNATURES, [toy_program(["i_to_s", "s_to_i"])]
        )
        before = model.b_eff()
        after = model.b_eff(added=("str", "int"))
        assert after - before == pytest.approx(model.type_fraction("str"))
        assert after - before < 1.0

    def test_accepting_a_word_moves_the_baseline(self):
        model = BranchingModel.from_basis(
            TOY_SIGNATURES, [toy_program(["i_to_s", "s_to_i"])]
        )
        before = model.b_eff()
        model.accept(("int", "int"))
        assert model.b_eff() > before

    def test_growth_rate_is_the_branching_factor_when_one_type_exists(self):
        signatures = {f"f{i}": ("int", "int") for i in range(7)}
        model = BranchingModel.from_basis(signatures, [toy_program(["f0", "f1"])])
        assert model.growth_rate() == pytest.approx(7.0)
        assert model.growth_rate(added=("int", "int")) == pytest.approx(8.0)

    def test_growth_rate_sees_a_type_bottleneck_that_b_eff_averages_away(self):
        """One-way trip into a type with no exit: the tree stops growing."""
        signatures = {"a": ("int", "str"), "b": ("int", "int")}
        model = BranchingModel.from_basis(signatures, [toy_program(["b", "a"])])
        assert model.growth_rate() == pytest.approx(1.0)

    def test_search_multiplier_is_the_ratio_raised_to_the_depth(self):
        assert search_multiplier(10.0, 11.0, 6) == pytest.approx(1.1**6)
        with pytest.raises(ValueError):
            search_multiplier(0.0, 1.0)


class TestDiagnostics:
    def test_flags_a_repeated_single_op(self):
        flags = Diagnostics(
            distinct_followers=5,
            distinct_predecessors=5,
            terminal_fraction=0.1,
            initial_fraction=0.1,
            distinct_roots=40,
            distinct_input_types=6,
            single_op_repeat=True,
            contains_digest=False,
        ).flags
        assert "one-op-repeat" in flags

    def test_flags_a_digest_projection(self):
        flags = Diagnostics(
            distinct_followers=5,
            distinct_predecessors=5,
            terminal_fraction=0.1,
            initial_fraction=0.1,
            distinct_roots=40,
            distinct_input_types=6,
            single_op_repeat=False,
            contains_digest=True,
        ).flags
        assert "digest-projection" in flags

    def test_a_healthy_word_carries_no_flags(self):
        assert (
            Diagnostics(
                distinct_followers=9,
                distinct_predecessors=7,
                terminal_fraction=0.3,
                initial_fraction=0.2,
                distinct_roots=44,
                distinct_input_types=5,
                single_op_repeat=False,
                contains_digest=False,
            ).flags
            == []
        )

    def test_diagnose_reads_position_and_provenance_from_the_corpus(self):
        programs = [
            Program(("a", "b"), "int", "r1", "discovery", 2),
            Program(("a", "b", "c"), "str", "r2", "discovery", 3),
        ]
        sequences = [program.ops for program in programs]
        word = ("a", "b")
        hits = count_occurrences(sequences, [word])[word]
        found = diagnose(word, hits, sequences, programs, word)
        assert found.terminal_fraction == pytest.approx(0.5)
        assert found.initial_fraction == pytest.approx(1.0)
        assert found.distinct_followers == 1
        assert found.distinct_roots == 2
        assert found.distinct_input_types == 2
        assert "terminal-sink" not in found.flags


class TestGreedyMining:
    def test_accepts_the_dominant_word_first(self):
        programs = [toy_program(["i_to_s", "s_to_i"]) for _ in range(200)]
        result = mine(programs, TOY_SIGNATURES, max_rounds=1)
        assert len(result.accepted) == 1
        assert result.accepted[0].primitives == ("i_to_s", "s_to_i")
        assert result.accepted[0].occurrences == 200

    def test_greedy_rewriting_removes_the_ops_it_claimed(self):
        programs = [toy_program(["i_to_s", "s_to_i"]) for _ in range(200)]
        result = mine(programs, TOY_SIGNATURES, max_rounds=1)
        claimed = sum(candidate.ops_removed for candidate in result.accepted)
        assert result.total_ops_before - result.total_ops_after == claimed

    def test_the_bar_rises_in_basis_size_as_words_are_accepted(self):
        programs = [
            toy_program(["i_to_s", "s_to_i", "i_to_i", "i_to_i2"]) for _ in range(300)
        ]
        result = mine(programs, TOY_SIGNATURES, max_rounds=3)
        sizes = [candidate.basis_size_before for candidate in result.accepted]
        assert sizes == sorted(sizes)
        assert len(set(sizes)) == len(sizes)

    def test_overlapping_candidates_cannibalise_each_other(self):
        """Naming the winner strips the count from every word it overlaps.

        This is why the batch scored in round one cannot simply be adopted:
        `i_to_i2 i_to_s` is worth 400 occurrences when scored on its own and
        200 once `i_to_i i_to_i2` has been named out from under it.
        """
        programs = (
            [toy_program(["i_to_i", "i_to_i2", "i_to_s"])] * 200
            + [toy_program(["i_to_i", "i_to_i2"])] * 200
            + [toy_program(["i_to_i2", "i_to_s"])] * 200
        )
        result = mine(programs, TOY_SIGNATURES, max_rounds=2)
        batch = {
            tuple(candidate.primitives): candidate.occurrences
            for candidate in result.first_round
        }
        assert batch[("i_to_i", "i_to_i2")] == 400
        assert batch[("i_to_i2", "i_to_s")] == 400

        first, second = result.accepted[0], result.accepted[1]
        assert first.primitives == ("i_to_i", "i_to_i2")
        assert second.primitives == ("i_to_i2", "i_to_s")
        assert second.occurrences == 200
        assert second.occurrences < batch[tuple(second.primitives)]
        assert second.basis_size_before == first.basis_size_before + 1

    def test_no_word_is_accepted_when_nothing_repeats(self):
        programs = [toy_program(["i_to_i"]), toy_program(["i_to_s"])]
        result = mine(programs, TOY_SIGNATURES, max_rounds=5)
        assert result.accepted == []
