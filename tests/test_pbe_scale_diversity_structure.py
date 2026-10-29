"""Conservative witness redundancy and saved-record aggregation."""

import json

import pytest

from experiments.pbe_scale import generate
from experiments.pbe_scale.diversity_structure import (
    CATEGORIES,
    redundant_stages,
    summarize,
)


@pytest.mark.parametrize(
    ("stages", "readout", "ordering", "absolute"),
    [
        (["reverse", "negative"], "span", [0], []),
        (["sort", "unique", "center"], "energy", [0], []),
        (["reverse", "neighbor-min"], "span", [], []),
        (["reverse", "running-sum"], "total", [], []),
        (["reverse", "negative"], "weighted-total", [], []),
        (["reverse", "negative"], "values", [], []),
        (["neighbor-min", "reverse", "absolute"], "energy", [1], [2]),
        (["absolute"], "absolute-total", [], [0]),
        (["absolute"], "total", [], []),
    ],
)
def test_conservative_redundancy(stages, readout, ordering, absolute):
    assert redundant_stages(
        stages, readout, family="series-pipeline", start="input"
    ) == {
        "ordering": ordering,
        "absolute": absolute,
    }


def test_saved_records_exclusion_type_aware_constants_and_overlap(tmp_path):
    accepted, exclusions = tmp_path / "accepted.jsonl", tmp_path / "exclusions.json"

    def row(identifier, kind, program, template):
        return {
            "candidate": {
                "id": identifier,
                "family": "series-pipeline" if kind == "list" else "baseline-pipeline",
                "input_kind": kind,
                "program": program,
                "template_id": template,
                "reference": "raise AssertionError('never execute')",
            },
            "task": {"length": 100},
        }

    rows = [
        row("one", "list", "0 sum", "input/reverse/absolute/energy"),
        row("two", "list", "1 sum", "input/reverse/running-sum/total"),
        row(
            "other-kind",
            "list_int",
            "0 sum",
            "subtract-argument/negative/absolute/total",
        ),
        row("excluded", "list", "9 sum", "input/negative/absolute/energy"),
    ]
    accepted.write_text("".join(json.dumps(r) + "\n" for r in rows))
    exclusions.write_text(json.dumps({"excluded_ids": ["excluded"]}))
    result = summarize(accepted, exclusions)
    assert result["kept_rows"] == 3 and result["excluded_rows"] == 1
    assert result["kind_macro_unique"] == 3
    assert result["kind_macro_integer_normalized_unique"] == 2
    assert result["redundant_witness_rows"] == 1
    assert result["redundancy_rows_by_family"]["series-pipeline"] == {
        "ordering": 1,
        "absolute": 1,
        "union": 1,
    }
    assert result["counts"]["redundancy_overlap"] == {"both": 1, "neither": 2}
    assert result["composition_groups"]["pipeline_core"]["unique"] == 3
    assert result["source_lengths"]["pipeline"]["expanded_tokens"]["0.5"] == 100
    assert result["execution_performed"] is False
    assert result["provenance"]["seed"] is None
    assert len(result["provenance"]["source_data_digests"][str(accepted)]) == 64


@pytest.mark.parametrize(
    ("family", "start"),
    [
        ("series-pipeline", "input"),
        ("baseline-pipeline", "subtract-argument"),
        ("paired-series-pipeline", "pair-sum"),
        ("original-comparison", "total-difference"),
        ("original-comparison", "sample-weighted"),
        ("original-comparison", "level-versus-change"),
        ("conditional-summary", "direction-by-total"),
    ],
)
def test_known_pipeline_contexts(family, start):
    assert redundant_stages(
        ["reverse", "absolute"], "energy", family=family, start=start
    ) == {
        "ordering": [0],
        "absolute": [1],
    }


@pytest.mark.parametrize(
    ("family", "start", "readout", "stages"),
    [
        ("unrecognized", "input", "energy", ["reverse", "absolute"]),
        ("series-pipeline", "unknown-wrapper", "energy", ["reverse", "absolute"]),
        ("series-pipeline", "input", "unknown-readout", ["reverse", "absolute"]),
        ("series-pipeline", "input", "energy", ["reverse", "unknown-stage"]),
        (
            "original-comparison",
            "direction-by-total",
            "energy",
            ["reverse", "absolute"],
        ),
        ("conditional-summary", "total-difference", "energy", ["reverse", "absolute"]),
    ],
)
def test_unknown_contexts_are_not_flagged(family, start, readout, stages):
    assert redundant_stages(stages, readout, family=family, start=start) == {
        "ordering": [],
        "absolute": [],
    }


def test_unknown_category_is_not_assumed_multiset_dependent(monkeypatch):
    monkeypatch.setitem(CATEGORIES, "future-stage", "unknown-category")
    assert redundant_stages(
        ["reverse", "future-stage", "absolute"],
        "energy",
        family="series-pipeline",
        start="input",
    ) == {
        "ordering": [],
        "absolute": [],
    }


@pytest.mark.parametrize("mode", [0, 1, 2, 3])
@pytest.mark.parametrize(
    "readout",
    [
        r
        for r in generate.READOUTS
        if r.name
        in {
            "total",
            "count",
            "energy",
            "absolute-total",
            "maximum",
            "minimum",
            "span",
            "mean",
        }
    ],
    ids=lambda r: r.name,
)
def test_actual_scalar_wrappers_preserve_reverse_removal(mode, readout):
    stages = {s.name: s for s in generate.STAGES}
    full = generate._branched((stages["reverse"], stages["negative"]), readout, mode)
    shortened = generate._branched((stages["negative"],), readout, mode)
    functions = []
    for candidate in (full, shortened):
        namespace = {}
        exec(candidate["reference"], namespace)
        functions.append(namespace["f"])
    # Reference-only tests: no stack execution, search, models or eval data.
    for values in ([], [0], [-3, 2, -1, 0], [4, -2, 7], [-4, -4, 1], [0, 0, 0]):
        assert functions[0](values) == functions[1](values)
