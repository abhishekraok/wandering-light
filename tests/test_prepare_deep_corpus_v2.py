import json
from collections import Counter
from pathlib import Path

from experiments import generate_deep_corpus as deep
from experiments import prepare_deep_corpus_v2 as prepare

MAX_TRACKED_MANIFEST_BYTES = 64 * 1024


def test_frozen_profile_matches_issue_54_contract():
    options = prepare.frozen_generation_options(progress=False)

    assert options["basis_set_id"] == "wl-core-v1"
    assert options["seed"] == 20_260_826
    assert options["roots"] == 2_400
    assert options["split_counts"] == {
        "discovery": 1_880,
        "validation": 260,
        "test": 260,
    }
    assert options["deep_roots"] == 360
    assert options["tasks_per_distance"] == 20
    assert options["max_states"] == 600_000
    assert options["max_transitions"] == 2_500_000
    assert options["deep_max_states"] == 3_000_000
    assert options["deep_max_transitions"] == 20_000_000
    assert options["resume_dir"] == (
        prepare.OUTPUT_DIR.parent / f".{prepare.OUTPUT_DIR.name}.resume"
    )
    assert options["target_contract"]["distance_targets"] == {
        "1": 18_800,
        "2": 47_500,
        "3": 48_000,
        "4": 48_000,
        "5": 48_000,
        "6": 48_000,
        "7": 48_000,
        "8": 7_200,
    }
    assert "require_all_expansions_complete" not in options["target_contract"]
    assert prepare.HUB_REPO_ID == "abhishekraok/wandering-light-deep-corpus-v2"


def test_tracked_v2_manifests_carry_identity_not_record_detail():
    """Keep the new policy enforceable; V1 is a documented history exception."""

    corpus_path = Path("wandering_light/training/data/deep_corpus_v2/manifest.json")
    tokenizer_path = Path(
        "wandering_light/training/tokenizers/deep_corpus_v2/tokenizer_manifest.json"
    )
    assert corpus_path.stat().st_size <= MAX_TRACKED_MANIFEST_BYTES
    assert tokenizer_path.stat().st_size <= MAX_TRACKED_MANIFEST_BYTES

    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    assert "roots" not in corpus["expansion"]
    assert corpus["detail_files"]["expansion"]["sha256"].startswith("sha256:")
    assert "excluded" not in tokenizer["selection"]
    assert tokenizer["selection"]["detail_file"]["sha256"].startswith("sha256:")
    assert not [
        value
        for value in _nested_values(tokenizer["selection"]["statistics"])
        if isinstance(value, dict) and "histogram" in value
    ]


def _nested_values(value):
    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from _nested_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _nested_values(item)


def test_run_removes_resume_only_after_successful_verification(tmp_path, monkeypatch):
    events = []
    output_dir = tmp_path / "corpus"
    reference = object()

    monkeypatch.setattr(
        prepare, "_reference_identities", lambda *args, **kwargs: reference
    )
    monkeypatch.setattr(prepare, "require_clean_generator_source", lambda: "a" * 40)

    def generate(**options):
        events.append(("generate", options["resume_dir"]))
        return {
            "basis_set_digest": prepare.BASIS_SET_DIGEST
        }, output_dir / "manifest.json"

    monkeypatch.setattr(prepare.deep, "generate_corpus", generate)
    monkeypatch.setattr(
        prepare,
        "_verify",
        lambda *args, **kwargs: events.append(("verify", None)) or {"ok": True},
    )
    monkeypatch.setattr(
        prepare.deep,
        "remove_resume_directory",
        lambda path: events.append(("cleanup", path)) or True,
    )

    assert prepare.main(["--mode", "run", "--output-dir", str(output_dir)]) == 0
    expected_resume = prepare.default_resume_dir(output_dir)
    assert events == [
        ("generate", expected_resume),
        ("verify", None),
        ("cleanup", expected_resume),
    ]


def test_generated_manifest_can_pin_the_clean_generator_commit():
    commit = "a" * 40
    options = prepare.frozen_generation_options(
        progress=False,
        generator_git_commit=commit,
    )

    assert options["target_contract"]["generator_source"] == {
        "git_commit": commit,
        "tree_state": "clean",
        "runner": "experiments/prepare_deep_corpus_v2.py",
        "generator": "experiments/generate_deep_corpus.py",
    }


def test_frozen_profile_draws_exact_split_counts():
    options = prepare.frozen_generation_options(progress=False)
    plans = deep.build_root_plans(
        roots=options["roots"],
        seed=options["seed"],
        max_depth=options["max_depth"],
        deep_roots=options["deep_roots"],
        deep_max_depth=options["deep_max_depth"],
        max_states=options["max_states"],
        max_transitions=options["max_transitions"],
        deep_max_states=options["deep_max_states"],
        deep_max_transitions=options["deep_max_transitions"],
        deep_root_types=options["deep_root_types"],
        split_counts=options["split_counts"],
    )

    assert Counter(plan.split for plan in plans) == prepare.SPLIT_COUNTS
    assert set(Counter(plan.input_type for plan in plans).values()) == {200}
    assert sum(plan.deep for plan in plans) == 360
