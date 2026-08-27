from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from wandering_light.basis_dataset import BasisTaskRecord
from wandering_light.basis_set import load_basis_set
from wandering_light.training import deep_corpus_tokenizer as shared
from wandering_light.typed_list import TypedList

BASIS = load_basis_set("wl-core-v1")


@pytest.fixture(autouse=True)
def _fixed_python_hash_seed_contract(monkeypatch):
    monkeypatch.setattr(
        shared,
        "require_deterministic_python_hash_seed",
        lambda: shared.DETERMINISTIC_PYTHON_HASH_SEED,
    )


def _rows(*, long_distance: int | None = None):
    names = shared.basis_function_names(BASIS)
    rows = []
    for distance in shared.REQUIRED_DISTANCES:
        prompt = f"Input: [{distance}]\nTarget Output: [{distance + 1}]\n\nAnswer:"
        if distance == long_distance:
            # Distinct Unicode scalars resist the tiny test BPE's learned merges
            # while keeping this fixture deterministic.
            prompt += "".join(chr(0x1000 + index) for index in range(4_000))
        rows.append(
            {
                "task_id": f"task-{distance}",
                "split": "discovery",
                "certified_distance": distance,
                "prompt": prompt,
                "completion": names[distance - 1],
            }
        )
    return tuple(rows)


def _source(rows):
    task_ids = [row["task_id"] for row in rows]
    return {
        "corpus": "deep-corpus-v2",
        "manifest_digest": "sha256:" + "1" * 64,
        "manifest_file_sha256": "sha256:" + "2" * 64,
        "basis_set_id": BASIS.basis_set_id,
        "basis_set_digest": BASIS.digest,
        "generator": "trajectory-graph-forward-expansion-v1",
        "generator_seed": 20260826,
        "hub": {
            "repo_id": shared.EXPECTED_CORPUS_REPO_ID,
            "repo_type": "dataset",
            "revision": "a" * 40,
        },
        "verification": {
            "records": len(rows),
            "distance_histogram": {str(i): 1 for i in range(1, 9)},
            "all_witnesses_reexecuted": True,
            "root_split_leaks": [],
            "ok": True,
        },
        "discovery": {
            "split": "discovery",
            "records": len(rows),
            "roots": len(rows),
            "by_certified_distance": {str(i): 1 for i in range(1, 9)},
            "split_file": {
                "name": "discovery.jsonl.gz",
                "bytes": 1,
                "sha256": "sha256:" + "3" * 64,
            },
            "ordered_task_ids_digest": shared._canonical_digest(task_ids),
            "task_id_set_digest": shared._canonical_digest(sorted(task_ids)),
        },
    }


def _freeze(tmp_path, name="tokenizer", *, rows=None):
    rows = rows or _rows()
    return shared.freeze_shared_tokenizer(
        tmp_path / name,
        rows=rows,
        basis=BASIS,
        source_identity=_source(rows),
        config=shared.SharedTokenizerConfig(vocab_size=512, min_frequency=1),
    )


def test_canonical_rows_use_pr53_prompt_helper_and_reject_other_splits():
    increment = next(function for function in BASIS.functions if function.name == "inc")
    input_value = TypedList([1, 2], int)
    record = BasisTaskRecord.create(
        split="discovery",
        input_value=input_value,
        output_value=TypedList([2, 3], int),
        witness_function_ids=[increment.function_id],
        witness_function_names=[increment.name],
        basis_set_id=BASIS.basis_set_id,
        basis_set_digest=BASIS.digest,
        generator="test",
        seed=1,
        source_index=0,
        metadata={
            "certified_distance": 1,
            "root_digest": shared._sha256_bytes(input_value.to_string().encode()),
            "root_index": 0,
        },
    )

    rows = shared.canonical_discovery_rows(
        [record],
        BASIS.as_function_set(),
    )
    assert rows[0]["prompt"].endswith("\n\nAnswer:")
    assert rows[0]["completion"] == "inc"

    leaked = BasisTaskRecord.from_dict({**record.to_dict(), "split": "validation"})
    validation_rows = shared.canonical_corpus_rows(
        [leaked],
        BASIS.as_function_set(),
    )
    assert validation_rows[0]["split"] == "validation"
    assert validation_rows[0]["task_id"] == leaked.task_id

    with pytest.raises(ValueError, match="accepts only"):
        shared.canonical_discovery_rows(
            [leaked],
            BASIS.as_function_set(),
            row_builder=lambda *_: {},
        )


def test_tokenizer_has_fixed_specials_atomic_normal_basis_names_and_byte_roundtrip():
    tokenizer = shared.train_shared_tokenizer(
        _rows(),
        BASIS,
        config=shared.SharedTokenizerConfig(vocab_size=512, min_frequency=1),
    )

    for expected_id, (role, content) in enumerate(shared.FIXED_SPECIAL_TOKENS):
        assert getattr(tokenizer, role) == content
        assert tokenizer.convert_tokens_to_ids(content) == expected_id
    added = tokenizer.backend_tokenizer.get_added_tokens_decoder()
    for name in shared.basis_function_names(BASIS):
        token_id = tokenizer.convert_tokens_to_ids(name)
        assert tokenizer.encode(name, add_special_tokens=False) == [token_id]
        assert added[token_id].special is False
        assert tokenizer.decode([token_id], skip_special_tokens=True) == name

    text = "NUL \x00, café, 東京, and Ω"
    assert (
        tokenizer.decode(
            tokenizer.encode(text, add_special_tokens=False),
            skip_special_tokens=False,
        )
        == text
    )
    raw = bytes(range(256))
    assert shared.decode_bytes(tokenizer, shared.encode_bytes(tokenizer, raw)) == raw
    payload = json.loads(tokenizer.backend_tokenizer.to_str())
    assert payload["model"]["byte_fallback"] is True
    assert payload["post_processor"]["type"] == "TemplateProcessing"


def test_trl_prompt_completion_tokenization_matches_manual_ids_mask_and_validation_audit():
    tokenizer = shared.train_shared_tokenizer(
        _rows(),
        BASIS,
        config=shared.SharedTokenizerConfig(vocab_size=512, min_frequency=1),
    )
    row = _rows()[3]
    encoded = shared.encode_training_row(tokenizer, row)

    # This is TRL's prompt-completion path: it adds EOS to the completion, then
    # tokenizes both the prompt and the concatenated example with specials on.
    prompt = tokenizer(text=row["prompt"], add_special_tokens=True)["input_ids"]
    combined = tokenizer(
        text=row["prompt"] + row["completion"] + tokenizer.eos_token,
        add_special_tokens=True,
    )["input_ids"]
    completion_mask = [0] * len(prompt) + [1] * (len(combined) - len(prompt))

    assert combined[: len(prompt)] == prompt
    assert combined == encoded["combined_ids"]
    assert completion_mask == encoded["completion_mask"]
    assert [
        token_id
        for token_id, selected in zip(combined, completion_mask, strict=True)
        if selected
    ] == encoded["completion_with_eos_ids"]
    assert combined[0] == tokenizer.bos_token_id
    assert combined[-1] == tokenizer.eos_token_id

    validation_rows = tuple({**item, "split": "validation"} for item in _rows())
    audit = shared.build_tokenization_audit(tokenizer, validation_rows)
    assert audit["retained_records"] == len(validation_rows)
    assert {item["split"] for item in audit["retained"]} == {"validation"}
    with pytest.raises(ValueError, match="accepts only"):
        shared.train_shared_tokenizer(validation_rows, BASIS)


def test_frozen_artifact_is_deterministic_and_validates_reload_identity(tmp_path):
    first = _freeze(tmp_path, "first")
    second = _freeze(tmp_path, "second")

    assert first["artifact_digest"] == second["artifact_digest"]
    assert first["tokenizer"]["files"] == second["tokenizer"]["files"]
    assert first["training_text"]["discovery_only"] is True
    assert first["training_text"]["pythonhashseed"] == "0"
    assert first["tokenizer"]["implementation"]["pythonhashseed"] == "0"
    assert first["training_text"]["records"] == 8
    reloaded = shared.validate_frozen_tokenizer(
        tmp_path / "first",
        expected_source_manifest_digest="sha256:" + "1" * 64,
    )
    assert reloaded == first


def test_selection_lists_statistics_and_token_id_digests_are_complete(tmp_path):
    rows = _rows(long_distance=8)
    manifest = _freeze(tmp_path, rows=rows)
    selection = manifest["selection"]

    assert selection["selected_records"] == 8
    assert selection["retained_records"] == 7
    assert selection["excluded_records"] == 1
    assert "retained" not in selection
    assert selection["retained_rows"]["storage"].startswith("omitted-from-artifact")
    assert "excluded" not in selection
    detail_identity = selection["detail_file"]
    detail_path = tmp_path / "tokenizer" / shared.DETAIL_FILENAME
    assert detail_identity["sha256"] == shared._sha256_file(detail_path)
    detail = json.loads(detail_path.read_text(encoding="utf-8"))["selection"]
    assert detail["excluded"] == [
        {
            "task_id": "task-8",
            "split": "discovery",
            "certified_distance": 8,
            "prompt_tokens": detail["excluded"][0]["prompt_tokens"],
            "completion_tokens": detail["excluded"][0]["completion_tokens"],
            "completion_tokens_with_eos": detail["excluded"][0][
                "completion_tokens_with_eos"
            ],
            "combined_tokens": detail["excluded"][0]["combined_tokens"],
            "reason": "combined_tokens_exceed_fixed_max_length",
            "max_length": shared.FIXED_MAX_LENGTH,
        }
    ]
    assert detail["excluded"][0]["combined_tokens"] > shared.FIXED_MAX_LENGTH
    selected_stats = selection["statistics"]["selected"]
    assert selected_stats["overall"]["combined"]["count"] == 8
    assert selected_stats["by_certified_distance"]["8"]["combined"]["count"] == 1
    assert "histogram" not in selected_stats["by_certified_distance"]["8"]["combined"]
    assert (
        detail["statistics"]["retained"]["by_certified_distance"]["8"]["combined"][
            "count"
        ]
        == 0
    )
    assert detail["statistics"]["selected"]["by_certified_distance"]["8"]["combined"][
        "histogram"
    ]
    for scope in ("selected", "retained"):
        for field in ("prompt", "completion", "completion_with_eos", "combined"):
            assert selection["token_id_digests"][scope][field].startswith("sha256:")


def test_validation_rejects_tampering_and_wrong_source(tmp_path):
    rows = _rows()
    wrong_source = _source(rows)
    wrong_source["hub"] = {**wrong_source["hub"], "repo_id": "acct/wrong"}
    with pytest.raises(ValueError, match="wrong V2 Hub"):
        shared.freeze_shared_tokenizer(
            tmp_path / "wrong",
            rows=rows,
            basis=BASIS,
            source_identity=wrong_source,
            config=shared.SharedTokenizerConfig(vocab_size=512, min_frequency=1),
        )

    _freeze(tmp_path, "valid")
    tokenizer_path = tmp_path / "valid" / "tokenizer.json"
    tokenizer_path.write_text(tokenizer_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file digest mismatch"):
        shared.validate_frozen_tokenizer(tmp_path / "valid")


def test_validation_rejects_detached_detail_tampering(tmp_path):
    _freeze(tmp_path, "valid")
    detail = tmp_path / "valid" / shared.DETAIL_FILENAME
    detail.write_text(detail.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="detail-file identity drifted"):
        shared.validate_frozen_tokenizer(tmp_path / "valid")


def test_record_hub_location_is_outside_tokenizer_identity(tmp_path):
    manifest = _freeze(tmp_path, "valid")

    shared.record_hub_location(
        tmp_path / "valid", repo_id="acct/dataset", revision="a" * 40
    )

    recorded = shared.validate_frozen_tokenizer(tmp_path / "valid")
    assert recorded["artifact_digest"] == manifest["artifact_digest"]
    assert recorded["hub"]["revision"] == "a" * 40


def test_dry_run_summary_has_no_tokenizer_side_effects(tmp_path):
    rows = _rows()
    source = shared.VerifiedDiscoveryCorpus(
        manifest={},
        basis=BASIS,
        records=(),
        rows=rows,
        source_identity=_source(rows),
    )
    summary = shared.dry_run_summary(source)

    assert summary["status"] == "dry-run"
    assert summary["fixed_max_length"] == 2048
    assert len(summary["preview"]) == 8
    assert not list(tmp_path.iterdir())


def test_loader_requests_discovery_only_after_complete_verification(
    tmp_path, monkeypatch
):
    corpus_dir = tmp_path / "deep_corpus_v2"
    corpus_dir.mkdir()
    manifest = {
        "basis_set_id": BASIS.basis_set_id,
        "basis_set_digest": BASIS.digest,
        "manifest_digest": "sha256:" + "1" * 64,
        "generator": "trajectory-graph-forward-expansion-v1",
        "seed": 9,
        "global_task_count": 8,
        "hub": {
            "repo_id": shared.EXPECTED_CORPUS_REPO_ID,
            "repo_type": "dataset",
            "revision": "a" * 40,
        },
        "splits": {
            split: {
                "path": f"{split}.jsonl.gz",
                "size": 8 if split == "discovery" else 1,
                "roots": 8 if split == "discovery" else 1,
                "by_certified_distance": {str(i): 1 for i in range(1, 9)},
            }
            for split in ("discovery", "validation", "test")
        },
    }
    (corpus_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (corpus_dir / "discovery.jsonl.gz").write_bytes(b"payload")
    records = tuple(
        SimpleNamespace(task_id=row["task_id"], split="discovery") for row in _rows()
    )
    calls = []

    from experiments import generate_deep_corpus as deep

    reference_identity = object()
    reference_dir = tmp_path / "deep_corpus_v1"
    monkeypatch.setattr(
        deep,
        "collect_corpus_identities",
        lambda path: calls.append(("reference", path)) or reference_identity,
    )

    def fake_verify(*args, **kwargs):
        calls.append(("verify", args, kwargs))
        assert kwargs["reference_identities"] == (reference_identity,)
        return {
            "ok": True,
            "records": 8,
            "witness_failures": [],
            "roots_leaking_across_splits": [],
            "distance_histogram": {str(i): 1 for i in range(1, 9)},
        }

    monkeypatch.setattr(deep, "verify_corpus", fake_verify)

    def fake_load(*args, **kwargs):
        calls.append(("load", args, kwargs))
        assert kwargs["splits"] == ("discovery",)
        return manifest, list(records)

    monkeypatch.setattr(deep, "load_corpus", fake_load)
    monkeypatch.setattr(
        shared,
        "canonical_discovery_rows",
        lambda items, functions: _rows(),
    )

    loaded = shared.load_verified_v2_discovery(
        corpus_dir,
        reference_corpus_dirs=(reference_dir,),
    )

    assert [call[0] for call in calls] == ["reference", "verify", "load"]
    assert loaded.rows == _rows()
