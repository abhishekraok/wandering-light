"""Complete independent verification uses synthetic pins and mock scorers."""

import hashlib
import json
import math
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import capped_glue_run as producer

fixture_helpers = runpy.run_path(
    str(Path(__file__).with_name("test_capped_glue_run.py"))
)
inputs = fixture_helpers["inputs"]
mock_loader = fixture_helpers["mock_loader"]

verifier = SimpleNamespace(
    **runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "reports/capped-glue-loss-20261003/verify.py"
        )
    )
)


def make_result(inputs, *, size=20, positive=False):
    args = inputs(size)
    loader, _, _ = mock_loader(args[2], positive=positive)
    assert producer.run_diagnostic(*args, loader=loader) == 0
    log = args[1] / "trial.log"
    log.write_text("exit 0\n")
    return args[2] / "final.json", log, args[0], args[1], args[4]


@pytest.mark.parametrize(
    ("size", "positive", "expected"),
    [
        (19, False, "INSUFFICIENT_SUPPORT"),
        (20, False, "NO_MATCHED_MIDSPAN_TOP_EXCESS"),
        (20, True, "TOP_LOSS_ROUTE"),
    ],
)
def test_independent_complete_reading_matches_synthetic_producer(
    inputs, size, positive, expected
):
    result = verifier.verify_result(*make_result(inputs, size=size, positive=positive))
    assert result["verified"] and result["models_deserialized"] == 0
    assert result["reading"] == expected
    assert result["reconstruction"]["support_passes"] == (size == 20)


@pytest.mark.parametrize(
    "change",
    ["metrics", "summary", "identity", "missing", "nonfinite", "stage", "extra"],
)
def test_rehashed_corrupt_artifacts_fail_independent_checks(inputs, change):
    args = make_result(inputs)
    final = args[0]
    manifest = json.loads(final.read_text())
    out = manifest["checkpoints"][0]
    if change in {"metrics", "identity", "missing", "nonfinite"}:
        path = Path(out["raw"]["path"])
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if change == "metrics":
            rows[0]["metrics"]["unmasked_nll"] += 1
        elif change == "identity":
            rows[0]["step"] += 1
        elif change == "missing":
            rows.pop()
        else:
            rows[0]["logits"][0] = float("nan")
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        out["raw"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif change == "summary":
        path = Path(out["summary"]["path"])
        summary = json.loads(path.read_text())
        summary["primary"]["delta"] += 1
        path.write_text(json.dumps(summary))
        out["summary"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif change == "stage":
        path = Path(manifest["stage_a"]["path"])
        stage = json.loads(path.read_text())
        stage["rows"][0]["product"][0] = "3"
        path.write_text(json.dumps(stage))
        manifest["stage_a"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        (final.parent / "unlisted-raw.jsonl").write_text("unlisted")
    final.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verifier.verify_result(*args)


@pytest.mark.parametrize("state", ["invalid", "kill", "false_complete"])
def test_unsuccessful_terminal_never_opens_raw_logits(inputs, monkeypatch, state):
    args = make_result(inputs)
    final, log = args[:2]
    manifest = json.loads(final.read_text())
    if state == "invalid":
        manifest["reading"] = "INVALID"
    elif state == "kill":
        log.write_text("")
    else:
        manifest["complete"] = False
    final.write_text(json.dumps(manifest))
    original_open = Path.open

    def guarded(path, *args, **kwargs):
        if path.name.endswith("-raw.jsonl"):
            pytest.fail("unsuccessful terminal exposed partial logits")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises(ValueError):
        verifier.verify_result(*args)


def test_support_only_forbids_model_artifacts_without_reading_them(inputs):
    args = make_result(inputs, size=19)
    (args[0].parent / "illegal-raw.jsonl").write_text("not JSON; must never be parsed")
    with pytest.raises(ValueError, match="unsupported model artifacts"):
        verifier.verify_result(*args)


def test_wrong_source_and_deadline_fail_before_effects(inputs):
    args = make_result(inputs)
    with pytest.raises(ValueError, match="wrong reviewed source"):
        verifier.verify_result(*args[:-1], "other-source")
    with pytest.raises(ValueError, match="deadline"):
        verifier.verify_result(*args, deadline=0)


def test_scalar_normalizers_remain_stable_when_legal_peak_is_far_below_illegal():
    values = [-10000.0] + [10000.0] * 27
    result = verifier.losses(values, [True] + [False] * 27, verifier.NAMES[0])
    assert math.isclose(result["unmasked_nll"], 20000 + math.log(27))
    assert result["legal_nll"] == 0
    assert result["negative_log_legal_mass"] == result["unmasked_nll"]
    assert result["legal_rank"] == result["legal_size"] == 1
