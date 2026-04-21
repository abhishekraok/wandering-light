"""End-to-end AppTest for explorer.py.

Verifies one invariant: every rendered edge selectbox displays the function
name stored in its node's `applied_fn_def`. Guards against widget-state
drift across sample switches, random sampling, and edge edits — the bug
class that motivated sample-specific widget keys.
"""

import json
import os

import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXPLORER_PATH = os.path.join(REPO_ROOT, "wandering_light/evals/explorer.py")
SOLVER_JSON = os.path.join(
    REPO_ROOT, "results/20260420_163811/TrajectorySolver_20260420_163819.json"
)


@pytest.fixture
def repo_cwd(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)


def _ss(at: AppTest, key: str, default=None):
    try:
        return at.session_state[key]
    except (KeyError, AttributeError):
        return default


def _invariant_violations(at: AppTest) -> list[str]:
    widget_values = {sb.key: sb.value for sb in at.selectbox if sb.key}
    issues: list[str] = []
    for tree_key, prefix in [
        ("tree_gold", "gold"),
        ("tree_pred", "pred"),
        ("tree_eval", "eval"),
    ]:
        tree = _ss(at, tree_key)
        if tree is None:
            continue
        for nid, node in tree.nodes.items():
            applied = node.get("applied_fn_def")
            if applied is None:
                continue
            for key, value in widget_values.items():
                if key.startswith(f"{prefix}_") and key.endswith(
                    f"_edge_sel_{nid}"
                ):
                    if value != applied.name:
                        issues.append(
                            f"{key}={value!r} but tree[{nid}].applied="
                            f"{applied.name!r}"
                        )
    return issues


def _pick_sample_with_multiple_gold_edges(json_path: str) -> int:
    with open(json_path) as f:
        details = json.load(f)["detailed_results"]
    return next(
        (
            i
            for i, d in enumerate(details)
            if len(d.get("golden_functions") or []) >= 2
        ),
        0,
    )


def test_widget_state_tracks_applied_fn_through_interactions(repo_cwd):
    if not os.path.exists(SOLVER_JSON):
        pytest.skip(f"Fixture missing: {SOLVER_JSON}")

    multi_edge_idx = _pick_sample_with_multiple_gold_edges(SOLVER_JSON)

    at = AppTest.from_file(EXPLORER_PATH, default_timeout=120)
    at.run()

    # Initial sample selection — enters the solver tab and seeds tree state.
    sb = next(s for s in at.selectbox if s.key == "solver_sample_idx")
    sb.set_value(0).run()
    assert _ss(at, "tree_gold") is not None, "gold tree should be initialised"
    assert _invariant_violations(at) == []

    # 1. Switch samples via the selectbox.
    sb = next(s for s in at.selectbox if s.key == "solver_sample_idx")
    sb.set_value(multi_edge_idx).run()
    assert _invariant_violations(at) == []

    # 2. Click the random sample button.
    btn = next(b for b in at.button if b.key == "solver_random_sample")
    btn.click().run()
    assert _invariant_violations(at) == []

    # 3. Force back to a sample known to have ≥2 gold edges so we can edit.
    sb = next(s for s in at.selectbox if s.key == "solver_sample_idx")
    sb.set_value(multi_edge_idx).run()
    assert _invariant_violations(at) == []

    gold_edges = [
        s
        for s in at.selectbox
        if (s.key or "").startswith("gold_") and "_edge_sel_" in s.key
    ]
    assert gold_edges, "expected at least one gold edge selectbox"

    sb = gold_edges[0]
    other = next((o for o in sb.options if o != sb.value), None)
    if other is None:
        pytest.skip(
            "Edge has no alternative compatible function; cannot exercise edit."
        )

    sb.set_value(other)
    ns_prefix, _, node_suffix = sb.key.partition("_edge_sel_")
    apply_btn = next(
        b for b in at.button if b.key == f"{ns_prefix}_edge_btn_{node_suffix}"
    )
    apply_btn.click().run()
    assert _invariant_violations(at) == []
