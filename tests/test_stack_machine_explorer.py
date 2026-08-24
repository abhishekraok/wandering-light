"""The explorer is a JS port of the machine. Pin it to the Python core.

`docs/stack_machine_explorer.html` reimplements `wandering_light.stack_machine`
in JavaScript so it can run in a browser. A port that quietly disagrees with the
core is worse than no explorer, so the page carries a fixture generated from the
core and replays it on load. This test regenerates that fixture and asserts the
embedded copy still matches.

If it fails: rerun `uv run python experiments/stack_machine_fixture.py`, re-embed
the output in the `<script id="conformance">` block, and make the JS agree.
"""

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from experiments.stack_machine_fixture import build

PAGE = Path(__file__).resolve().parent.parent / "docs" / "stack_machine_explorer.html"
EMBEDDED = re.compile(
    r'<script id="conformance" type="application/json">(.*?)</script>', re.DOTALL
)


@pytest.fixture(scope="module")
def embedded() -> dict:
    match = EMBEDDED.search(PAGE.read_text(encoding="utf-8"))
    assert match is not None, "explorer page has no conformance fixture"
    return json.loads(match.group(1))


def test_embedded_fixture_matches_the_core(embedded):
    assert embedded == build()


def test_fixture_covers_every_op(embedded):
    """A fixture that never exercises an op cannot catch a port that breaks it."""
    exercised = {
        name for case in embedded["examples"] for name in case["program"].split()
    } | {name for walk in embedded["walks"] for name in walk["program"].split()}
    assert exercised == {op["name"] for op in embedded["ops"]}


def masks(embedded) -> list[str]:
    return [row["mask"] for case in embedded["examples"] for row in case["trace"]] + [
        walk["mask"] for walk in embedded["walks"]
    ]


def test_fixture_pins_every_guard_open_and_closed(embedded):
    """Each consuming op must be seen both offered and refused.

    An op that is legal in every recorded state cannot catch a port whose guard
    is broken open, and one that is never legal cannot catch the reverse.
    """
    rows = masks(embedded)
    pushing = {"literals", "symbols"}
    for index, op in enumerate(embedded["ops"]):
        seen = {row[index] for row in rows}
        if op["group"] in pushing:
            # nothing here sets a fuel limit, so a push is always available
            assert seen == {"1"}, op["name"]
        else:
            assert seen == {"0", "1"}, (op["name"], seen)


def test_fixture_exercises_the_combinators_and_symbols(embedded):
    """The point of this basis is quotations; the fixture must reach them."""
    groups = {op["name"]: op["group"] for op in embedded["ops"]}
    used = {name for case in embedded["examples"] for name in case["program"].split()}
    assert {n for n in used if groups.get(n) == "combinators"} == {
        "map",
        "filter",
        "fold",
        "ifte",
    }
    assert {n for n in used if groups.get(n) == "symbols"}
    assert any(case["inputs"] != [""] for case in embedded["examples"])


MACHINE = re.compile(
    r"/\* ==== PORTABLE MACHINE START ==== \*/(.*?)/\* ==== PORTABLE MACHINE END ==== \*/",
    re.DOTALL,
)


def test_javascript_port_replays_the_fixture():
    """Run the page's own machine under node and require zero disagreements.

    The page shows this same result as a badge in its header, so a browser
    visit reports drift too; this is the version CI can see.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")

    page = PAGE.read_text(encoding="utf-8")
    machine = MACHINE.search(page)
    assert machine is not None, "explorer page is missing the machine markers"
    fixture = EMBEDDED.search(page).group(1)

    with tempfile.TemporaryDirectory() as workdir:
        script = Path(workdir) / "conformance.mjs"
        script.write_text(
            "const TEXT = "
            + json.dumps(fixture)
            + ";\nglobalThis.document = { getElementById: () => ({ textContent: TEXT }) };\n"
            + machine.group(1)
            + "\nconst r = checkConformance();\n"
            "console.log(JSON.stringify(r));\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [node, str(script)], capture_output=True, text=True, check=True
        )

    report = json.loads(result.stdout)
    assert report["fails"] == [], report["fails"]
    assert report["checks"] > 1000


# ---------------------------------------------------------------------------
# Layout invariants
#
# The page hides bars by setting the `hidden` attribute from script. That only
# works if nothing outranks the user-agent rule for it, and the shell only
# holds together if the grid declares a track per child. Both have been broken
# by adding a row before, and neither shows up in a behavioural test.
# ---------------------------------------------------------------------------

STYLE = re.compile(r"<style>(.*?)</style>", re.DOTALL)
SHELL = re.compile(r'<div class="machine">(.*?)\n</div>', re.DOTALL)
TRACKS = re.compile(r"\.machine \{[^}]*grid-template-rows:([^;]+);", re.DOTALL)


@pytest.fixture(scope="module")
def page() -> str:
    return PAGE.read_text(encoding="utf-8")


def test_hidden_actually_hides(page):
    """Components set display:flex, which beats the UA rule for [hidden]."""
    css = STYLE.search(page).group(1)
    assert re.search(r"\[hidden\]\s*\{[^}]*display\s*:\s*none\s*!important", css), (
        "no [hidden] rule strong enough to beat the component display rules"
    )


def test_every_shell_child_has_a_grid_track(page):
    """A child with no declared track lands in an implicit row."""
    css = STYLE.search(page).group(1)
    shell = SHELL.search(page).group(1)
    children = re.findall(r"^  <(\w+)[^>]*>", shell, re.MULTILINE)
    tracks = TRACKS.search(css).group(1).split()
    assert len(children) == len(tracks), (children, tracks)


def test_bars_the_script_toggles_are_hidden_in_the_markup(page):
    """The bars start hidden, so a first paint never flashes an empty strip."""
    shell = SHELL.search(page).group(1)
    for element_id in ("tape-row", "pbe"):
        row = re.search(rf'<div class="[^"]*" id="{element_id}"[^>]*>', shell)
        assert row and " hidden" in row.group(0), element_id
