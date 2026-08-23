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
        name
        for case in embedded["examples"] + embedded["probes"]
        for name in case["program"].split()
    } | {name for walk in embedded["walks"] for name in walk["program"].split()}
    assert exercised == {op["name"] for op in embedded["ops"]}


def test_fixture_pins_the_guards_that_matter(embedded):
    """Each guard must be observably closed somewhere in the recorded traces."""
    legal_sets = [
        set(row["legal"])
        for case in embedded["examples"] + embedded["probes"]
        for row in case["trace"]
    ]
    for name in ("div", "uncons", "cons", "add", "null?", "lt?", "swap"):
        assert any(name not in legal for legal in legal_sets), name
        assert any(name in legal for legal in legal_sets), name


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
