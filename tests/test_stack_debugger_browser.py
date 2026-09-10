"""Drive the stack debugger page in headless Chromium (stack-rollout-debugger, #115).

Against the tiny untrained policy from the server tests, so no checkpoint or
Hub access is needed. ``playwright`` is a dev extra; the browser binary is a
separate download and the tests skip without it:

    uv run playwright install chromium
"""

import importlib.util
import socket
import threading
import time

import pytest

from tests.test_stack_debugger_server import task_for
from wandering_light.rollout_policy import PolicyArchitecture
from wandering_light.stack_debugger.context import DebugContext
from wandering_light.stack_debugger.server import create_app
from wandering_light.stack_policy import PolicyScorer, StackPairPolicy

playwright_missing = importlib.util.find_spec("playwright") is None
pytestmark = pytest.mark.skipif(
    playwright_missing, reason="playwright is not installed"
)


def free_port() -> int:
    with socket.socket() as handle:
        handle.bind(("127.0.0.1", 0))
        return handle.getsockname()[1]


@pytest.fixture(scope="module")
def served():
    import torch
    import uvicorn

    torch.manual_seed(0)
    context = DebugContext(
        tasks=[task_for("min_pair", 2), task_for("sum")],
        scorer=PolicyScorer(StackPairPolicy(PolicyArchitecture(8, 16, 32)), "cpu"),
        provenance={"policy": {"source": "test", "arm": "tiny", "seed": 0}},
    )
    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(context), host="127.0.0.1", port=port, log_level="error"
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:
        pytest.fail("stack debugger server did not start")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.fixture(scope="module")
def page(served):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except Exception as error:  # no browser binary on this machine
            pytest.skip(f"chromium is not available: {error}")
        view = browser.new_page(viewport={"width": 1500, "height": 950})
        errors: list[str] = []
        view.on("pageerror", lambda error: errors.append(str(error)))
        view.on(
            "console", lambda m: errors.append(m.text) if m.type == "error" else None
        )
        view.goto(f"{served}/?task=0&beam_width=8&max_depth=6")
        view.wait_for_selector(".node", timeout=30_000)
        yield view, errors
        browser.close()


def text(page, selector: str) -> str:
    return page.inner_text(selector).strip().lower()


def test_the_page_paints_its_three_columns(page):
    view, _ = page
    assert view.query_selector("#beam .node") is not None
    assert len(view.query_selector_all("#tape .exp")) == 28
    assert len(view.query_selector_all("#insp .exrow")) == 3
    assert len(view.query_selector_all("#alive .cell")) == 6
    assert view.evaluate("document.documentElement.scrollWidth <= window.innerWidth")


def test_keys_move_through_the_trace(page):
    view, _ = page
    view.keyboard.press("r")
    assert "depth 0" in text(view, "#story")
    view.keyboard.press("ArrowDown")
    assert "depth 1" in text(view, "#story")
    view.keyboard.press("2")
    assert "node 2" in text(view, "#story")
    before = view.evaluate("S.cursor")
    view.keyboard.press("ArrowRight")
    assert view.evaluate("S.cursor") == before + 1
    view.keyboard.press("r")


def test_down_follows_the_selected_op_and_up_returns_to_it(page):
    view, _ = page
    view.keyboard.press("r")
    # Put the cursor on an op whose result was kept, then follow it.
    view.evaluate(
        "(() => { const l = layer(); const row = l.expansions[0];"
        " S.cursor = row.findIndex((e) => e && e.verdict === 'retained'); render(); })()"
    )
    op = view.evaluate("ops()[S.cursor]")
    view.keyboard.press("ArrowDown")
    assert view.evaluate("S.depth") == 1
    assert view.evaluate("layer().nodes[S.node].program") == [op]
    view.keyboard.press("ArrowUp")
    assert view.evaluate("S.depth") == 0
    assert view.evaluate("ops()[S.cursor]") == op
    view.keyboard.press("r")


def test_ground_truth_is_marked_only_on_the_witness_node(page):
    view, _ = page
    view.keyboard.press("r")
    view.keyboard.press("ArrowDown")
    witness = view.evaluate("S.session.witness_overlay[S.depth].witnesses[0].node")
    if witness is None:
        pytest.skip("the witness left the beam at depth 1 under the untrained policy")
    view.keyboard.press(str(witness + 1))
    assert "witness next" in text(view, "#insp")
    view.keyboard.press(str(witness + 2 if witness == 0 else witness))
    assert "witness next" not in text(view, "#insp")
    view.keyboard.press("t")
    view.keyboard.press(str(witness + 1))
    assert "witness next" not in text(view, "#insp")
    view.keyboard.press("t")
    view.keyboard.press("r")


def test_panels_open_and_close(page):
    view, _ = page
    view.keyboard.press("v")
    view.wait_for_function(
        "document.querySelector('#insp').innerText.toLowerCase().includes('raw logits')"
    )
    view.keyboard.press("v")
    assert "raw logits" not in text(view, "#insp")
    view.keyboard.press("e")
    view.wait_for_function("document.querySelectorAll('#insp .tokens .t').length > 0")
    view.keyboard.press("e")
    view.keyboard.press("?")
    assert not view.evaluate("document.getElementById('help').hidden")
    view.keyboard.press("Escape")
    assert view.evaluate("document.getElementById('help').hidden")


def test_drive_mode_applies_and_undoes_a_step(page):
    view, _ = page
    view.keyboard.press("r")
    view.keyboard.press("m")
    view.wait_for_selector("#tape .keycap")
    view.keyboard.press("a")
    view.wait_for_function("S.driveHistory.length === 1")
    assert len(view.query_selector_all("#beam .node")) == 2
    view.keyboard.press("w")
    view.wait_for_function("S.driveHistory.length === 0")
    view.keyboard.press("m")
    assert view.evaluate("S.mode") == "replay"


def test_play_mode_steps_the_witness_with_inner_work(page):
    view, _ = page
    view.keyboard.press("p")
    view.wait_for_selector("#beam .step")
    steps = view.evaluate("S.play.steps.length")
    for _ in range(steps - 1):
        view.keyboard.press("ArrowRight")
    assert "every example matches" in text(view, "#story")
    assert len(view.query_selector_all("#tape .frame")) > 0
    view.keyboard.press("m")


def test_no_console_errors_were_raised(page):
    _, errors = page
    assert errors == []
