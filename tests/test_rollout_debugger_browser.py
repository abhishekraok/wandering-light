"""Drive the debugger page in a real browser.

The node DOM stub in ``test_rollout_debugger_ui.py`` proves the render path
does not throw. It has no CSS, no layout and no paint, so it cannot see a
circle stretched into an ellipse, a control clipped off the edge, a column with
no height, or a heading whose rendered text differs from its source. Those are
the failures this catches.

``playwright`` is a dev extra, so this runs under a plain ``uv run pytest``.
The browser binary is a separate download and these skip themselves without it:

    uv run playwright install chromium
"""

import importlib.util
import socket
import threading
import time

import pytest

from tests.test_rollout_debugger_server import make_context
from wandering_light.rollout_debugger.server import create_app

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
    import uvicorn

    context, record = make_context()
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
        pytest.fail("debugger server did not start")
    try:
        yield f"http://127.0.0.1:{port}", record
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.fixture(scope="module")
def page(served):
    from playwright.sync_api import sync_playwright

    base, record = served
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except Exception as error:  # no browser binary on this machine
            pytest.skip(f"chromium is not available: {error}")
        view = browser.new_page(viewport={"width": 1500, "height": 950})
        errors: list[str] = []
        view.on("pageerror", lambda error: errors.append(str(error)))
        view.on(
            "console",
            lambda message: errors.append(message.text)
            if message.type == "error"
            else None,
        )
        view.goto(f"{base}/?task_id={record.task_id}&beam_width=4")
        view.wait_for_selector(".node", timeout=30_000)
        yield view, errors
        browser.close()


def test_the_page_paints_its_three_columns(page):
    view, _ = page
    boxes = view.evaluate("""() => ['beam','tape','insp'].map((id) => {
        const r = document.getElementById(id).getBoundingClientRect();
        return [Math.round(r.width), Math.round(r.height)];
    })""")
    for width, height in boxes:
        assert width > 100 and height > 100, boxes


def test_nothing_overflows_the_viewport(page):
    view, _ = page
    overflow = view.evaluate(
        "() => document.documentElement.scrollWidth - window.innerWidth"
    )
    assert overflow <= 0, f"page scrolls horizontally by {overflow}px"


def test_keys_move_through_the_trace(page):
    view, _ = page
    view.keyboard.press("r")
    view.wait_for_timeout(150)
    before = view.inner_text("#exp-sub")
    view.keyboard.press("ArrowRight")
    view.wait_for_timeout(150)
    assert view.inner_text("#exp-sub") != before

    depth = view.inner_text("#r-depth")
    view.keyboard.press("ArrowDown")
    view.wait_for_timeout(200)
    assert view.inner_text("#r-depth") != depth


def test_panels_open_and_close(page):
    view, _ = page
    view.keyboard.press("v")
    view.wait_for_selector("text=raw logits", timeout=20_000)
    view.keyboard.press("e")
    view.wait_for_selector("text=exact model input", timeout=20_000)
    view.keyboard.press("q")
    view.wait_for_timeout(200)
    # Rendered text, so lowercase: the headings are uppercased by CSS.
    assert "type-normalized" in view.inner_text("#insp").lower()


def test_drive_mode_applies_and_undoes_a_step(page):
    view, _ = page
    view.keyboard.press("m")
    view.wait_for_selector("text=press the key", timeout=30_000)
    assert view.inner_text("#r-mode").startswith("drive")
    # The search's termination must not be shown over a hand-driven walk.
    assert "solved" not in view.inner_text("#r-term").lower()

    steps = view.locator("#beam .node").count()
    view.keyboard.press("a")
    view.wait_for_timeout(1200)
    assert view.locator("#beam .node").count() == steps + 1
    view.keyboard.press("w")
    view.wait_for_timeout(1200)
    assert view.locator("#beam .node").count() == steps
    view.keyboard.press("m")
    view.wait_for_timeout(300)


def test_no_console_errors_were_raised(page):
    _, errors = page
    assert not errors, errors
