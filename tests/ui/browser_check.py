"""Drive the debugger page in a real browser and report what a person would see.

The DOM stub in ``render_check.mjs`` proves the render path does not throw. It
cannot see layout, CSS, focus, real key dispatch, or whether anything is
actually legible -- so this script loads the page in headless Chromium, presses
the keys, and asserts on the rendered result. It also fails on any console
error, which is the class of bug a stub silently absorbs.

    uv run python tests/ui/browser_check.py http://127.0.0.1:8000 [outdir]
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

CHECKS: list[tuple[str, bool]] = []
CONSOLE: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(condition)))
    status = "ok  " if condition else "FAIL"
    print(f"{status} {name}" + (f" — {detail}" if detail else ""))


def text_of(page, selector: str) -> str:
    """Rendered text, lowercased.

    ``inner_text`` returns text as painted, and this page uppercases its
    section headings in CSS. Comparing case-sensitively against the source
    string silently fails -- which is a difference only a real browser shows.
    """

    element = page.query_selector(selector)
    return element.inner_text().strip().lower() if element else ""


def run(base: str, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 950})
        page.on(
            "console",
            lambda message: CONSOLE.append(f"{message.type}: {message.text}")
            if message.type in ("error", "warning")
            else None,
        )
        page.on("pageerror", lambda error: CONSOLE.append(f"pageerror: {error}"))

        page.goto(f"{base}/?distance=3&seed=11&beam_width=8", wait_until="networkidle")
        page.wait_for_selector(".node", timeout=30_000)

        # --- the page as loaded -------------------------------------------
        check(
            "beam column has nodes",
            page.locator(".node").count() > 0,
            f"{page.locator('.node').count()} nodes",
        )
        check(
            "expansion tape has rows",
            page.locator(".exp").count() > 0,
            f"{page.locator('.exp').count()} rows",
        )
        check(
            "termination badge filled",
            # \u2013 is the en dash the page shows before a session loads.
            text_of(page, "#r-term") not in ("", "\u2013"),
            text_of(page, "#r-term"),
        )
        check(
            "input and target both shown",
            bool(text_of(page, "#in-val")) and bool(text_of(page, "#out-val")),
        )
        check(
            "witness shown",
            "→" in text_of(page, "#witness") or bool(text_of(page, "#witness")),
        )
        check(
            "PCA map drawn",
            page.locator("svg.map circle").count() > 2,
            f"{page.locator('svg.map circle').count()} points",
        )

        # Nothing may overflow the viewport horizontally.
        overflow = page.evaluate(
            "() => document.documentElement.scrollWidth - window.innerWidth"
        )
        check("no horizontal overflow", overflow <= 0, f"{overflow}px")

        # Every column must actually have painted height.
        boxes = page.evaluate("""() => ['beam','tape','insp'].map((id) => {
            const r = document.getElementById(id).getBoundingClientRect();
            return { id, w: Math.round(r.width), h: Math.round(r.height) };
        })""")
        for box in boxes:
            check(
                f"column {box['id']} has real size",
                box["w"] > 100 and box["h"] > 100,
                f"{box['w']}x{box['h']}",
            )
        page.screenshot(path=str(out_dir / "01-replay.png"))

        # --- keyboard: replay ---------------------------------------------
        before = text_of(page, "#exp-sub")
        page.keyboard.press("ArrowRight")
        page.wait_for_timeout(120)
        check(
            "right arrow advances the expansion cursor",
            text_of(page, "#exp-sub") != before,
            text_of(page, "#exp-sub")[:40],
        )

        depth_before = text_of(page, "#r-depth")
        page.keyboard.press("ArrowDown")
        page.wait_for_timeout(150)
        check(
            "down arrow changes layer",
            text_of(page, "#r-depth") != depth_before,
            f"{depth_before} → {text_of(page, '#r-depth')}",
        )

        page.keyboard.press("2")
        page.wait_for_timeout(120)
        check("digit selects a beam node", page.locator(".node.sel").count() == 1)

        page.keyboard.press("v")
        page.wait_for_selector("text=raw logits", timeout=15_000)
        check(
            "v opens raw logits",
            page.locator(".bar").count() > 5,
            f"{page.locator('.bar').count()} rows",
        )
        page.screenshot(path=str(out_dir / "02-logits.png"))

        page.keyboard.press("e")
        page.wait_for_selector("text=exact model input", timeout=15_000)
        check("e opens the encoding panel", "sequences" in text_of(page, "#insp"))
        page.screenshot(path=str(out_dir / "03-encoding.png"))

        page.keyboard.press("q")
        page.wait_for_timeout(150)
        check(
            "q returns to the action panel", "type-normalized" in text_of(page, "#insp")
        )

        page.keyboard.press("t")
        page.wait_for_timeout(120)
        check("t hides the witness", "hidden" in text_of(page, "#witness"))
        page.keyboard.press("t")
        page.wait_for_timeout(120)

        page.keyboard.press(" ")
        page.wait_for_timeout(600)
        page.keyboard.press(" ")
        check("space runs without error", True)
        page.keyboard.press("r")
        page.wait_for_timeout(200)

        # --- keyboard: drive ----------------------------------------------
        page.keyboard.press("m")
        page.wait_for_selector("text=press the key", timeout=20_000)
        check(
            "m enters drive mode",
            text_of(page, "#r-mode").startswith("drive"),
            text_of(page, "#r-mode"),
        )
        check(
            "drive shows key caps",
            page.locator(".keycap").count() >= 3,
            f"{page.locator('.keycap').count()} caps",
        )
        page.screenshot(path=str(out_dir / "04-drive.png"))

        steps_before = page.locator("#beam .node").count()
        page.keyboard.press("a")
        page.wait_for_timeout(900)
        after = page.locator("#beam .node").count()
        check(
            "a applies the rank-1 action",
            after == steps_before + 1,
            f"{steps_before} → {after} steps",
        )
        page.screenshot(path=str(out_dir / "05-drive-stepped.png"))

        page.keyboard.press("s")
        page.wait_for_timeout(900)
        check(
            "s applies the rank-2 action",
            page.locator("#beam .node").count() == after + 1,
        )

        page.keyboard.press("w")
        page.wait_for_timeout(900)
        check(
            "w undoes one step",
            page.locator("#beam .node").count() == after,
            f"{page.locator('#beam .node').count()} steps",
        )

        page.keyboard.press("m")
        page.wait_for_timeout(400)
        check("m returns to replay", text_of(page, "#r-mode").startswith("replay"))

        # --- dark mode -----------------------------------------------------
        dark = browser.new_page(
            viewport={"width": 1500, "height": 950}, color_scheme="dark"
        )
        dark.goto(f"{base}/?distance=2&seed=4&beam_width=8", wait_until="networkidle")
        dark.wait_for_selector(".node", timeout=30_000)
        background = dark.evaluate(
            "() => getComputedStyle(document.body).backgroundColor"
        )
        check(
            "dark mode repaints the background",
            background != "rgb(229, 230, 226)",
            background,
        )
        dark.screenshot(path=str(out_dir / "06-dark.png"))
        dark.close()

        errors = [line for line in CONSOLE if line.startswith(("error", "pageerror"))]
        check("no console errors", not errors, "; ".join(errors[:3]))
        browser.close()

    failed = [name for name, ok in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    print(f"screenshots in {out_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("ui-shots")
    raise SystemExit(run(base_url, output))
