"""Drive the stack debugger page in headless Chromium and report what a person sees.

uv run python tests/ui/stack_browser_check.py http://127.0.0.1:8010 [outdir]
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

CHECKS: list[tuple[str, bool]] = []
CONSOLE: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(condition)))
    print(
        f"{'ok  ' if condition else 'FAIL'} {name}" + (f" — {detail}" if detail else "")
    )


def text_of(page, selector: str) -> str:
    element = page.query_selector(selector)
    return element.inner_text().strip().lower() if element else ""


def run(base: str, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 950})
        page.on(
            "console",
            lambda m: (
                CONSOLE.append(f"{m.type}: {m.text}")
                if m.type in ("error", "warning")
                else None
            ),
        )
        page.on("pageerror", lambda e: CONSOLE.append(f"pageerror: {e}"))

        page.goto(f"{base}/?task=359", wait_until="networkidle")
        page.wait_for_selector(".node", timeout=120_000)
        check(
            "three columns painted",
            all(
                page.query_selector(s)
                for s in ("#beam .node", "#tape .exp", "#insp .exrow")
            ),
        )
        check(
            "task line names the task",
            "min_pair" in text_of(page, "#taskline"),
            text_of(page, "#taskline"),
        )
        check(
            "alive strip has cells", len(page.query_selector_all("#alive .cell")) == 48
        )
        check(
            "examples drawn as chips",
            len(page.query_selector_all("#examples .chip")) > 0,
        )
        overflow = page.evaluate(
            "document.documentElement.scrollWidth <= window.innerWidth"
        )
        check("no horizontal overflow", overflow)
        page.screenshot(path=str(out_dir / "01-loaded.png"))

        story_before = text_of(page, "#story")
        page.keyboard.press("ArrowRight")
        check(
            "→ moves the cursor",
            text_of(page, "#story") != story_before,
            text_of(page, "#story"),
        )
        page.keyboard.press("ArrowDown")
        page.keyboard.press("ArrowDown")
        check(
            "↓ changes depth",
            "depth 2" in text_of(page, "#story"),
            text_of(page, "#story"),
        )
        page.keyboard.press("3")
        check(
            "3 selects node 3",
            "node 3" in text_of(page, "#story"),
            text_of(page, "#story"),
        )
        check(
            "witness node marked",
            page.query_selector("#beam .node.witness") is not None,
        )
        page.screenshot(path=str(out_dir / "02-depth2-node3.png"))

        # Ground truth only on the witness node.
        wit = page.evaluate(
            "(() => { const w = S.session.witness_overlay[S.depth].witnesses[0]; return w.node; })()"
        )
        page.keyboard.press(str(wit + 1) if wit is not None and wit < 9 else "1")
        check(
            "witness line shown on the witness node",
            "witness next" in text_of(page, "#insp") or wit is None or wit >= 9,
            text_of(page, "#insp")[:200],
        )
        page.keyboard.press("t")
        check("t hides witness marks", "witness next" not in text_of(page, "#insp"))
        page.keyboard.press("t")

        page.keyboard.press("v")
        page.wait_for_function(
            "document.querySelector('#insp').innerText.toLowerCase().includes('raw logits') && !document.querySelector('#insp .spin')",
            timeout=20_000,
        )
        check("v shows raw logits", "raw logits" in text_of(page, "#insp"))
        page.keyboard.press("v")
        page.keyboard.press("e")
        page.wait_for_function(
            "document.querySelector('#insp').innerText.toLowerCase().includes('what the model sees')",
            timeout=20_000,
        )
        check("e shows tokens", len(page.query_selector_all("#insp .tokens .t")) > 0)
        page.screenshot(path=str(out_dir / "03-tokens.png"))
        page.keyboard.press("e")

        page.keyboard.press("?")
        check("help opens", not page.evaluate("document.getElementById('help').hidden"))
        page.keyboard.press("Escape")
        check("help closes", page.evaluate("document.getElementById('help').hidden"))

        # drive
        page.keyboard.press("m")
        page.wait_for_selector("#tape .exp .keycap", timeout=20_000)
        check(
            "drive mode shows keycaps",
            len(page.query_selector_all("#tape .keycap")) >= 5,
        )
        before = page.evaluate("S.driveHistory.length")
        page.keyboard.press("a")
        page.wait_for_function(
            f"S.driveHistory.length === {before + 1}", timeout=20_000
        )
        check(
            "a applies the top-ranked op",
            page.evaluate("S.driveHistory.length") == before + 1,
        )
        page.keyboard.press("ArrowRight")
        page.keyboard.press("Enter")
        page.wait_for_function(
            f"S.driveHistory.length === {before + 2} || (S.drive && S.drive.ops && S.drive.ops[S.cursor].outcome === 'refused')",
            timeout=20_000,
        )
        page.keyboard.press("w")
        page.wait_for_function(f"S.driveHistory.length <= {before + 1}", timeout=20_000)
        check("w undoes", page.evaluate("S.driveHistory.length") <= before + 1)
        check("progress map drawn", page.query_selector("#insp svg.map") is not None)
        page.screenshot(path=str(out_dir / "04-drive.png"))
        page.keyboard.press("m")
        check("m returns to replay", page.evaluate("S.mode") == "replay")

        # play
        page.keyboard.press("p")
        page.wait_for_selector("#beam .step", timeout=20_000)
        check(
            "play lists the witness ops",
            len(page.query_selector_all("#beam .step")) == 13,
        )
        for _ in range(12):
            page.keyboard.press("ArrowRight")
        check("play reaches the last step", page.evaluate("S.playStep") == 12)
        check(
            "ifte shows inner frames",
            len(page.query_selector_all("#tape .frame")) > 3,
            text_of(page, "#exp-sub"),
        )
        check(
            "every example matches at the end",
            "every example matches" in text_of(page, "#story"),
        )
        page.screenshot(path=str(out_dir / "05-play-ifte.png"))
        page.keyboard.press("o")
        page.wait_for_function(
            "S.play && S.play.which === 'found' && S.play.steps", timeout=20_000
        )
        check(
            "o plays the found program",
            len(page.query_selector_all("#beam .step")) == 12,
        )
        page.keyboard.press("m")

        # run
        page.keyboard.press(" ")
        page.wait_for_timeout(600)
        page.keyboard.press(" ")
        check(
            "space runs the tape",
            page.evaluate("S.depth > 0 || S.node > 0 || S.cursor > 3"),
        )
        page.keyboard.press("r")
        check("r resets", page.evaluate("S.depth") == 0)

        browser.close()

    for line in CONSOLE:
        print("console:", line)
    check("no console errors", not CONSOLE)
    failed = [name for name, ok in CHECKS if not ok]
    print(
        f"\n{len(CHECKS) - len(failed)} of {len(CHECKS)} checks passed; screenshots in {out_dir}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8010"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("stack_shots")
    raise SystemExit(run(base.rstrip("/"), out))
