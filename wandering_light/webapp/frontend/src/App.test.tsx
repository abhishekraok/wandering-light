import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";

/** State wires are opaque to the UI, so the stub can keep them readable. */
function state(label: string) {
  return { id: label, wire: `wire:${label}`, label, type: "int", size: 3 };
}

/**
 * A stand-in for the API that records what the UI asked for.
 *
 * The point is the request the UI makes after an interaction -- replacing step
 * one must re-run the trajectory as a two-step list, not a three-step one.
 */
function stubApi() {
  const calls: { url: string; body: unknown }[] = [];
  const fetchStub = vi.fn(async (url: string, init?: RequestInit) => {
    const body = init?.body ? JSON.parse(init.body as string) : null;
    calls.push({ url, body });
    const json = (payload: unknown) =>
      ({ ok: true, json: async () => payload }) as unknown as Response;

    if (url.startsWith("/api/basis")) {
      return json({
        available: [{ id: "wl-core-v1", alias: "default" }],
        basis_set_id: "wl-core-v1",
        digest: "sha256:abc",
        description: "core",
        functions: [
          { name: "inc", input_type: "int", output_type: "int", code: "return x+1", function_id: "bf:inc" },
          { name: "double", input_type: "int", output_type: "int", code: "return x*2", function_id: "bf:double" },
        ],
      });
    }
    if (url === "/api/state") return json(state("TL<int>([3, 5, 7])"));
    if (url === "/api/trajectory") {
      const steps = (body as { steps: string[] }).steps;
      return json({
        root: state("TL<int>([1, 2, 3])"),
        steps: steps.map((name, index) => ({
          function: name,
          ok: true,
          state: state(`after-${name}-${index}`),
          error: null,
        })),
      });
    }
    if (url === "/api/successors") {
      return json({
        state: state("TL<int>([1, 2, 3])"),
        successors: [
          { function: "inc", ok: true, state: state("TL<int>([2, 3, 4])"), error: null, self_loop: false },
          { function: "double", ok: true, state: state("TL<int>([2, 4, 6])"), error: null, self_loop: false },
          { function: "half", ok: false, state: null, error: "TypeError", self_loop: false },
        ],
        applicable: 3,
        dead_end: false,
      });
    }
    if (url === "/api/corpora") return json({ corpora: [] });
    return json({});
  });
  vi.stubGlobal("fetch", fetchStub);
  return calls;
}

describe("App", () => {
  let calls: { url: string; body: unknown }[];

  beforeEach(() => {
    calls = stubApi();
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  async function lastTrajectoryCall() {
    const trajectoryCalls = calls.filter((call) => call.url === "/api/trajectory");
    return trajectoryCalls[trajectoryCalls.length - 1]?.body as { steps: string[] };
  }

  it("renders the root state once the trajectory loads", async () => {
    render(<App />);
    expect(await screen.findByTitle("root")).toHaveTextContent("1, 2, 3");
  });

  it("appends a step chosen from the picker", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByTitle("root");

    await user.click(screen.getByRole("button", { name: /add step/ }));
    const picker = await screen.findByPlaceholderText("filter functions…");
    await user.click(within(picker.parentElement as HTMLElement).getByRole("button", { name: /^.?inc/ }));

    await waitFor(async () => expect((await lastTrajectoryCall()).steps).toEqual(["inc"]));
  });

  it("drops downstream steps when an earlier edge is replaced", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByTitle("root");

    // Build inc → inc, then change the first edge to double.
    await user.click(screen.getByRole("button", { name: /add step/ }));
    let picker = await screen.findByPlaceholderText("filter functions…");
    await user.click(within(picker.parentElement as HTMLElement).getByRole("button", { name: /^.?inc/ }));
    await waitFor(async () => expect((await lastTrajectoryCall()).steps).toEqual(["inc"]));

    await user.click(await screen.findByRole("button", { name: /add step/ }));
    picker = await screen.findByPlaceholderText("filter functions…");
    await user.click(within(picker.parentElement as HTMLElement).getByRole("button", { name: /^.?inc/ }));
    await waitFor(async () => expect((await lastTrajectoryCall()).steps).toEqual(["inc", "inc"]));

    // Two steps exist by now; the first one is the edge being replaced.
    await user.click(screen.getAllByTitle("change this function")[0]);
    picker = await screen.findByPlaceholderText("filter functions…");
    await user.click(within(picker.parentElement as HTMLElement).getByRole("button", { name: /^.?double/ }));

    await waitFor(async () => expect((await lastTrajectoryCall()).steps).toEqual(["double"]));
  });

  it("shows what each candidate function would produce", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByTitle("root");
    await user.click(screen.getByRole("button", { name: /add step/ }));

    expect(await screen.findByText(/→ TL<int>\(\[2, 4, 6\]\)/)).toBeInTheDocument();
    // A function that fails on this state is shown, not hidden.
    expect(screen.getByText(/✗ TypeError/)).toBeInTheDocument();
  });
});
