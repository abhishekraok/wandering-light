import { describe, expect, it } from "vitest";

import { finalState, reachesTarget, replaceStep, truncateAt } from "./trajectory";
import type { Step } from "../types";

function step(name: string, wire: string): Step {
  return {
    function: name,
    ok: true,
    state: { id: wire, wire, label: wire, type: "int", size: 1 },
    error: null,
  };
}

describe("editing a trajectory", () => {
  it("drops downstream steps when one is replaced", () => {
    expect(replaceStep(["inc", "double", "neg"], 1, "half")).toEqual(["inc", "half"]);
  });

  it("truncates at a deleted step", () => {
    expect(truncateAt(["inc", "double", "neg"], 1)).toEqual(["inc"]);
    expect(truncateAt(["inc"], 0)).toEqual([]);
  });
});

describe("reading a trajectory's result", () => {
  it("has no final state once a step failed", () => {
    const failed: Step = { function: "half", ok: false, state: null, error: "boom" };
    expect(finalState([step("inc", "a"), failed])).toBeNull();
  });

  it("matches the target on the wire form", () => {
    expect(reachesTarget([step("inc", "target")], "root", "target")).toBe(true);
    expect(reachesTarget([step("inc", "other")], "root", "target")).toBe(false);
  });

  it("counts an empty trajectory as solved only when root is the target", () => {
    expect(reachesTarget([], "same", "same")).toBe(true);
    expect(reachesTarget([], "root", "target")).toBe(false);
  });
});
