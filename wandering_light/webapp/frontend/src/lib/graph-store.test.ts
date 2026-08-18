import { describe, expect, it } from "vitest";

import {
  addEdge,
  addNode,
  compactLabel,
  depthsFrom,
  emptyGraph,
  layoutNew,
  mergeExpansion,
  mergeTrajectory,
  pathBetween,
  reciprocalKeys,
  trajectoryOnly,
} from "./graph-store";
import type { Expansion, Step, TrajectoryResult } from "../types";

function state(name: string) {
  return { id: name, wire: `w:${name}`, label: `TL<int>([${name}])`, type: "int", size: 1 };
}

function step(fn: string, name: string): Step {
  return { function: fn, ok: true, state: state(name), error: null };
}

const trajectory: TrajectoryResult = {
  root: state("a"),
  steps: [step("inc", "b"), step("dec", "a")],
};

function expansion(
  nodes: [number, string][],
  edges: [number, number, string][],
): Expansion {
  return {
    nodes: nodes.map(([id, name], index) => ({ ...state(name), node_id: id, depth: index })),
    edges: edges.map(([source, target, fn]) => ({ source, target, function: fn, depth: 0 })),
    stats: { root_id: nodes[0][0] } as Expansion["stats"],
    types: {},
    function_edges: {},
    idle_functions: [],
  };
}

describe("merging into one graph", () => {
  it("closes an involution onto the node it returns to", () => {
    const graph = mergeTrajectory(emptyGraph(), trajectory);
    // a -inc-> b -dec-> a is two nodes and two edges, not three nodes.
    expect(graph.nodes.size).toBe(2);
    expect(graph.edges.size).toBe(2);
    expect([...graph.edges.values()].map((e) => e.fn).sort()).toEqual(["dec", "inc"]);
  });

  it("keeps a self-loop as an edge from a node to itself", () => {
    const graph = mergeExpansion(emptyGraph(), expansion([[0, "a"]], [[0, 0, "abs"]]));
    const edge = [...graph.edges.values()][0];
    expect(edge.source).toBe(edge.target);
    expect(graph.nodes.size).toBe(1);
  });

  it("merges expansions and trajectories into the same nodes", () => {
    const first = mergeTrajectory(emptyGraph(), trajectory);
    const merged = mergeExpansion(
      first,
      expansion(
        [
          [7, "a"],
          [8, "c"],
        ],
        [[7, 8, "double"]],
      ),
    );
    // `a` came from the trajectory and from the expansion under a different id.
    expect(merged.nodes.size).toBe(3);
    expect(merged.edges.size).toBe(3);
  });

  it("is idempotent: re-merging the same payload adds nothing", () => {
    const once = mergeTrajectory(emptyGraph(), trajectory);
    const twice = mergeTrajectory(once, trajectory);
    expect(twice.nodes.size).toBe(once.nodes.size);
    expect(twice.edges.size).toBe(once.edges.size);
  });

  it("stops folding a trajectory at the first failed step", () => {
    const broken: TrajectoryResult = {
      root: state("a"),
      steps: [step("inc", "b"), { function: "half", ok: false, state: null, error: "no" }],
    };
    expect(mergeTrajectory(emptyGraph(), broken).nodes.size).toBe(2);
  });

  it("clears back to the current trajectory rather than to nothing", () => {
    const grown = mergeExpansion(
      mergeTrajectory(emptyGraph(), trajectory),
      expansion(
        [
          [0, "a"],
          [1, "z"],
        ],
        [[0, 1, "square"]],
      ),
    );
    expect(grown.nodes.size).toBe(3);
    expect(trajectoryOnly(trajectory).nodes.size).toBe(2);
    expect(trajectoryOnly(null).nodes.size).toBe(0);
  });
});

describe("layout", () => {
  const graph = addEdge(
    addEdge(
      addNode(
        addNode(addNode(emptyGraph(), { wire: "w:a", label: "a", type: "int" }), {
          wire: "w:b",
          label: "b",
          type: "int",
        }),
        { wire: "w:c", label: "c", type: "int" },
      ),
      "w:a",
      "inc",
      "w:b",
    ),
    "w:b",
    "inc",
    "w:c",
  );

  it("measures depth from the anchor", () => {
    const depths = depthsFrom(graph, "w:a");
    expect(depths.get("w:a")).toBe(0);
    expect(depths.get("w:c")).toBe(2);
    expect(depthsFrom(graph, "w:missing").size).toBe(0);
  });

  it("never moves a node that already has a position", () => {
    const pinned = new Map([["w:a", { x: -999, y: -999 }]]);
    const positions = layoutNew(graph, "w:a", pinned);
    expect(positions.get("w:a")).toEqual({ x: -999, y: -999 });
    expect(positions.size).toBe(3);
  });

  it("places deeper nodes further right", () => {
    const positions = layoutNew(graph, "w:a", new Map());
    expect(positions.get("w:b")?.x).toBeGreaterThan(positions.get("w:a")?.x as number);
    expect(positions.get("w:c")?.x).toBeGreaterThan(positions.get("w:b")?.x as number);
  });

  it("grows an expansion beside the node it came from", () => {
    // `w:c` is already placed far to the right; a new node grown from `w:a`
    // belongs next to `w:a`, not past everything on the canvas.
    const pinned = new Map([
      ["w:a", { x: 0, y: 0 }],
      ["w:b", { x: 300, y: 0 }],
      ["w:c", { x: 3000, y: 0 }],
    ]);
    const grown = addEdge(
      addNode(graph, { wire: "w:new", label: "n", type: "int" }),
      "w:a",
      "double",
      "w:new",
    );
    const positions = layoutNew(grown, "w:a", pinned, "w:a");
    expect(positions.get("w:new")?.x).toBeLessThan(3000);
    expect(positions.get("w:new")?.x).toBeGreaterThan(0);
  });

  it("stacks a new node under one already sitting in that column", () => {
    // Lay the chain out first so the pinned positions use the real column gap,
    // then grow a second child of the anchor into the column `w:b` occupies.
    const chain = layoutNew(graph, "w:a", new Map());
    const grown = addEdge(
      addNode(graph, { wire: "w:new", label: "n", type: "int" }),
      "w:a",
      "double",
      "w:new",
    );
    const positions = layoutNew(grown, "w:a", chain, "w:a");
    const sibling = positions.get("w:b") as { x: number; y: number };
    const placed = positions.get("w:new") as { x: number; y: number };
    expect(placed.x).toBe(sibling.x);
    expect(placed.y).toBeGreaterThan(sibling.y);
  });

  it("still places a node the anchor cannot reach", () => {
    const detached = addNode(graph, { wire: "w:island", label: "i", type: "int" });
    expect(layoutNew(detached, "w:a", new Map()).has("w:island")).toBe(true);
  });
});

describe("compactLabel", () => {
  it("keeps the items and drops the wrapper", () => {
    expect(compactLabel("TL<int>([1, 2, 3])")).toBe("[1, 2, 3]");
    expect(compactLabel("TL<str>(['a'])")).toBe("['a']");
  });

  it("leaves anything it does not recognise alone", () => {
    expect(compactLabel("[1, 2]")).toBe("[1, 2]");
  });
});

describe("pathBetween", () => {
  it("returns the functions to follow, root outwards", () => {
    const graph = mergeTrajectory(emptyGraph(), {
      root: state("a"),
      steps: [step("inc", "b"), step("double", "c")],
    });
    expect(pathBetween(graph, "w:a", "w:c").map((e) => e.fn)).toEqual(["inc", "double"]);
  });

  it("is empty for a node itself and for one that is unreachable", () => {
    const graph = mergeTrajectory(emptyGraph(), trajectory);
    expect(pathBetween(graph, "w:a", "w:a")).toEqual([]);
    expect(pathBetween(graph, "w:a", "w:nowhere")).toEqual([]);
  });
});

describe("reciprocalKeys", () => {
  it("finds edges that have a partner going the other way", () => {
    const graph = mergeTrajectory(emptyGraph(), trajectory);
    expect(reciprocalKeys(graph).size).toBe(2);
  });

  it("does not count a self-loop as reciprocal", () => {
    const graph = mergeExpansion(emptyGraph(), expansion([[0, "a"]], [[0, 0, "abs"]]));
    expect(reciprocalKeys(graph).size).toBe(0);
  });
});
