import { describe, expect, it } from "vitest";

import { layoutExpansion, shortestPathEdges, shortestPathFunctions } from "./layout";
import type { Expansion, ExpansionEdge } from "../types";

function node(node_id: number, depth: number) {
  return {
    id: `s${node_id}`,
    wire: `{"n":${node_id}}`,
    label: `TL<int>([${node_id}])`,
    type: "int",
    size: 1,
    depth,
    node_id,
  };
}

const edges: ExpansionEdge[] = [
  { source: 0, target: 1, function: "inc", depth: 0 },
  { source: 0, target: 2, function: "double", depth: 0 },
  { source: 1, target: 3, function: "inc", depth: 1 },
  { source: 2, target: 3, function: "dec", depth: 1 },
];

const expansion: Expansion = {
  nodes: [node(0, 0), node(1, 1), node(2, 1), node(3, 2)],
  edges,
  stats: {} as Expansion["stats"],
  types: {},
  function_edges: {},
  idle_functions: [],
};

describe("layoutExpansion", () => {
  it("puts each depth in its own column", () => {
    const positioned = layoutExpansion(expansion);
    const columns = new Map(positioned.map((p) => [p.nodeId, p.x]));
    expect(columns.get(0)).toBeLessThan(columns.get(1) as number);
    expect(columns.get(1)).toBe(columns.get(2));
    expect(columns.get(2)).toBeLessThan(columns.get(3) as number);
  });

  it("keeps every node exactly once", () => {
    expect(layoutExpansion(expansion).map((p) => p.nodeId).sort()).toEqual([0, 1, 2, 3]);
  });

  it("centres a short layer against the tallest one", () => {
    const positioned = layoutExpansion(expansion);
    const root = positioned.find((p) => p.nodeId === 0);
    const first = positioned.find((p) => p.nodeId === 1);
    expect(root?.y).toBeGreaterThan(first?.y as number);
  });

  it("wraps a wide layer instead of drawing one unreadable column", () => {
    const wide: Expansion = {
      ...expansion,
      nodes: [node(0, 0), ...Array.from({ length: 60 }, (_, i) => node(i + 1, 1))],
    };
    const positioned = layoutExpansion(wide);
    const layer = positioned.filter((p) => p.depth === 1);
    const columns = new Set(layer.map((p) => p.x));
    expect(columns.size).toBeGreaterThan(1);
    // Height stays bounded no matter how wide the frontier gets.
    const span = Math.max(...layer.map((p) => p.y)) - Math.min(...layer.map((p) => p.y));
    expect(span).toBeLessThan(18 * 74);
    // The root still sits left of every node one step out.
    const rootX = positioned.find((p) => p.nodeId === 0)?.x as number;
    expect(Math.min(...layer.map((p) => p.x))).toBeGreaterThan(rootX);
  });
});

describe("shortestPathEdges", () => {
  it("follows one shortest route to the target", () => {
    const path = shortestPathEdges(edges, 0, 3);
    expect(path.size).toBe(2);
    expect([...path].some((key) => key.endsWith(":3"))).toBe(true);
  });

  it("is empty for the root itself and for unreachable nodes", () => {
    expect(shortestPathEdges(edges, 0, 0).size).toBe(0);
    expect(shortestPathEdges(edges, 0, 99).size).toBe(0);
  });

  it("names the functions root-outwards", () => {
    const names = shortestPathFunctions(edges, 0, 3);
    expect(names).toHaveLength(2);
    expect(["inc", "double"]).toContain(names[0]);
  });
});
