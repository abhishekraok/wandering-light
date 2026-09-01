import { describe, expect, it } from "vitest";

import {
  canvasEdgeId,
  canvasNodeId,
  graphElements,
  graphLayoutPositions,
  serializedTypedListType,
  shortestRootPath,
} from "./graph-model";
import type { GraphPayload } from "./types";

function graph(): GraphPayload {
  return {
    nodes: [
      { id: 0, x: 0, y: 0, depth: 0, label: "root", value: "root", role: "root" },
      { id: 1, x: 1, y: 0, depth: 1, label: "one", value: "one", role: "state" },
      { id: 2, x: 2, y: 0, depth: 2, label: "two", value: "two", role: "target" },
    ],
    edges: [
      { id: "0:0", source: 0, target: 0, function_names: ["identity"], highlighted: false },
      { id: "0:1", source: 0, target: 1, function_names: ["inc"], highlighted: true },
      { id: "1:0", source: 1, target: 0, function_names: ["dec"], highlighted: false },
      { id: "1:2", source: 1, target: 2, function_names: ["double"], highlighted: true },
    ],
    root_ids: [0],
    total_nodes: 3,
    total_edges: 4,
    rendered_nodes: 3,
    rendered_edge_groups: 4,
    truncated: false,
    diagnostics: {
      self_loop_groups: 1,
      parallel_function_groups: 0,
      convergent_nodes: 1,
      directed_cycle_groups: 1,
    },
  };
}

describe("graphElements", () => {
  it("preserves self-loops and marks both sides of a reciprocal pair", () => {
    const elements = graphElements(graph());
    const selfLoop = elements.find((element) => element.data.id === canvasEdgeId("0:0"));
    const forward = elements.find((element) => element.data.id === canvasEdgeId("0:1"));
    const reverse = elements.find((element) => element.data.id === canvasEdgeId("1:0"));

    expect(selfLoop?.classes).toContain("self-loop");
    expect(forward?.classes).toContain("reciprocal");
    expect(reverse?.classes).toContain("reciprocal");
    expect(forward?.data.source).toBe(canvasNodeId(0));
    expect(reverse?.data.target).toBe(canvasNodeId(0));
  });

  it("uses a compact overview label and retains a longer interactive label", () => {
    const payload = graph();
    payload.nodes[0].value = "a deliberately long typed-list value that would overlap nearby nodes";
    const root = graphElements(payload).find(
      (element) => element.data.id === canvasNodeId(0),
    );

    expect(String(root?.data.displayLabel).length).toBeLessThan(
      String(root?.data.expandedLabel).length,
    );
    expect(root?.data.expandedLabel).toContain("deliberately long typed-list value");
  });
});

describe("shortestRootPath", () => {
  it("ignores self-loops and returns a deterministic root path", () => {
    expect(shortestRootPath(graph(), 2).map((edge) => edge.id)).toEqual(["0:1", "1:2"]);
  });

  it("returns an empty path for a root or an unreachable node", () => {
    expect(shortestRootPath(graph(), 0)).toEqual([]);
    expect(shortestRootPath(graph(), 99)).toEqual([]);
  });
});

describe("graphLayoutPositions", () => {
  it("gives ordinary graphs relaxed depth spacing", () => {
    const positions = graphLayoutPositions(graph());
    expect((positions.get(1)?.x ?? 0) - (positions.get(0)?.x ?? 0)).toBeGreaterThanOrEqual(
      300,
    );
  });

  it("wraps a dense depth layer so fit is not defeated by minimum zoom", () => {
    const payload = graph();
    payload.nodes = Array.from({ length: 900 }, (_, id) => ({
      id,
      x: 1,
      y: 449.5 - id,
      depth: 1,
      label: String(id),
      value: String(id),
      role: "state" as const,
    }));
    const positions = graphLayoutPositions(payload);
    const values = [...positions.values()];
    const width = Math.max(...values.map((point) => point.x)) - Math.min(...values.map((point) => point.x));
    const height = Math.max(...values.map((point) => point.y)) - Math.min(...values.map((point) => point.y));

    expect(new Set(values.map((point) => point.x)).size).toBeGreaterThan(1);
    expect(width).toBeLessThan(3_000);
    expect(height).toBeLessThan(3_000);
  });

  it("keeps several dense depth bands within the graph's minimum fit zoom", () => {
    const payload = graph();
    payload.nodes = Array.from({ length: 900 }, (_, id) => ({
      id,
      x: id % 4,
      y: 112 - Math.floor(id / 4),
      depth: id % 4,
      label: String(id),
      value: String(id),
      role: "state" as const,
    }));
    const values = [...graphLayoutPositions(payload).values()];
    const width = Math.max(...values.map((point) => point.x)) - Math.min(...values.map((point) => point.x)) + 48;
    const height = Math.max(...values.map((point) => point.y)) - Math.min(...values.map((point) => point.y)) + 48;
    const fitZoom = Math.min((694 - 144) / width, (600 - 144) / height);

    expect(fitZoom).toBeGreaterThan(0.08);
  });
});

describe("serializedTypedListType", () => {
  it("reads only an exact, safe TypedList envelope", () => {
    expect(serializedTypedListType('{"type":"builtins.str","items":["hello"]}')).toBe(
      "builtins.str",
    );
    expect(serializedTypedListType('{"type":"custom.Widget","items":[]}')).toBeNull();
    expect(serializedTypedListType('{"type":"builtins.int","items":[],"extra":true}')).toBeNull();
    expect(serializedTypedListType("not json")).toBeNull();
  });
});
