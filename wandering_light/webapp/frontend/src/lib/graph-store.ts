import type { Expansion, TrajectoryResult } from "../types";

export interface GraphNode {
  wire: string;
  label: string;
  type: string;
}

export interface GraphEdge {
  key: string;
  source: string;
  target: string;
  fn: string;
}

export interface GraphState {
  nodes: Map<string, GraphNode>;
  edges: Map<string, GraphEdge>;
}

export interface Point {
  x: number;
  y: number;
}

export const NODE_WIDTH = 168;
export const NODE_HEIGHT = 42;
const COLUMN_GAP = 232;
const ROW_GAP = 78;

/**
 * The items alone: the type already has its own line on the node, so repeating
 * `TL<int>(...)` around every label costs width that the whole drawing pays for
 * in zoom.
 */
export function compactLabel(label: string): string {
  return label.replace(/^TL<[^>]+>\(/, "").replace(/\)$/, "");
}

export function emptyGraph(): GraphState {
  return { nodes: new Map(), edges: new Map() };
}

export function edgeKey(source: string, fn: string, target: string): string {
  return `${source} ${fn} ${target}`;
}

/**
 * Nodes are keyed by a state's wire form, which is canonical.
 *
 * That single choice is what makes the canvas a graph rather than a tree: two
 * routes that land on the same value land on the same node, so an involution
 * (`inc` then `dec`) closes a two-cycle and a no-op (`abs` on positives) closes
 * a self-loop, without anything having to detect either case.
 */
export function addNode(graph: GraphState, node: GraphNode): GraphState {
  if (graph.nodes.has(node.wire)) return graph;
  const nodes = new Map(graph.nodes);
  nodes.set(node.wire, node);
  return { nodes, edges: graph.edges };
}

export function addEdge(
  graph: GraphState,
  source: string,
  fn: string,
  target: string,
): GraphState {
  const key = edgeKey(source, fn, target);
  if (graph.edges.has(key)) return graph;
  const edges = new Map(graph.edges);
  edges.set(key, { key, source, target, fn });
  return { nodes: graph.nodes, edges };
}

/** Fold an expansion into the graph, keeping whatever is already there. */
export function mergeExpansion(graph: GraphState, expansion: Expansion): GraphState {
  const nodes = new Map(graph.nodes);
  const edges = new Map(graph.edges);
  const wireOf = new Map<number, string>();
  for (const node of expansion.nodes) {
    wireOf.set(node.node_id, node.wire);
    if (!nodes.has(node.wire)) {
      nodes.set(node.wire, { wire: node.wire, label: node.label, type: node.type });
    }
  }
  for (const edge of expansion.edges) {
    const source = wireOf.get(edge.source);
    const target = wireOf.get(edge.target);
    if (source === undefined || target === undefined) continue;
    const key = edgeKey(source, edge.function, target);
    if (!edges.has(key)) edges.set(key, { key, source, target, fn: edge.function });
  }
  return { nodes, edges };
}

/** Fold the trajectory's own path in, so editing steps draws on the canvas. */
export function mergeTrajectory(
  graph: GraphState,
  trajectory: TrajectoryResult,
): GraphState {
  let next = addNode(graph, {
    wire: trajectory.root.wire,
    label: trajectory.root.label,
    type: trajectory.root.type,
  });
  let previous = trajectory.root.wire;
  for (const step of trajectory.steps) {
    if (!step.ok || step.state === null) break;
    next = addNode(next, {
      wire: step.state.wire,
      label: step.state.label,
      type: step.state.type,
    });
    next = addEdge(next, previous, step.function, step.state.wire);
    previous = step.state.wire;
  }
  return next;
}

/** The graph containing only this trajectory: what "clear" goes back to. */
export function trajectoryOnly(trajectory: TrajectoryResult | null): GraphState {
  return trajectory === null ? emptyGraph() : mergeTrajectory(emptyGraph(), trajectory);
}

/** Breadth-first depth of every node from an anchor; unreached nodes are absent. */
export function depthsFrom(graph: GraphState, anchor: string): Map<string, number> {
  const depths = new Map<string, number>();
  if (!graph.nodes.has(anchor)) return depths;
  const outgoing = new Map<string, string[]>();
  for (const edge of graph.edges.values()) {
    const bucket = outgoing.get(edge.source) ?? [];
    bucket.push(edge.target);
    outgoing.set(edge.source, bucket);
  }
  depths.set(anchor, 0);
  const queue = [anchor];
  while (queue.length > 0) {
    const current = queue.shift() as string;
    const depth = depths.get(current) as number;
    for (const next of outgoing.get(current) ?? []) {
      if (depths.has(next)) continue;
      depths.set(next, depth + 1);
      queue.push(next);
    }
  }
  return depths;
}

/**
 * Positions for nodes that do not have one yet.
 *
 * Existing entries are returned untouched, so a node never jumps when the graph
 * grows and a node dragged by hand stays where it was put. Layers are wrapped
 * into columns sized to keep the drawing roughly square.
 */
export function layoutNew(
  graph: GraphState,
  anchor: string,
  positions: ReadonlyMap<string, Point>,
  origin: string | null = null,
): Map<string, Point> {
  const next = new Map(positions);
  const missing = [...graph.nodes.keys()].filter((wire) => !next.has(wire));
  if (missing.length === 0) return next;

  const depths = depthsFrom(graph, anchor);
  const layers = new Map<number, string[]>();
  const orphans: string[] = [];
  for (const wire of missing) {
    const depth = depths.get(wire);
    if (depth === undefined) orphans.push(wire);
    else layers.set(depth, [...(layers.get(depth) ?? []), wire]);
  }
  const maxDepth = Math.max(0, ...[...layers.keys()]);
  if (orphans.length > 0) layers.set(maxDepth + 1, orphans);

  const rowsPerColumn = Math.max(
    2,
    Math.round(Math.sqrt((graph.nodes.size * COLUMN_GAP) / ROW_GAP)),
  );
  // New nodes appear just right of whatever they grew from, so an expansion
  // reads as growth next to its cause rather than as a slab bolted onto the far
  // side of the canvas.
  const placed = [...next.values()];
  const originPoint = origin === null ? undefined : next.get(origin);
  let column =
    originPoint !== undefined
      ? Math.round(originPoint.x / COLUMN_GAP) + 1
      : placed.length === 0
        ? 0
        : Math.round(Math.max(...placed.map((p) => p.x)) / COLUMN_GAP) + 1;

  /** Lowest free y in a column, so new rows stack under what is already there. */
  const floorOf = (columnIndex: number): number => {
    const x = columnIndex * COLUMN_GAP;
    const occupied = placed.filter((point) => Math.abs(point.x - x) < COLUMN_GAP / 2);
    return occupied.length === 0 ? 0 : Math.max(...occupied.map((p) => p.y)) + ROW_GAP;
  };

  for (const depth of [...layers.keys()].sort((a, b) => a - b)) {
    const layer = layers.get(depth) as string[];
    layer.forEach((wire, index) => {
      const row = index % rowsPerColumn;
      const offset = Math.floor(index / rowsPerColumn);
      const columnIndex = column + offset;
      next.set(wire, {
        x: columnIndex * COLUMN_GAP,
        y:
          floorOf(columnIndex) +
          (row - Math.min(layer.length, rowsPerColumn) / 2) * ROW_GAP,
      });
    });
    column += Math.ceil(layer.length / rowsPerColumn);
  }
  return next;
}

/** Shortest route between two nodes, as the edges to follow. Empty if none. */
export function pathBetween(graph: GraphState, from: string, to: string): GraphEdge[] {
  if (from === to || !graph.nodes.has(from) || !graph.nodes.has(to)) return [];
  const outgoing = new Map<string, GraphEdge[]>();
  for (const edge of graph.edges.values()) {
    const bucket = outgoing.get(edge.source) ?? [];
    bucket.push(edge);
    outgoing.set(edge.source, bucket);
  }
  const cameFrom = new Map<string, GraphEdge>();
  const seen = new Set<string>([from]);
  const queue = [from];
  while (queue.length > 0) {
    const current = queue.shift() as string;
    if (current === to) break;
    for (const edge of outgoing.get(current) ?? []) {
      if (seen.has(edge.target)) continue;
      seen.add(edge.target);
      cameFrom.set(edge.target, edge);
      queue.push(edge.target);
    }
  }
  const path: GraphEdge[] = [];
  let cursor = to;
  while (cursor !== from) {
    const edge = cameFrom.get(cursor);
    if (edge === undefined) return [];
    path.push(edge);
    cursor = edge.source;
  }
  return path.reverse();
}

/** Edges that have a partner running the other way, keyed both ways. */
export function reciprocalKeys(graph: GraphState): Set<string> {
  const pairs = new Set<string>();
  for (const edge of graph.edges.values()) {
    if (edge.source !== edge.target) pairs.add(`${edge.source} ${edge.target}`);
  }
  const reciprocal = new Set<string>();
  for (const edge of graph.edges.values()) {
    if (edge.source === edge.target) continue;
    if (pairs.has(`${edge.target} ${edge.source}`)) reciprocal.add(edge.key);
  }
  return reciprocal;
}
