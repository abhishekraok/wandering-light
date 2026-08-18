import type { Expansion, ExpansionEdge } from "../types";

export interface Positioned {
  id: string;
  nodeId: number;
  x: number;
  y: number;
  depth: number;
  label: string;
  type: string;
  wire: string;
}

const COLUMN = 260;
const ROW = 74;

/**
 * Lay an expansion out in depth columns.
 *
 * Depth is the meaningful axis here -- it is the certified distance from the
 * root -- so a column per depth reads directly as "one step", which a force
 * layout would scramble. Rows are centred so shallow layers stay legible next
 * to a wide frontier.
 */
export function layoutExpansion(expansion: Expansion): Positioned[] {
  const byDepth = new Map<number, typeof expansion.nodes>();
  for (const node of expansion.nodes) {
    const bucket = byDepth.get(node.depth) ?? [];
    bucket.push(node);
    byDepth.set(node.depth, bucket);
  }
  const tallest = Math.max(...[...byDepth.values()].map((b) => b.length), 1);
  const positioned: Positioned[] = [];
  for (const [depth, bucket] of [...byDepth.entries()].sort((a, b) => a[0] - b[0])) {
    const offset = ((tallest - bucket.length) * ROW) / 2;
    bucket.forEach((node, index) => {
      positioned.push({
        id: String(node.node_id),
        nodeId: node.node_id,
        x: depth * COLUMN,
        y: offset + index * ROW,
        depth: node.depth,
        label: node.label,
        type: node.type,
        wire: node.wire,
      });
    });
  }
  return positioned;
}

/**
 * Edges on some shortest root-to-target path, as `source:function:target` keys.
 *
 * Breadth-first from the root over the expansion's own edges: the first time a
 * node is reached is along a shortest path, which is the same argument the
 * corpus generator's distances rest on.
 */
export function shortestPathEdges(
  edges: ExpansionEdge[],
  rootId: number,
  targetId: number,
): Set<string> {
  if (rootId === targetId) return new Set();
  const outgoing = new Map<number, ExpansionEdge[]>();
  for (const edge of edges) {
    const bucket = outgoing.get(edge.source) ?? [];
    bucket.push(edge);
    outgoing.set(edge.source, bucket);
  }
  const cameFrom = new Map<number, ExpansionEdge>();
  const seen = new Set<number>([rootId]);
  const queue: number[] = [rootId];
  while (queue.length > 0) {
    const current = queue.shift() as number;
    if (current === targetId) break;
    for (const edge of outgoing.get(current) ?? []) {
      if (seen.has(edge.target)) continue;
      seen.add(edge.target);
      cameFrom.set(edge.target, edge);
      queue.push(edge.target);
    }
  }
  const path = new Set<string>();
  let cursor = targetId;
  while (cursor !== rootId) {
    const edge = cameFrom.get(cursor);
    if (!edge) return new Set();
    path.add(`${edge.source}:${edge.function}:${edge.target}`);
    cursor = edge.source;
  }
  return path;
}

/** Function names along a shortest root-to-target path, root outwards. */
export function shortestPathFunctions(
  edges: ExpansionEdge[],
  rootId: number,
  targetId: number,
): string[] {
  const keys = shortestPathEdges(edges, rootId, targetId);
  const byTarget = new Map<number, ExpansionEdge>();
  for (const edge of edges) {
    if (keys.has(`${edge.source}:${edge.function}:${edge.target}`)) {
      byTarget.set(edge.target, edge);
    }
  }
  const names: string[] = [];
  let cursor = targetId;
  while (cursor !== rootId) {
    const edge = byTarget.get(cursor);
    if (!edge) break;
    names.push(edge.function);
    cursor = edge.source;
  }
  return names.reverse();
}
