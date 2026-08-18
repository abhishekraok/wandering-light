import type { GraphEdge, GraphPayload } from "./types";

export interface Point {
  x: number;
  y: number;
}

export interface MappedGraphElement {
  group: "nodes" | "edges";
  data: Record<string, string | number>;
  classes: string;
  position?: Point;
}

export const canvasNodeId = (id: number): string => `node:${id}`;
export const canvasEdgeId = (id: string): string => `edge:${id}`;

const DEPTH_GAP = 250;
const WRAPPED_COLUMN_GAP = 96;
const ROW_GAP = 88;
const SAFE_TYPED_LIST_TYPES = new Set([
  "builtins.int",
  "builtins.float",
  "builtins.str",
  "builtins.bool",
  "builtins.list",
  "builtins.tuple",
  "builtins.set",
  "builtins.dict",
  "builtins.bytes",
  "builtins.bytearray",
  "builtins.complex",
  "builtins.range",
]);

export function graphElements(
  payload: GraphPayload,
  savedPositions: ReadonlyMap<number, Point> = new Map<number, Point>(),
): MappedGraphElement[] {
  const directedPairs = new Set(payload.edges.map((edge) => `${edge.source}:${edge.target}`));
  const layoutPositions = graphLayoutPositions(payload);
  const nodeElements: MappedGraphElement[] = payload.nodes.map((node) => ({
    group: "nodes",
    data: {
      id: canvasNodeId(node.id),
      nodeId: node.id,
      displayLabel: displayLabel(node.value, node.id),
      role: node.role,
    },
    classes: `role-${node.role}`,
    position: savedPositions.get(node.id) ?? layoutPositions.get(node.id),
  }));
  const edgeElements: MappedGraphElement[] = payload.edges.map((edge) => {
    const classes = [
      edge.highlighted ? "stored-highlight" : "",
      edge.source === edge.target ? "self-loop" : "",
      edge.source !== edge.target && directedPairs.has(`${edge.target}:${edge.source}`)
        ? "reciprocal"
        : "",
    ].filter(Boolean);
    return {
      group: "edges",
      data: {
        id: canvasEdgeId(edge.id),
        edgeKey: edge.id,
        source: canvasNodeId(edge.source),
        target: canvasNodeId(edge.target),
        functionLabel: edge.function_names.join(" · "),
      },
      classes: classes.join(" "),
    };
  });
  return [...nodeElements, ...edgeElements];
}

/**
 * Preserve breadth-first depth from the server while wrapping dense layers into
 * narrow columns. A single unwrapped layer can otherwise be tens of thousands
 * of model pixels tall, beyond the canvas's minimum useful zoom.
 */
export function graphLayoutPositions(payload: GraphPayload): Map<number, Point> {
  const layers = new Map<number, GraphPayload["nodes"]>();
  for (const node of payload.nodes) {
    const layer = layers.get(node.depth) ?? [];
    layer.push(node);
    layers.set(node.depth, layer);
  }

  const positions = new Map<number, Point>();
  // Balance the total model-space width and height. This keeps the full graph
  // comfortably above Cytoscape's minimum zoom even when one or several BFS
  // layers contain most of the render cap.
  const rowsPerColumn = Math.max(
    1,
    Math.round(Math.sqrt((payload.nodes.length * WRAPPED_COLUMN_GAP) / ROW_GAP)),
  );
  let layerStartX = 100;
  const depths = [...layers.keys()].sort((left, right) => left - right);
  for (const depth of depths) {
    const nodes = layers.get(depth) ?? [];
    nodes.sort((left, right) => right.y - left.y || left.id - right.id);
    const columnCount = Math.max(1, Math.ceil(nodes.length / rowsPerColumn));
    for (let column = 0; column < columnCount; column += 1) {
      const start = column * rowsPerColumn;
      const columnNodes = nodes.slice(start, start + rowsPerColumn);
      const center = (columnNodes.length - 1) / 2;
      for (const [row, node] of columnNodes.entries()) {
        positions.set(node.id, {
          x: layerStartX + column * WRAPPED_COLUMN_GAP,
          y: 120 + (row - center) * ROW_GAP,
        });
      }
    }
    layerStartX += (columnCount - 1) * WRAPPED_COLUMN_GAP + DEPTH_GAP;
  }
  return positions;
}

/** Read only the safe builtin type tag; validation/execution remains server-side. */
export function serializedTypedListType(serialized: string): string | null {
  try {
    const payload = JSON.parse(serialized) as unknown;
    if (typeof payload !== "object" || payload === null || Array.isArray(payload)) return null;
    const record = payload as Record<string, unknown>;
    if (
      Object.keys(record).length !== 2 ||
      typeof record.type !== "string" ||
      !Array.isArray(record.items) ||
      !SAFE_TYPED_LIST_TYPES.has(record.type)
    ) {
      return null;
    }
    return record.type;
  } catch {
    return null;
  }
}

export function shortestRootPath(payload: GraphPayload, destination: number): GraphEdge[] {
  if (payload.root_ids.includes(destination)) return [];
  const outgoing = new Map<number, GraphEdge[]>();
  for (const edge of payload.edges) {
    const group = outgoing.get(edge.source) ?? [];
    group.push(edge);
    outgoing.set(edge.source, group);
  }
  for (const edges of outgoing.values()) {
    edges.sort((left, right) => left.target - right.target || left.id.localeCompare(right.id));
  }

  const queue = [...payload.root_ids].sort((left, right) => left - right);
  const visited = new Set(queue);
  const parent = new Map<number, GraphEdge>();
  while (queue.length > 0 && !visited.has(destination)) {
    const current = queue.shift();
    if (current === undefined) break;
    for (const edge of outgoing.get(current) ?? []) {
      if (edge.source === edge.target || visited.has(edge.target)) continue;
      visited.add(edge.target);
      parent.set(edge.target, edge);
      queue.push(edge.target);
    }
  }
  if (!visited.has(destination)) return [];

  const reversed: GraphEdge[] = [];
  let current = destination;
  while (!payload.root_ids.includes(current)) {
    const edge = parent.get(current);
    if (!edge) return [];
    reversed.push(edge);
    current = edge.source;
  }
  return reversed.reverse();
}

function displayLabel(value: string, id: number): string {
  const normalized = value.replaceAll("\n", " ");
  const valueLabel = normalized.length > 34 ? `${normalized.slice(0, 33)}…` : normalized;
  return `#${id}  ${valueLabel}`;
}
