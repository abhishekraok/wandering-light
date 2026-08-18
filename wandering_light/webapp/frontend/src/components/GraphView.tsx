import { useMemo } from "react";
import {
  Background,
  Controls,
  MiniMap,
  ReactFlow,
  type Edge,
  type Node,
} from "@xyflow/react";

import { layoutExpansion, shortestPathEdges } from "../lib/layout";
import type { Expansion } from "../types";

function StateNode({ data }: { data: Record<string, unknown> }) {
  const classes = ["node-card"];
  if (data.isRoot) classes.push("root");
  if (data.isTarget) classes.push("target");
  if (data.onPath) classes.push("on-path");
  return (
    <div className={classes.join(" ")} title={String(data.label)}>
      <div className="depth">
        d{String(data.depth)} · {String(data.type)}
      </div>
      {String(data.label)}
    </div>
  );
}

const nodeTypes = { state: StateNode };

/**
 * The expansion, drawn in depth columns.
 *
 * Clicking a node selects it; the shortest route from the root is highlighted
 * so the graph answers "how would I get there" as directly as it answers "what
 * is reachable".
 */
export function GraphView({
  expansion,
  selected,
  targetWire,
  onSelect,
}: {
  expansion: Expansion | null;
  selected: number | null;
  targetWire: string | null;
  onSelect: (nodeId: number) => void;
}) {
  const { nodes, edges } = useMemo(() => {
    if (expansion === null) return { nodes: [] as Node[], edges: [] as Edge[] };
    const rootId = expansion.stats.root_id;
    const positioned = layoutExpansion(expansion);
    const path =
      selected === null ? new Set<string>() : shortestPathEdges(expansion.edges, rootId, selected);
    const onPath = new Set<number>([rootId]);
    for (const key of path) {
      const [source, , target] = key.split(":");
      onPath.add(Number(source));
      onPath.add(Number(target));
    }
    const flowNodes: Node[] = positioned.map((item) => ({
      id: item.id,
      type: "state",
      position: { x: item.x, y: item.y },
      data: {
        label: item.label,
        depth: item.depth,
        type: item.type,
        isRoot: item.nodeId === rootId,
        isTarget: targetWire !== null && item.wire === targetWire,
        onPath: onPath.has(item.nodeId) && (selected !== null || item.nodeId === rootId),
      },
      selected: item.nodeId === selected,
    }));
    const flowEdges: Edge[] = expansion.edges.map((edge, index) => {
      const key = `${edge.source}:${edge.function}:${edge.target}`;
      const highlighted = path.has(key);
      return {
        id: `${index}-${key}`,
        source: String(edge.source),
        target: String(edge.target),
        label: edge.function,
        animated: highlighted,
        style: highlighted
          ? { stroke: "var(--accent)", strokeWidth: 2 }
          : { stroke: "var(--line)" },
        labelStyle: { fill: "var(--muted)", fontSize: 10 },
        labelBgStyle: { fill: "var(--bg)" },
      };
    });
    return { nodes: flowNodes, edges: flowEdges };
  }, [expansion, selected, targetWire]);

  if (expansion === null) {
    return (
      <div className="section muted" style={{ height: "100%" }}>
        Expand a state to draw its graph.
      </div>
    );
  }

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      onNodeClick={(_event, node) => onSelect(Number(node.id))}
      fitView
      minZoom={0.1}
      proOptions={{ hideAttribution: true }}
    >
      <Background gap={22} color="var(--line)" />
      <Controls showInteractive={false} />
      <MiniMap pannable zoomable nodeColor="var(--line)" maskColor="rgba(0,0,0,0.5)" />
    </ReactFlow>
  );
}
