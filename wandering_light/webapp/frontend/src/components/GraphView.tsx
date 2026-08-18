import { useCallback, useEffect, useMemo, useRef } from "react";
import {
  Background,
  Controls,
  Handle,
  MiniMap,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeChange,
  type ReactFlowInstance,
} from "@xyflow/react";

import {
  NODE_HEIGHT,
  NODE_WIDTH,
  compactLabel,
  depthsFrom,
  pathBetween,
  reciprocalKeys,
  type GraphState,
  type Point,
} from "../lib/graph-store";
import { FlowEdge } from "./FlowEdge";

function StateNode({ data }: { data: Record<string, unknown> }) {
  const classes = ["node-card"];
  if (data.isAnchor) classes.push("root");
  if (data.isTarget) classes.push("target");
  if (data.onPath) classes.push("on-path");
  return (
    <div className={classes.join(" ")} title={String(data.full ?? data.label)}>
      {/* Edges attach to handles; without them React Flow draws no lines at all. */}
      <Handle type="target" position={Position.Left} className="node-handle" />
      <div className="depth">
        {data.depth === undefined ? "—" : `d${String(data.depth)}`} · {String(data.type)}
      </div>
      {String(data.label)}
      <Handle type="source" position={Position.Right} className="node-handle" />
    </div>
  );
}

const nodeTypes = { state: StateNode };
const edgeTypes = { flow: FlowEdge };

/**
 * The accumulated graph.
 *
 * Positions live outside this component so a node keeps its place when the
 * graph grows and keeps a hand-dragged place across every later expansion.
 */
export function GraphView({
  graph,
  positions,
  anchor,
  selected,
  targetWire,
  onSelect,
  onMove,
  fitSignal,
}: {
  graph: GraphState;
  positions: ReadonlyMap<string, Point>;
  anchor: string | null;
  selected: string | null;
  targetWire: string | null;
  onSelect: (wire: string | null) => void;
  onMove: (wire: string, position: Point) => void;
  /** Bumped by the caller when the graph grew and should be re-framed. */
  fitSignal: number;
}) {
  const flow = useRef<ReactFlowInstance | null>(null);
  const { nodes, edges } = useMemo(() => {
    const depths = anchor === null ? new Map<string, number>() : depthsFrom(graph, anchor);
    const highlighted = new Set(
      anchor === null || selected === null
        ? []
        : pathBetween(graph, anchor, selected).map((edge) => edge.key),
    );
    const reciprocal = reciprocalKeys(graph);

    const flowNodes: Node[] = [...graph.nodes.values()].map((node) => ({
      id: node.wire,
      type: "state",
      position: positions.get(node.wire) ?? { x: 0, y: 0 },
      width: NODE_WIDTH,
      height: NODE_HEIGHT,
      selected: node.wire === selected,
      data: {
        label: compactLabel(node.label),
        full: node.label,
        type: node.type,
        depth: depths.get(node.wire),
        isAnchor: node.wire === anchor,
        isTarget: targetWire !== null && node.wire === targetWire,
        onPath: highlighted.size > 0 && (node.wire === anchor || node.wire === selected),
      },
    }));

    const flowEdges: Edge[] = [...graph.edges.values()].map((edge) => ({
      id: edge.key,
      source: edge.source,
      target: edge.target,
      type: "flow",
      label: edge.fn,
      data: {
        highlighted: highlighted.has(edge.key),
        reciprocal: reciprocal.has(edge.key),
      },
    }));
    return { nodes: flowNodes, edges: flowEdges };
  }, [graph, positions, anchor, selected, targetWire]);

  // A drag-end change can omit the position; the node carries it either way.
  const positionOf = useMemo(
    () => new Map(nodes.map((node) => [node.id, node.position])),
    [nodes],
  );

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      for (const change of changes) {
        // Only the end of a drag is persisted. Feeding intermediate positions
        // back through props re-renders mid-gesture and fights React Flow's own
        // drag state, which makes the node stutter and land short.
        if (change.type === "position" && change.dragging === false) {
          const moved = change.position ?? positionOf.get(change.id);
          if (moved) onMove(change.id, moved);
        }
      }
    },
    [onMove, positionOf],
  );

  useEffect(() => {
    if (fitSignal === 0) return;
    // After the nodes for this signal have rendered.
    const timer = window.setTimeout(
      () => flow.current?.fitView({ padding: 0.2, maxZoom: 1, duration: 350 }),
      60,
    );
    return () => window.clearTimeout(timer);
  }, [fitSignal]);

  if (graph.nodes.size === 0) {
    return (
      <div className="section muted" style={{ height: "100%" }}>
        Enter a root state to start the graph.
      </div>
    );
  }

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      onInit={(instance) => (flow.current = instance)}
      onNodesChange={onNodesChange}
      onNodeClick={(_event, node) => onSelect(node.id)}
      onPaneClick={() => onSelect(null)}
      fitView
      fitViewOptions={{ padding: 0.2, maxZoom: 1 }}
      minZoom={0.05}
      nodesConnectable={false}
      proOptions={{ hideAttribution: true }}
    >
      <Background gap={22} color="var(--line)" />
      <Controls showInteractive={false} position="top-left" />
      <MiniMap
        position="top-right"
        pannable
        zoomable
        // The minimap paints SVG fills, where a CSS variable does not resolve.
        nodeColor="#6ea8fe"
        maskColor="rgba(120,130,150,0.25)"
      />
    </ReactFlow>
  );
}
