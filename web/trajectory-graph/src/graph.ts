import cytoscape, {
  type Core,
  type EdgeSingular,
  type ElementDefinition,
  type NodeSingular,
  type Position,
} from "cytoscape";

import {
  canvasEdgeId,
  canvasNodeId,
  graphElements,
  graphLayoutPositions,
  shortestRootPath,
} from "./graph-model";
import type { GraphEdge, GraphNode, GraphPayload } from "./types";

export interface PathStep {
  edge: GraphEdge;
  from: GraphNode;
  to: GraphNode;
  functionName: string;
}

export type GraphSelection =
  | {
      kind: "node";
      node: GraphNode;
      path: PathStep[];
      incoming: GraphEdge[];
      outgoing: GraphEdge[];
    }
  | { kind: "edge"; edge: GraphEdge }
  | { kind: "none" };

interface ViewSnapshot {
  pan: Position;
  zoom: number;
  positions: Map<number, Position>;
  selectedNodeId: number | null;
}

type SelectionListener = (selection: GraphSelection) => void;
const MAX_VIEW_SNAPSHOTS = 16;

export class TrajectoryGraphCanvas {
  private readonly cy: Core;
  private readonly snapshots = new Map<string, ViewSnapshot>();
  private readonly onSelection: SelectionListener;
  private readonly resizeObserver: ResizeObserver;
  private payload: GraphPayload | null = null;
  private graphKey: string | null = null;
  private selectedNodeId: number | null = null;

  constructor(container: HTMLElement, onSelection: SelectionListener) {
    this.onSelection = onSelection;
    this.cy = cytoscape({
      container,
      elements: [],
      minZoom: 0.08,
      maxZoom: 3.5,
      // Cytoscape's normalized wheel delta is already small on high-resolution
      // trackpads. Keep this responsive enough that one deliberate scroll is
      // visible without making pinch/wheel zoom feel jumpy.
      wheelSensitivity: 0.7,
      boxSelectionEnabled: false,
      selectionType: "single",
      textureOnViewport: true,
      motionBlur: false,
      style: [
        {
          selector: "node",
          style: {
            width: 34,
            height: 34,
            shape: "ellipse",
            "background-color": "#8da2b5",
            "border-color": "#d5e2eb",
            "border-width": 1.5,
            label: "data(displayLabel)",
            color: "#dce8ef",
            "font-family": "IBM Plex Mono, SFMono-Regular, Consolas, monospace",
            "font-size": 11,
            "font-weight": 600,
            "text-margin-y": -12,
            "text-background-color": "#06101a",
            "text-background-opacity": 0.96,
            "text-background-padding": "5px",
            "text-background-shape": "roundrectangle",
            "text-border-color": "#294154",
            "text-border-opacity": 0.9,
            "text-border-width": 1,
            "min-zoomed-font-size": 7,
            "overlay-opacity": 0,
            "transition-property": "opacity, background-color, border-color, border-width",
            "transition-duration": 120,
          },
        },
        {
          selector: "node.hover-label, node.path-terminal",
          style: {
            label: "data(expandedLabel)",
            "z-index": 30,
          },
        },
        {
          selector: "node.role-root",
          style: {
            width: 42,
            height: 42,
            shape: "roundrectangle",
            "background-color": "#63e5c5",
            "border-color": "#d9fff6",
            color: "#ecfffb",
          },
        },
        {
          selector: "node.role-selected_path",
          style: {
            "background-color": "#ffb45b",
            "border-color": "#ffe0b9",
          },
        },
        {
          selector: "node.role-target",
          style: {
            "background-color": "#9d8cff",
            "border-color": "#e0d9ff",
          },
        },
        {
          selector: "node.role-selected_target",
          style: {
            width: 44,
            height: 44,
            "background-color": "#ff6b7d",
            "border-color": "#ffd3d9",
            "border-width": 3,
          },
        },
        {
          selector: "edge",
          style: {
            width: 1.45,
            "line-color": "#617689",
            "target-arrow-color": "#758b9d",
            "target-arrow-shape": "triangle",
            "arrow-scale": 0.8,
            "curve-style": "bezier",
            "loop-direction": "-35deg",
            "loop-sweep": "70deg",
            label: "",
            color: "#9eb0bd",
            "font-family": "IBM Plex Mono, SFMono-Regular, Consolas, monospace",
            "font-size": 8,
            "text-rotation": "autorotate",
            "text-margin-y": -8,
            "text-background-color": "#09111a",
            "text-background-opacity": 0.75,
            "text-background-padding": "2px",
            "min-zoomed-font-size": 7,
            "overlay-opacity": 0,
            "transition-property": "opacity, line-color, target-arrow-color, width",
            "transition-duration": 120,
          },
        },
        {
          selector: "edge.hover-label, edge:selected, edge.path-active, edge.stored-highlight",
          style: {
            label: "data(functionLabel)",
          },
        },
        {
          selector: "edge.stored-highlight",
          style: {
            width: 3,
            "line-color": "#ffb45b",
            "target-arrow-color": "#ffb45b",
            color: "#ffd7a6",
            "z-index": 8,
          },
        },
        {
          selector: "edge.reciprocal",
          style: {
            "curve-style": "unbundled-bezier",
            "control-point-distances": 34,
            "control-point-weights": 0.5,
          },
        },
        {
          selector: "edge.self-loop",
          style: {
            "loop-direction": "-42deg",
            "loop-sweep": "78deg",
          },
        },
        {
          selector: ".path-muted",
          style: {
            opacity: 0.12,
          },
        },
        {
          selector: "node.path-active",
          style: {
            opacity: 1,
            "background-color": "#63e5c5",
            "border-color": "#e4fff9",
            "border-width": 3,
            "z-index": 20,
          },
        },
        {
          selector: "node.path-terminal",
          style: {
            width: 48,
            height: 48,
            "background-color": "#ffb45b",
            "border-color": "#fff0dc",
          },
        },
        {
          selector: "edge.path-active",
          style: {
            opacity: 1,
            width: 4,
            "line-color": "#63e5c5",
            "target-arrow-color": "#63e5c5",
            color: "#bcfff0",
            "z-index": 18,
          },
        },
        {
          selector: ":selected",
          style: {
            "overlay-color": "#ffffff",
            "overlay-opacity": 0.1,
            "overlay-padding": 8,
          },
        },
      ],
    });

    this.cy.on("tap", "node", (event) => {
      const node = event.target as NodeSingular;
      this.selectNode(Number(node.data("nodeId")), false);
    });
    this.cy.on("tap", "edge", (event) => {
      const edge = event.target as EdgeSingular;
      this.selectEdge(String(edge.data("edgeKey")));
    });
    this.cy.on("mouseover", "node, edge", (event) => {
      event.target.addClass("hover-label");
    });
    this.cy.on("mouseout", "node, edge", (event) => {
      event.target.removeClass("hover-label");
    });
    this.cy.on("tap", (event) => {
      if (event.target === this.cy) {
        this.clearPath();
      }
    });
    this.cy.on("dragfree", "node", () => this.captureSnapshot());
    this.cy.on("pan zoom", () => this.captureViewport());

    this.resizeObserver = new ResizeObserver(() => this.cy.resize());
    this.resizeObserver.observe(container);
  }

  get hasGraph(): boolean {
    return this.payload !== null && this.payload.nodes.length > 0;
  }

  clear(): void {
    this.captureSnapshot();
    this.cy.stop();
    this.cy.elements().remove();
    this.payload = null;
    this.graphKey = null;
    this.selectedNodeId = null;
    this.onSelection({ kind: "none" });
  }

  load(graphKey: string, payload: GraphPayload): void {
    this.captureSnapshot();
    this.graphKey = graphKey;
    this.payload = payload;
    const snapshot = this.snapshots.get(graphKey);
    this.selectedNodeId = snapshot?.selectedNodeId ?? null;

    const elements = graphElements(payload, snapshot?.positions) as ElementDefinition[];

    this.cy.batch(() => {
      this.cy.elements().remove();
      this.cy.add(elements);
    });
    this.cy.layout({ name: "preset", fit: false, animate: false }).run();

    requestAnimationFrame(() => {
      this.cy.resize();
      if (snapshot) {
        this.cy.zoom(snapshot.zoom);
        this.cy.pan(snapshot.pan);
        if (snapshot.selectedNodeId !== null) {
          this.selectNode(snapshot.selectedNodeId, false);
        }
      } else {
        this.fit();
        this.onSelection({ kind: "none" });
      }
      this.captureSnapshot();
    });
  }

  fit(): void {
    if (!this.hasGraph) return;
    this.cy.stop();
    this.cy.animate({ fit: { eles: this.cy.elements(), padding: 72 }, duration: 240 });
  }

  resetLayout(): void {
    if (!this.payload) return;
    const positions = graphLayoutPositions(this.payload);
    this.cy.batch(() => {
      for (const node of this.payload?.nodes ?? []) {
        const position = positions.get(node.id);
        if (position) this.cy.getElementById(canvasNodeId(node.id)).position(position);
      }
    });
    this.fit();
    this.captureSnapshot();
  }

  spread(): void {
    if (!this.hasGraph) return;
    const extent = this.cy.extent();
    const center = {
      x: (extent.x1 + extent.x2) / 2,
      y: (extent.y1 + extent.y2) / 2,
    };
    this.cy.batch(() => {
      this.cy.nodes().forEach((node) => {
        const position = node.position();
        node.position({
          x: center.x + (position.x - center.x) * 1.22,
          y: center.y + (position.y - center.y) * 1.22,
        });
      });
    });
    this.captureSnapshot();
  }

  focusNode(nodeIdValue: number): void {
    this.selectNode(nodeIdValue, true);
  }

  clearPath(): void {
    this.selectedNodeId = null;
    this.cy.elements().removeClass("path-muted path-active path-terminal");
    this.cy.elements().unselect();
    this.captureSnapshot();
    this.onSelection({ kind: "none" });
  }

  private selectNode(nodeIdValue: number, center: boolean): void {
    if (!this.payload) return;
    const node = this.payload.nodes.find((item) => item.id === nodeIdValue);
    if (!node) return;
    const path = this.pathTo(nodeIdValue);
    const pathNodeIds = new Set<number>(this.payload.root_ids);
    for (const step of path) {
      pathNodeIds.add(step.from.id);
      pathNodeIds.add(step.to.id);
    }
    pathNodeIds.add(nodeIdValue);

    this.cy.batch(() => {
      this.cy.elements().removeClass("path-active path-terminal").addClass("path-muted");
      for (const id of pathNodeIds) {
        this.cy.getElementById(canvasNodeId(id)).removeClass("path-muted").addClass("path-active");
      }
      for (const step of path) {
        this.cy
          .getElementById(canvasEdgeId(step.edge.id))
          .removeClass("path-muted")
          .addClass("path-active");
      }
      const selected = this.cy.getElementById(canvasNodeId(nodeIdValue));
      selected.addClass("path-terminal").select();
    });

    this.selectedNodeId = nodeIdValue;
    if (center) {
      const selected = this.cy.getElementById(canvasNodeId(nodeIdValue));
      this.cy.stop();
      this.cy.animate({ center: { eles: selected }, duration: 220 });
    }
    this.captureSnapshot();
    this.onSelection({
      kind: "node",
      node,
      path,
      incoming: this.payload.edges.filter((edge) => edge.target === nodeIdValue),
      outgoing: this.payload.edges.filter((edge) => edge.source === nodeIdValue),
    });
  }

  private selectEdge(edgeKey: string): void {
    if (!this.payload) return;
    const edge = this.payload.edges.find((item) => item.id === edgeKey);
    if (!edge) return;
    this.selectedNodeId = null;
    this.cy.elements().removeClass("path-muted path-active path-terminal").unselect();
    this.cy.getElementById(canvasEdgeId(edge.id)).select();
    this.captureSnapshot();
    this.onSelection({ kind: "edge", edge });
  }

  private pathTo(destination: number): PathStep[] {
    if (!this.payload) return [];
    const nodes = new Map(this.payload.nodes.map((node) => [node.id, node]));
    const path: PathStep[] = [];
    for (const edge of shortestRootPath(this.payload, destination)) {
      const from = nodes.get(edge.source);
      const to = nodes.get(edge.target);
      if (!from || !to) return [];
      path.push({
        edge,
        from,
        to,
        functionName: edge.function_names[0] ?? "transition",
      });
    }
    return path;
  }

  private captureSnapshot(): void {
    if (!this.graphKey || !this.payload || this.cy.nodes().empty()) return;
    const positions = new Map<number, Position>();
    this.cy.nodes().forEach((node) => {
      positions.set(Number(node.data("nodeId")), { ...node.position() });
    });
    this.rememberSnapshot(this.graphKey, {
      pan: { ...this.cy.pan() },
      zoom: this.cy.zoom(),
      positions,
      selectedNodeId: this.selectedNodeId,
    });
  }

  private captureViewport(): void {
    if (!this.graphKey || !this.payload || this.cy.nodes().empty()) return;
    const existing = this.snapshots.get(this.graphKey);
    if (!existing) return;
    this.rememberSnapshot(this.graphKey, {
      ...existing,
      pan: { ...this.cy.pan() },
      zoom: this.cy.zoom(),
      selectedNodeId: this.selectedNodeId,
    });
  }

  private rememberSnapshot(key: string, snapshot: ViewSnapshot): void {
    this.snapshots.delete(key);
    this.snapshots.set(key, snapshot);
    while (this.snapshots.size > MAX_VIEW_SNAPSHOTS) {
      const oldestKey = this.snapshots.keys().next().value as string | undefined;
      if (oldestKey === undefined) break;
      this.snapshots.delete(oldestKey);
    }
  }
}
