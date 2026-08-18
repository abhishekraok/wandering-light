import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api } from "./api";
import { BasisPanel } from "./components/BasisPanel";
import { CorpusPanel } from "./components/CorpusPanel";
import { ExpandBar, defaultPalette } from "./components/ExpandBar";
import { FunctionPicker } from "./components/FunctionPicker";
import { GraphView } from "./components/GraphView";
import { SolverPanel } from "./components/SolverPanel";
import { TrajectoryStrip } from "./components/TrajectoryStrip";
import {
  emptyGraph,
  layoutNew,
  mergeExpansion,
  mergeTrajectory,
  pathBetween,
  trajectoryOnly,
  type GraphState,
  type Point,
} from "./lib/graph-store";
import { replaceStep, truncateAt } from "./lib/trajectory";
import type {
  BasisInfo,
  CorpusTask,
  Expansion,
  SolveAttempt,
  StateView,
  Successor,
  TrajectoryResult,
} from "./types";

const DEFAULT_ROOT = "TL<int>([1, 2, 3])";
const DEFAULT_TARGET = "TL<int>([3, 5, 7])";

interface PickerState {
  index: number | null; // null = appending a new step
  anchor: { x: number; y: number };
  successors: Successor[];
  loading: boolean;
}

export function App() {
  const [basis, setBasis] = useState<BasisInfo | null>(null);
  const [basisSetId, setBasisSetId] = useState("wl-core-v1");
  const [rootText, setRootText] = useState(DEFAULT_ROOT);
  const [targetText, setTargetText] = useState(DEFAULT_TARGET);
  const [target, setTarget] = useState<StateView | null>(null);
  const [steps, setSteps] = useState<string[]>([]);
  const [trajectory, setTrajectory] = useState<TrajectoryResult | null>(null);

  const [graph, setGraph] = useState<GraphState>(emptyGraph);
  const [positions, setPositions] = useState<Map<string, Point>>(() => new Map());
  const [selected, setSelected] = useState<string | null>(null);

  const [paletteByType, setPaletteByType] = useState<Record<string, string[]>>({});
  const [depth, setDepth] = useState(2);
  const [selfLoops, setSelfLoops] = useState(true);

  const [lastExpansion, setLastExpansion] = useState<Expansion | null>(null);
  const [fitSignal, setFitSignal] = useState(0);
  const [growOrigin, setGrowOrigin] = useState<string | null>(null);
  const [attempt, setAttempt] = useState<SolveAttempt | null>(null);
  const [picker, setPicker] = useState<PickerState | null>(null);
  const [tab, setTab] = useState<"solver" | "basis" | "corpus">("solver");
  const [busy, setBusy] = useState(false);
  const [expanding, setExpanding] = useState(false);
  const [solving, setSolving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const expandSeq = useRef(0);

  useEffect(() => {
    api.basis(basisSetId).then(setBasis).catch((err: Error) => setError(err.message));
  }, [basisSetId]);

  // Re-run the whole trajectory whenever the root, the steps or the basis
  // change. Debounced because the root is a text box someone is typing into.
  useEffect(() => {
    const timer = window.setTimeout(() => {
      setBusy(true);
      api
        .trajectory(rootText, steps, basisSetId)
        .then((result) => {
          setTrajectory(result);
          setError(null);
        })
        .catch((err: Error) => {
          setTrajectory(null);
          setError(err.message);
        })
        .finally(() => setBusy(false));
    }, 220);
    return () => window.clearTimeout(timer);
  }, [rootText, steps, basisSetId]);

  // Every state the trajectory passes through belongs on the canvas, so editing
  // steps draws immediately rather than waiting for an expansion.
  useEffect(() => {
    if (trajectory === null) return;
    setGraph((current) => mergeTrajectory(current, trajectory));
  }, [trajectory]);

  const anchor = trajectory?.root.wire ?? null;

  // One place assigns positions, and only to nodes that lack one.
  useEffect(() => {
    if (anchor === null) return;
    setPositions((current) => layoutNew(graph, anchor, current, growOrigin));
  }, [graph, anchor, growOrigin]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      if (!targetText.trim()) {
        setTarget(null);
        return;
      }
      api
        .parseState(targetText)
        .then(setTarget)
        .catch(() => setTarget(null));
    }, 220);
    return () => window.clearTimeout(timer);
  }, [targetText]);

  const currentState: StateView | null = useMemo(() => {
    if (trajectory === null) return null;
    const last = trajectory.steps[trajectory.steps.length - 1];
    if (!last) return trajectory.root;
    return last.ok ? last.state : null;
  }, [trajectory]);

  // Expansion anchors on the selected node, falling back to where the
  // trajectory currently stands.
  const expandFrom = useMemo(() => {
    if (selected !== null) return graph.nodes.get(selected) ?? null;
    if (currentState === null) return null;
    return { wire: currentState.wire, label: currentState.label, type: currentState.type };
  }, [selected, graph, currentState]);

  const compatible = useMemo(
    () => (basis === null || expandFrom === null
      ? []
      : basis.functions.filter((fn) => fn.input_type === expandFrom.type)),
    [basis, expandFrom],
  );

  const palette = useMemo(() => {
    if (expandFrom === null) return [];
    const stored = paletteByType[expandFrom.type];
    return stored ?? defaultPalette(compatible);
  }, [paletteByType, expandFrom, compatible]);

  const setPalette = useCallback(
    (names: string[]) => {
      if (expandFrom === null) return;
      setPaletteByType((current) => ({ ...current, [expandFrom.type]: names }));
    },
    [expandFrom],
  );

  const stateBefore = useCallback(
    (index: number): string | null => {
      if (trajectory === null) return null;
      if (index === 0) return trajectory.root.wire;
      const previous = trajectory.steps[index - 1];
      return previous?.ok ? (previous.state?.wire ?? null) : null;
    },
    [trajectory],
  );

  const openPicker = useCallback(
    (index: number | null, event: React.MouseEvent) => {
      const wire = index === null ? currentState?.wire : stateBefore(index);
      if (!wire) return;
      const rect = (event.currentTarget as HTMLElement).getBoundingClientRect();
      setPicker({
        index,
        anchor: { x: rect.right + 8, y: rect.top },
        successors: [],
        loading: true,
      });
      api
        .successors(wire, basisSetId)
        .then((result) =>
          setPicker((current) =>
            current === null
              ? null
              : { ...current, successors: result.successors, loading: false },
          ),
        )
        .catch((err: Error) => {
          setError(err.message);
          setPicker(null);
        });
    },
    [basisSetId, currentState, stateBefore],
  );

  const pick = useCallback((name: string) => {
    setPicker((current) => {
      if (current === null) return null;
      setSteps((existing) =>
        current.index === null ? [...existing, name] : replaceStep(existing, current.index, name),
      );
      return null;
    });
  }, []);

  const expand = useCallback(async () => {
    if (expandFrom === null || palette.length === 0) return;
    const sequence = ++expandSeq.current;
    setExpanding(true);
    try {
      const result = await api.expand(expandFrom.wire, basisSetId, palette, {
        max_depth: depth,
        max_states: 400,
        max_transitions: 200_000,
        include_self_loops: selfLoops,
      });
      if (sequence !== expandSeq.current) return; // a newer expansion won
      setLastExpansion(result);
      setGrowOrigin(expandFrom.wire);
      setGraph((current) => mergeExpansion(current, result));
      // New nodes land outside the viewport otherwise, which reads as "expand
      // did nothing" -- the expansion is an explicit action, so re-frame it.
      setFitSignal((value) => value + 1);
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setExpanding(false);
    }
  }, [basisSetId, depth, expandFrom, palette, selfLoops]);

  const clearCanvas = useCallback(() => {
    setGraph(trajectoryOnly(trajectory));
    setPositions(new Map());
    setSelected(null);
    setLastExpansion(null);
    setGrowOrigin(null);
    setFitSignal((value) => value + 1);
  }, [trajectory]);

  const walkToSelected = useCallback(() => {
    if (selected === null || currentState === null) return;
    const route = pathBetween(graph, currentState.wire, selected).map((edge) => edge.fn);
    if (route.length > 0) setSteps((existing) => [...existing, ...route]);
  }, [graph, selected, currentState]);

  const moveNode = useCallback((wire: string, position: Point) => {
    setPositions((current) => new Map(current).set(wire, position));
  }, []);

  const runSolver = useCallback(
    (options: { solver: string; budget: number; max_depth: number }) => {
      if (trajectory === null || target === null) return;
      setSolving(true);
      api
        .solve(trajectory.root.wire, target.wire, basisSetId, options)
        .then((result) => {
          setAttempt(result);
          setError(null);
        })
        .catch((err: Error) => setError(err.message))
        .finally(() => setSolving(false));
    },
    [basisSetId, target, trajectory],
  );

  const loadTask = useCallback((task: CorpusTask) => {
    setRootText(task.input);
    setTargetText(task.output);
    setSteps([]);
    setAttempt(null);
    setGraph(emptyGraph());
    setPositions(new Map());
    setSelected(null);
    setLastExpansion(null);
  }, []);

  const selectedNode = selected === null ? null : (graph.nodes.get(selected) ?? null);

  return (
    <div className="app">
      <header className="topbar">
        <h1>Wandering Light</h1>
        <div className="field">
          <label>basis</label>
          <select
            value={basisSetId}
            onChange={(event) => setBasisSetId(event.target.value)}
            style={{ width: 220 }}
          >
            {(basis?.available ?? []).map((entry) => (
              <option key={entry.id} value={entry.id}>
                {entry.id}
                {entry.alias ? ` (${entry.alias})` : ""}
              </option>
            ))}
          </select>
        </div>
        <div className="field grow">
          <label>root</label>
          <input
            type="text"
            value={rootText}
            onChange={(event) => setRootText(event.target.value)}
            spellCheck={false}
          />
        </div>
        <div className="field grow">
          <label>target</label>
          <input
            type="text"
            value={targetText}
            onChange={(event) => setTargetText(event.target.value)}
            spellCheck={false}
            placeholder="optional"
          />
        </div>
        {error && <span className="error small">{error}</span>}
      </header>

      <div className="columns">
        <aside className="pane-left">
          <div className="section">
            <h2>Trajectory</h2>
            <TrajectoryStrip
              trajectory={trajectory}
              target={target}
              busy={busy}
              onEditStep={(index, event) => openPicker(index, event)}
              onAppend={(event) => openPicker(null, event)}
              onDeleteFrom={(index) => setSteps((existing) => truncateAt(existing, index))}
            />
            {steps.length > 0 && (
              <div className="row" style={{ marginTop: 8 }}>
                <button className="ghost" onClick={() => setSteps([])}>
                  clear steps
                </button>
                <span className="muted small mono">{steps.join(" → ")}</span>
              </div>
            )}
          </div>
          <div className="section">
            <h2>Canvas</h2>
            <div className="row spread">
              <span className="muted small">
                {graph.nodes.size} nodes · {graph.edges.size} edges
              </span>
              <button onClick={clearCanvas} disabled={graph.nodes.size === 0}>
                clear canvas
              </button>
            </div>
            <div className="muted small" style={{ marginTop: 6 }}>
              The canvas keeps everything you visit or expand, merged by state, so
              involutions and self-loops close on themselves.
            </div>
          </div>
        </aside>

        <main className="pane-center">
          <div className="canvas-area">
            <GraphView
              graph={graph}
              positions={positions}
              anchor={anchor}
              selected={selected}
              targetWire={target?.wire ?? null}
              onSelect={setSelected}
              onMove={moveNode}
              fitSignal={fitSignal}
            />
          </div>
          {expandFrom !== null && (
            <div className="canvas-bar">
              <ExpandBar
                label={expandFrom.label}
                type={expandFrom.type}
                depth={depth}
                compatible={compatible}
                palette={palette}
                selfLoops={selfLoops}
                busy={expanding}
                onPaletteChange={setPalette}
                onDepthChange={setDepth}
                onSelfLoopsChange={setSelfLoops}
                onExpand={expand}
              />
              <div className="row" style={{ marginTop: 6 }}>
                <span className="muted small">
                  {selectedNode === null
                    ? "expanding from the trajectory's current state — click a node to anchor elsewhere"
                    : "selected node"}
                </span>
                <span style={{ flex: 1 }} />
                {selectedNode !== null && (
                  <>
                    <button onClick={walkToSelected}>walk here</button>
                    <button onClick={() => setTargetText(selectedNode.wire)}>set as target</button>
                    <button
                      onClick={() => {
                        setRootText(selectedNode.wire);
                        setSteps([]);
                      }}
                    >
                      make root
                    </button>
                    <button className="ghost" onClick={() => setSelected(null)}>
                      deselect
                    </button>
                  </>
                )}
              </div>
            </div>
          )}
        </main>

        <aside className="pane-right">
          <div className="tabs">
            {(["solver", "basis", "corpus"] as const).map((name) => (
              <button
                key={name}
                className={`tab${tab === name ? " active" : ""}`}
                onClick={() => setTab(name)}
              >
                {name}
              </button>
            ))}
          </div>
          {tab === "solver" && (
            <SolverPanel
              root={trajectory?.root ?? null}
              target={target}
              attempt={attempt}
              expansion={lastExpansion}
              yourSteps={steps.length}
              running={solving}
              onSolve={runSolver}
              onUseAttempt={(functions) => setSteps(functions)}
            />
          )}
          {tab === "basis" && (
            <BasisPanel
              basis={basis}
              expansion={lastExpansion}
              palette={palette}
              onPaletteChange={(names) => setPalette(names ?? compatible.map((fn) => fn.name))}
            />
          )}
          {tab === "corpus" && <CorpusPanel onLoadTask={loadTask} />}
        </aside>
      </div>

      {picker !== null && (
        <FunctionPicker
          anchor={picker.anchor}
          successors={picker.successors}
          loading={picker.loading}
          current={picker.index === null ? null : (steps[picker.index] ?? null)}
          onPick={pick}
          onClose={() => setPicker(null)}
        />
      )}
    </div>
  );
}
