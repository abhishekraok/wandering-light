import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api } from "./api";
import { BasisPanel } from "./components/BasisPanel";
import { CorpusPanel } from "./components/CorpusPanel";
import { FunctionPicker } from "./components/FunctionPicker";
import { GraphView } from "./components/GraphView";
import { SolverPanel } from "./components/SolverPanel";
import { StateChip } from "./components/StateChip";
import { TrajectoryStrip } from "./components/TrajectoryStrip";
import { shortestPathFunctions } from "./lib/layout";
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
  const [expansion, setExpansion] = useState<Expansion | null>(null);
  const [selectedNode, setSelectedNode] = useState<number | null>(null);
  const [palette, setPalette] = useState<string[] | null>(null);
  const [attempt, setAttempt] = useState<SolveAttempt | null>(null);
  const [picker, setPicker] = useState<PickerState | null>(null);
  const [tab, setTab] = useState<"solver" | "basis" | "corpus">("solver");
  const [depth, setDepth] = useState(2);
  const [busy, setBusy] = useState(false);
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

  const pick = useCallback(
    (name: string) => {
      setPicker((current) => {
        if (current === null) return null;
        setSteps((existing) =>
          current.index === null
            ? [...existing, name]
            : replaceStep(existing, current.index, name),
        );
        return null;
      });
    },
    [],
  );

  const expandHere = useCallback(async () => {
    const wire = currentState?.wire;
    if (!wire) return;
    const sequence = ++expandSeq.current;
    setBusy(true);
    try {
      const result = await api.expand(wire, basisSetId, palette, {
        max_depth: depth,
        max_states: 300,
        max_transitions: 200_000,
      });
      if (sequence !== expandSeq.current) return; // a newer expansion won
      setExpansion(result);
      setSelectedNode(null);
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }, [basisSetId, currentState, depth, palette]);

  const walkToSelected = useCallback(() => {
    if (expansion === null || selectedNode === null) return;
    const names = shortestPathFunctions(
      expansion.edges,
      expansion.stats.root_id,
      selectedNode,
    );
    if (names.length > 0) setSteps((existing) => [...existing, ...names]);
  }, [expansion, selectedNode]);

  const selectedState = useMemo(() => {
    if (expansion === null || selectedNode === null) return null;
    return expansion.nodes.find((node) => node.node_id === selectedNode) ?? null;
  }, [expansion, selectedNode]);

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
    setExpansion(null);
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <h1>🌱 Wandering Light</h1>
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
                  clear
                </button>
                <span className="muted small mono">{steps.join(" → ")}</span>
              </div>
            )}
          </div>
          <div className="section">
            <h2>Expand from here</h2>
            <div className="row">
              <label className="muted small">depth</label>
              <input
                type="text"
                value={depth}
                onChange={(event) => setDepth(Number(event.target.value) || 1)}
                style={{ width: 48 }}
              />
              <button className="primary" onClick={expandHere} disabled={busy || !currentState}>
                🌐 expand
              </button>
            </div>
            <div className="muted small" style={{ marginTop: 6 }}>
              {palette === null
                ? "whole basis"
                : `${palette.length} function(s) selected in the Basis tab`}
            </div>
          </div>
        </aside>

        <main className="pane-center">
          <GraphView
            expansion={expansion}
            selected={selectedNode}
            targetWire={target?.wire ?? null}
            onSelect={setSelectedNode}
          />
          {selectedState !== null && (
            <div
              className="section"
              style={{ position: "absolute", bottom: 0, left: 0, right: 0, background: "var(--panel)" }}
            >
              <div className="row spread">
                <StateChip state={selectedState} />
                <div className="row">
                  <span className="muted small">depth {selectedState.depth}</span>
                  <button onClick={walkToSelected}>walk here</button>
                  <button onClick={() => setTargetText(selectedState.wire)}>set as target</button>
                  <button
                    onClick={() => {
                      setRootText(selectedState.wire);
                      setSteps([]);
                    }}
                  >
                    make root
                  </button>
                </div>
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
              expansion={expansion}
              yourSteps={steps.length}
              running={solving}
              onSolve={runSolver}
              onUseAttempt={(functions) => setSteps(functions)}
            />
          )}
          {tab === "basis" && (
            <BasisPanel
              basis={basis}
              expansion={expansion}
              palette={palette}
              onPaletteChange={setPalette}
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
