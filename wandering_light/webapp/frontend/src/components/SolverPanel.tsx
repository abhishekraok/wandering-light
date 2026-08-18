import { useState } from "react";

import type { Expansion, SolveAttempt, StateView } from "../types";

/**
 * What the solver is up against, and what you managed yourself.
 *
 * The attempt is hidden until asked for: seeing BFS's answer first turns the
 * task into reading, and the point of the panel is to feel the search before
 * being told the route.
 */
export function SolverPanel({
  root,
  target,
  attempt,
  expansion,
  yourSteps,
  running,
  onSolve,
  onUseAttempt,
}: {
  root: StateView | null;
  target: StateView | null;
  attempt: SolveAttempt | null;
  expansion: Expansion | null;
  yourSteps: number;
  running: boolean;
  onSolve: (options: { solver: string; budget: number; max_depth: number }) => void;
  onUseAttempt: (functions: string[]) => void;
}) {
  const [solver, setSolver] = useState("bfs");
  const [budget, setBudget] = useState(5000);
  const [maxDepth, setMaxDepth] = useState(3);
  const [revealed, setRevealed] = useState(false);

  const ready = root !== null && target !== null;

  return (
    <div className="section">
      <h2>Solver</h2>
      {!ready && <div className="muted small">Set a root and a target to run a solver.</div>}
      <div className="row">
        <select value={solver} onChange={(event) => setSolver(event.target.value)}>
          <option value="bfs">bfs</option>
          <option value="random">random</option>
        </select>
        <label className="muted small">depth</label>
        <input
          type="text"
          value={maxDepth}
          onChange={(event) => setMaxDepth(Number(event.target.value) || 1)}
          style={{ width: 48 }}
        />
        <label className="muted small">budget</label>
        <input
          type="text"
          value={budget}
          onChange={(event) => setBudget(Number(event.target.value) || 1)}
          style={{ width: 76 }}
        />
      </div>
      <div className="row">
        <button
          className="primary"
          disabled={!ready || running}
          onClick={() => {
            setRevealed(false);
            onSolve({ solver, budget, max_depth: maxDepth });
          }}
        >
          {running ? "searching…" : "▶ run solver"}
        </button>
        <span className="muted small">your trajectory: {yourSteps} step(s)</span>
      </div>

      {attempt !== null && (
        <div className="row" style={{ display: "block", marginTop: 8 }}>
          <div className="row spread">
            <span className={attempt.success ? "good" : "error"}>
              {attempt.success ? "solved" : "failed"}
            </span>
            <span className="muted small mono">{attempt.elapsed_seconds}s</span>
          </div>
          <div className="muted small mono">{attempt.solver}</div>
          {attempt.success ? (
            revealed ? (
              <>
                <div className="mono small" style={{ margin: "6px 0" }}>
                  {attempt.functions.join(" → ")}
                </div>
                <button onClick={() => onUseAttempt(attempt.functions)}>
                  load into trajectory
                </button>
              </>
            ) : (
              <div className="row" style={{ marginTop: 6 }}>
                <span className="muted small">
                  found a {attempt.functions.length}-step path
                </span>
                <button onClick={() => setRevealed(true)}>reveal</button>
              </div>
            )
          ) : (
            <div className="small error">{attempt.error ?? "no path within budget"}</div>
          )}
        </div>
      )}

      {expansion !== null && (
        <>
          <h2 style={{ marginTop: 14 }}>Search cost from this root</h2>
          <table className="grid">
            <thead>
              <tr>
                <th>depth</th>
                <th>states</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {Object.entries(expansion.stats.by_depth).map(([depth, count]) => {
                const total = expansion.stats.nodes || 1;
                return (
                  <tr key={depth}>
                    <td>{depth}</td>
                    <td>{count}</td>
                    <td style={{ width: "55%" }}>
                      <div className="bar">
                        <span style={{ width: `${(count / total) * 100}%` }} />
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div className="muted small" style={{ marginTop: 6 }}>
            {expansion.stats.attempted_transitions.toLocaleString()} transitions tried,{" "}
            {expansion.stats.failed_transitions.toLocaleString()} failed,{" "}
            {expansion.stats.skipped_self_loops.toLocaleString()} self-loops ·{" "}
            {expansion.stats.elapsed_seconds}s
            {!expansion.stats.complete && (
              <span className="warn"> · stopped at {expansion.stats.stop_reason}</span>
            )}
          </div>
        </>
      )}
    </div>
  );
}
