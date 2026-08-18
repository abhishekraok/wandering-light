import type { Step, StateView, TrajectoryResult } from "../types";
import { StateChip } from "./StateChip";

/**
 * The trajectory as a column of clickable edges.
 *
 * Clicking a function opens the picker for that edge; replacing it drops every
 * step after it, because those were computed from a state that no longer
 * exists.
 */
export function TrajectoryStrip({
  trajectory,
  target,
  onEditStep,
  onAppend,
  onDeleteFrom,
  busy,
}: {
  trajectory: TrajectoryResult | null;
  target: StateView | null;
  onEditStep: (index: number, event: React.MouseEvent) => void;
  onAppend: (event: React.MouseEvent) => void;
  onDeleteFrom: (index: number) => void;
  busy: boolean;
}) {
  if (trajectory === null) {
    return <div className="muted small">Enter a root state to begin.</div>;
  }
  const steps: Step[] = trajectory.steps;
  const last = steps.length > 0 ? steps[steps.length - 1] : null;
  const reached =
    target !== null &&
    (last === null ? trajectory.root.wire === target.wire : last.state?.wire === target.wire);
  const broken = last !== null && !last.ok;

  return (
    <div>
      <StateChip state={trajectory.root} title="root" />
      {steps.map((step, index) => (
        <div key={`${index}-${step.function}`}>
          <div className="step">
            <div className="rail" />
            <button
              className={`step-fn${step.ok ? "" : " failed"}`}
              onClick={(event) => onEditStep(index, event)}
              disabled={busy}
              title="change this function"
            >
              <span className="idx">{index + 1}.</span>
              {step.function}
              <span className="grow" style={{ flex: 1 }} />
              <span className="muted small">▾</span>
            </button>
            <button
              className="ghost"
              onClick={() => onDeleteFrom(index)}
              disabled={busy}
              title="remove this step and everything after it"
            >
              ✕
            </button>
          </div>
          {step.ok ? (
            <StateChip state={step.state} />
          ) : (
            <div className="state-chip error">{step.error}</div>
          )}
        </div>
      ))}
      <div className="row" style={{ marginTop: 6 }}>
        <button onClick={onAppend} disabled={busy || broken}>
          + add step
        </button>
        {target !== null && (
          <span className={`badge ${reached ? "good" : "bad"}`}>
            {reached ? `reaches target in ${steps.length}` : "target not reached"}
          </span>
        )}
      </div>
    </div>
  );
}
