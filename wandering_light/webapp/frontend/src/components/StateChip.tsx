import type { StateView } from "../types";

export function StateChip({
  state,
  variant,
  title,
}: {
  state: StateView | null;
  variant?: "target";
  title?: string;
}) {
  if (state === null) {
    return <div className="state-chip muted">—</div>;
  }
  return (
    <div className={`state-chip${variant ? ` ${variant}` : ""}`} title={title ?? state.label}>
      <span className="type">{state.type}</span>
      {state.label.replace(/^TL<[^>]+>\(/, "").replace(/\)$/, "")}
    </div>
  );
}
