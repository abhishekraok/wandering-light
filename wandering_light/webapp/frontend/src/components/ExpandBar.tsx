import { useState } from "react";

import type { BasisFunction } from "../types";

/** A small, involution-rich default: two-cycles and no-ops you can see. */
export const INT_PRESET = ["inc", "dec", "double", "half", "neg", "abs", "square", "mod2"];

export function defaultPalette(compatible: BasisFunction[]): string[] {
  const names = compatible.map((fn) => fn.name);
  const preset = INT_PRESET.filter((name) => names.includes(name));
  // A first expansion over a hundred-odd functions is a wall of nodes; a
  // handful is a graph you can read. Everything stays one click away.
  return preset.length > 0 ? preset : names.slice(0, 8);
}

/**
 * Expansion, anchored on whichever node is selected.
 *
 * The palette is part of the question being asked -- "what can these functions
 * reach from here" -- so it lives next to the button rather than in a settings
 * panel somewhere else.
 */
export function ExpandBar({
  label,
  type,
  depth,
  compatible,
  palette,
  selfLoops,
  busy,
  onPaletteChange,
  onDepthChange,
  onSelfLoopsChange,
  onExpand,
}: {
  label: string;
  type: string;
  depth: number;
  compatible: BasisFunction[];
  palette: string[];
  selfLoops: boolean;
  busy: boolean;
  onPaletteChange: (palette: string[]) => void;
  onDepthChange: (depth: number) => void;
  onSelfLoopsChange: (value: boolean) => void;
  onExpand: () => void;
}) {
  const [open, setOpen] = useState(false);

  return (
    <div className="expand-bar">
      <div className="row spread">
        <div className="row">
          <span className="state-chip" title={label}>
            {label}
          </span>
          <button onClick={() => setOpen(!open)} title="choose which functions may be applied">
            {palette.length} of {compatible.length} {type} fns ▾
          </button>
          <label className="muted small">depth</label>
          <input
            type="text"
            value={depth}
            onChange={(event) => onDepthChange(Number(event.target.value) || 1)}
            style={{ width: 42 }}
          />
          <label className="muted small" title="a function that leaves this state unchanged">
            <input
              type="checkbox"
              checked={selfLoops}
              onChange={(event) => onSelfLoopsChange(event.target.checked)}
            />{" "}
            self-loops
          </label>
          <button className="primary" onClick={onExpand} disabled={busy || palette.length === 0}>
            {busy ? "expanding…" : "expand"}
          </button>
        </div>
      </div>

      {open && (
        <div className="palette">
          <div className="row" style={{ marginBottom: 6 }}>
            <button onClick={() => onPaletteChange(compatible.map((fn) => fn.name))}>
              all {compatible.length}
            </button>
            <button onClick={() => onPaletteChange(defaultPalette(compatible))}>preset</button>
            <button onClick={() => onPaletteChange([])}>none</button>
            <span className="muted small">
              functions taking <code>{type}</code>
            </span>
          </div>
          <div className="palette-grid">
            {compatible.map((fn) => {
              const checked = palette.includes(fn.name);
              return (
                <label key={fn.name} className="palette-item" title={fn.code}>
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() =>
                      onPaletteChange(
                        checked
                          ? palette.filter((name) => name !== fn.name)
                          : [...palette, fn.name],
                      )
                    }
                  />{" "}
                  {fn.name}
                  <span className="muted"> → {fn.output_type}</span>
                </label>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
