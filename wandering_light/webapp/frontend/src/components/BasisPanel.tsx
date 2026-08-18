import { useMemo, useState } from "react";

import type { BasisInfo, Expansion } from "../types";

/**
 * The palette, and how much of it the last expansion actually used.
 *
 * "Idle" is the interesting column: a function that never produced an edge
 * from these states is either inapplicable by type or redundant here, and both
 * are answers to "is the basis expressive enough for this corner".
 */
export function BasisPanel({
  basis,
  expansion,
  palette,
  onPaletteChange,
}: {
  basis: BasisInfo | null;
  expansion: Expansion | null;
  palette: string[] | null;
  onPaletteChange: (palette: string[] | null) => void;
}) {
  const [query, setQuery] = useState("");

  const rows = useMemo(() => {
    if (basis === null) return [];
    const needle = query.trim().toLowerCase();
    return basis.functions
      .filter(
        (fn) =>
          !needle ||
          fn.name.toLowerCase().includes(needle) ||
          fn.input_type.includes(needle) ||
          fn.output_type.includes(needle),
      )
      .map((fn) => ({
        ...fn,
        edges: expansion?.function_edges[fn.name] ?? 0,
        selected: palette === null || palette.includes(fn.name),
      }));
  }, [basis, query, expansion, palette]);

  if (basis === null) return <div className="section muted">Loading basis…</div>;

  const reachedTypes = expansion === null ? [] : Object.entries(expansion.types);

  return (
    <div className="section">
      <h2>Basis · {basis.basis_set_id}</h2>
      <div className="row">
        <input
          type="text"
          placeholder="filter by name or type…"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
      </div>
      <div className="row">
        <button onClick={() => onPaletteChange(null)} disabled={palette === null}>
          use whole basis
        </button>
        <button
          onClick={() => onPaletteChange(rows.map((row) => row.name))}
          disabled={rows.length === 0}
        >
          restrict to filtered ({rows.length})
        </button>
      </div>

      {reachedTypes.length > 0 && (
        <>
          <h2 style={{ marginTop: 14 }}>Types reached</h2>
          <div className="row small mono" style={{ flexWrap: "wrap", gap: 6 }}>
            {reachedTypes
              .sort((a, b) => b[1] - a[1])
              .map(([type, count]) => (
                <span className="badge" key={type}>
                  {type} · {count}
                </span>
              ))}
          </div>
          <div className="muted small" style={{ marginTop: 6 }}>
            {expansion?.idle_functions.length ?? 0} of {basis.functions.length} functions
            produced no edge here.
          </div>
        </>
      )}

      <h2 style={{ marginTop: 14 }}>Functions</h2>
      <table className="grid">
        <thead>
          <tr>
            <th>name</th>
            <th>type</th>
            <th title="edges produced in the last expansion">used</th>
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 200).map((row) => (
            <tr key={row.name} title={row.code}>
              <td style={{ opacity: row.selected ? 1 : 0.45 }}>{row.name}</td>
              <td className="muted">
                {row.input_type}→{row.output_type}
              </td>
              <td className={row.edges > 0 ? "good" : "muted"}>{row.edges || "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
