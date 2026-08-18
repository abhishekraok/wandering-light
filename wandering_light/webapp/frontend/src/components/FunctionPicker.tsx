import { useEffect, useMemo, useRef, useState } from "react";

import type { Successor } from "../types";

/**
 * Choose the function for one edge.
 *
 * Every option shows what it would actually produce from this state, so
 * picking an edge is an observation rather than a guess -- including the
 * options that fail here, and the ones that change nothing.
 */
export function FunctionPicker({
  anchor,
  successors,
  loading,
  current,
  onPick,
  onClose,
}: {
  anchor: { x: number; y: number };
  successors: Successor[];
  loading: boolean;
  current: string | null;
  onPick: (name: string) => void;
  onClose: () => void;
}) {
  const [query, setQuery] = useState("");
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    function onClick(event: MouseEvent) {
      if (box.current && !box.current.contains(event.target as Node)) onClose();
    }
    document.addEventListener("keydown", onKey);
    // Deferred so the click that opened the picker does not close it.
    const timer = window.setTimeout(() => document.addEventListener("mousedown", onClick), 0);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("mousedown", onClick);
      window.clearTimeout(timer);
    };
  }, [onClose]);

  const matches = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const filtered = needle
      ? successors.filter((s) => s.function.toLowerCase().includes(needle))
      : successors;
    // Usable options first: a function that fails or no-ops here is still worth
    // showing, but it is not what someone is reaching for.
    return [...filtered].sort((a, b) => {
      const rank = (s: Successor) => (s.ok && !s.self_loop ? 0 : s.ok ? 1 : 2);
      return rank(a) - rank(b) || a.function.localeCompare(b.function);
    });
  }, [successors, query]);

  const top = Math.min(anchor.y, window.innerHeight - 320);
  const left = Math.min(anchor.x, window.innerWidth - 400);

  return (
    <div className="picker" ref={box} style={{ top, left }}>
      <input
        type="text"
        autoFocus
        placeholder="filter functions…"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && matches.length > 0) onPick(matches[0].function);
        }}
      />
      {loading && <div className="muted small">applying every compatible function…</div>}
      {!loading && matches.length === 0 && (
        <div className="muted small">No compatible function for this state's type.</div>
      )}
      {matches.map((option) => (
        <button
          key={option.function}
          className={`option${option.ok ? "" : " dead"}${option.self_loop ? " loop" : ""}`}
          onClick={() => onPick(option.function)}
          title={option.error ?? option.state?.label ?? ""}
        >
          {option.function === current ? "● " : "　"}
          {option.function}{" "}
          <span className="result">
            {option.ok
              ? option.self_loop
                ? "→ unchanged"
                : `→ ${option.state?.label ?? ""}`
              : `✗ ${option.error}`}
          </span>
        </button>
      ))}
    </div>
  );
}
