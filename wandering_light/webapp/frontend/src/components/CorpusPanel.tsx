import { useEffect, useState } from "react";

import { api } from "../api";
import type { CorpusSummary, CorpusTask } from "../types";

/** Pull a real task out of a corpus and load it as root and target. */
export function CorpusPanel({
  onLoadTask,
}: {
  onLoadTask: (task: CorpusTask) => void;
}) {
  const [corpora, setCorpora] = useState<CorpusSummary[]>([]);
  const [name, setName] = useState<string>("");
  const [split, setSplit] = useState<string>("");
  const [distance, setDistance] = useState<string>("");
  const [tasks, setTasks] = useState<CorpusTask[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api
      .corpora()
      .then((result) => {
        setCorpora(result.corpora);
        const first = result.corpora[0];
        if (first) {
          setName(first.name);
          setSplit(first.splits.includes("test") ? "test" : first.splits[0]);
        }
      })
      .catch((err: Error) => setError(err.message));
  }, []);

  const current = corpora.find((entry) => entry.name === name) ?? null;
  const unavailable = current !== null && current.missing_splits.includes(split);

  useEffect(() => {
    if (!name || !split || unavailable) {
      setTasks([]);
      return;
    }
    setLoading(true);
    api
      .corpusTasks(name, split, 60, distance ? Number(distance) : null)
      .then((result) => {
        setTasks(result.tasks);
        setError(null);
      })
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  }, [name, split, distance, unavailable]);

  return (
    <div className="section">
      <h2>Corpus</h2>
      {corpora.length === 0 && !error && <div className="muted small">No local corpus.</div>}
      {error && <div className="error small">{error}</div>}
      {current !== null && (
        <>
          <div className="row">
            <select value={name} onChange={(event) => setName(event.target.value)}>
              {corpora.map((entry) => (
                <option key={entry.name} value={entry.name}>
                  {entry.name} ({entry.tasks.toLocaleString()})
                </option>
              ))}
            </select>
            <select value={split} onChange={(event) => setSplit(event.target.value)}>
              {current.splits.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
            <select value={distance} onChange={(event) => setDistance(event.target.value)}>
              <option value="">any d</option>
              {Object.keys(current.distances).map((value) => (
                <option key={value} value={value}>
                  d{value}
                </option>
              ))}
            </select>
          </div>
          {unavailable && (
            <div className="warn small" style={{ marginTop: 6 }}>
              Payload for {split} is not downloaded. Fetch it with corpus_hub.fetch_corpus.
            </div>
          )}
          {loading && <div className="muted small">loading…</div>}
          <table className="grid">
            <thead>
              <tr>
                <th>d</th>
                <th>input → output</th>
              </tr>
            </thead>
            <tbody>
              {tasks.map((task) => (
                <tr
                  key={task.task_id}
                  className="clickable"
                  onClick={() => onLoadTask(task)}
                  title={`${task.input_label}\n→ ${task.output_label}\nwitness: ${task.witness.join(", ")}`}
                >
                  <td>{task.distance}</td>
                  <td
                    style={{
                      maxWidth: 260,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {task.input_label} → {task.output_label}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
