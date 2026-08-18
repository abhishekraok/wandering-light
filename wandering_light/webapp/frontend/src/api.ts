import type {
  BasisInfo,
  CorpusSummary,
  CorpusTask,
  Expansion,
  SolveAttempt,
  StateView,
  SuccessorResult,
  TrajectoryResult,
} from "./types";

async function post<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(detail.detail ?? response.statusText);
  }
  return response.json() as Promise<T>;
}

async function get<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(detail.detail ?? response.statusText);
  }
  return response.json() as Promise<T>;
}

export const api = {
  basis: (basisSetId?: string) =>
    get<BasisInfo>(`/api/basis${basisSetId ? `?basis_set_id=${basisSetId}` : ""}`),

  parseState: (state: string) => post<StateView>("/api/state", { state }),

  trajectory: (state: string, steps: string[], basisSetId: string) =>
    post<TrajectoryResult>("/api/trajectory", {
      state,
      steps,
      basis_set_id: basisSetId,
    }),

  successors: (state: string, basisSetId: string) =>
    post<SuccessorResult>("/api/successors", { state, basis_set_id: basisSetId }),

  expand: (
    state: string,
    basisSetId: string,
    functions: string[] | null,
    budgets: { max_depth: number; max_states: number; max_transitions: number },
  ) =>
    post<Expansion>("/api/expand", {
      state,
      basis_set_id: basisSetId,
      functions,
      ...budgets,
    }),

  solve: (
    state: string,
    target: string,
    basisSetId: string,
    options: { solver: string; budget: number; max_depth: number },
  ) =>
    post<SolveAttempt>("/api/solve", {
      state,
      target,
      basis_set_id: basisSetId,
      ...options,
    }),

  corpora: () => get<{ corpora: CorpusSummary[] }>("/api/corpora"),

  corpusTasks: (name: string, split: string, limit: number, distance: number | null) =>
    get<{ tasks: CorpusTask[] }>(
      `/api/corpora/${encodeURIComponent(name)}/tasks?split=${split}&limit=${limit}` +
        (distance === null ? "" : `&distance=${distance}`),
    ),
};
