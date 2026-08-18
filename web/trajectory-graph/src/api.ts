import type {
  CorpusSource,
  ExpansionGraphRequest,
  ExpansionGraphResponse,
  FacetsResponse,
  IndexState,
  SourcesResponse,
  TaskDetail,
  TaskQuery,
  TasksResponse,
  WitnessGraphRequest,
  WitnessGraphResponse,
} from "./types";

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export function apiErrorMessage(status: number, statusText: string, payload: unknown): string {
  if (
    typeof payload === "object" &&
    payload !== null &&
    "detail" in payload &&
    typeof payload.detail === "string"
  ) {
    return payload.detail;
  }
  return `${status} ${statusText}`.trim();
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body !== undefined) {
    headers.set("Content-Type", "application/json");
  }
  headers.set("Accept", "application/json");

  const response = await fetch(path, { ...init, headers });
  if (!response.ok) {
    let payload: unknown;
    try {
      payload = await response.json();
    } catch {
      payload = undefined;
    }
    throw new ApiError(
      response.status,
      apiErrorMessage(response.status, response.statusText, payload),
    );
  }
  return (await response.json()) as T;
}

function sourcePath(sourceId: string): string {
  return `/api/v1/sources/${encodeURIComponent(sourceId)}`;
}

export const api = {
  sources(signal?: AbortSignal): Promise<SourcesResponse> {
    return request("/api/v1/sources", { signal });
  },

  startIndex(sourceId: string, signal?: AbortSignal): Promise<IndexState> {
    return request(`${sourcePath(sourceId)}/index`, { method: "POST", signal });
  },

  fetchSource(sourceId: string, signal?: AbortSignal): Promise<CorpusSource> {
    return request(`${sourcePath(sourceId)}/fetch`, { method: "POST", signal });
  },

  facets(sourceId: string, signal?: AbortSignal): Promise<FacetsResponse> {
    return request(`${sourcePath(sourceId)}/facets`, { signal });
  },

  tasks(sourceId: string, query: TaskQuery, signal?: AbortSignal): Promise<TasksResponse> {
    const params = new URLSearchParams();
    if (query.split) params.append("split", query.split);
    if (query.minDistance !== undefined) params.set("min_distance", String(query.minDistance));
    if (query.maxDistance !== undefined) params.set("max_distance", String(query.maxDistance));
    if (query.functionKey) params.append("function_key", query.functionKey);
    if (query.functionRole) params.append("function_role", query.functionRole);
    if (query.taskPrefix) params.set("task_prefix", query.taskPrefix);
    if (query.cursor) params.set("cursor", query.cursor);
    params.set("limit", String(query.limit ?? 50));
    return request(`${sourcePath(sourceId)}/tasks?${params}`, { signal });
  },

  task(sourceId: string, rowId: number, signal?: AbortSignal): Promise<TaskDetail> {
    return request(`${sourcePath(sourceId)}/tasks/${rowId}`, { signal });
  },

  witnessGraph(
    sourceId: string,
    body: WitnessGraphRequest,
    signal?: AbortSignal,
  ): Promise<WitnessGraphResponse> {
    return request(`${sourcePath(sourceId)}/graphs/witnesses`, {
      method: "POST",
      body: JSON.stringify(body),
      signal,
    });
  },

  expansionGraph(
    sourceId: string,
    body: ExpansionGraphRequest,
    signal?: AbortSignal,
  ): Promise<ExpansionGraphResponse> {
    return request(`${sourcePath(sourceId)}/graphs/expand`, {
      method: "POST",
      body: JSON.stringify(body),
      signal,
    });
  },
};
