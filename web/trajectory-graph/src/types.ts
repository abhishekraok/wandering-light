export type IndexStatus = "idle" | "indexing" | "ready" | "error";

export interface IndexState {
  status: IndexStatus;
  records_indexed: number | null;
  file_index?: number;
  file_count?: number;
  message?: string;
  error?: string | null;
}

export interface CorpusSource {
  id: string;
  name: string;
  ready: boolean;
  missing_files: string[];
  expected_records: number | null;
  basis_set_id: string | null;
  basis_set_digest: string | null;
  fetchable: boolean;
  hub_repo_id: string | null;
  hub_revision: string | null;
  index: IndexState;
}

export interface SourcesResponse {
  sources: CorpusSource[];
  errors: Array<{ location: string; message: string }>;
}

export interface CorpusStats {
  records: number;
  certified_records: number;
  min_distance: number | null;
  max_distance: number | null;
  mean_distance: number | null;
  roots: number;
}

export interface FunctionFacet {
  function_key: string;
  function_name: string;
  role: string;
  records: number;
}

export interface FacetsResponse {
  stats: CorpusStats;
  splits: string[];
  distance_counts: Array<{ value: number; records: number }>;
  input_types: string[];
  output_types: string[];
  certifications: Array<{ value: string; records: number }>;
  functions: FunctionFacet[];
}

export interface TaskSummary {
  row_id: number;
  task_id: string;
  split: string;
  distance: number | null;
  input_type: string;
  output_type: string;
  certified: boolean | null;
  witness_function_names: string[];
  input_preview: string;
  output_preview: string;
  root_index: number | null;
}

export interface TasksResponse {
  items: TaskSummary[];
  next_cursor: string | null;
  total: number;
}

export interface BasisFunction {
  id: string | null;
  name: string;
  input_type: string;
  output_type: string;
}

export interface TaskDetail {
  row_id: number;
  task_id: string;
  schema_kind: string;
  split: string;
  input: string;
  output: string | null;
  input_type: string;
  output_type: string;
  distance: number | null;
  certified: boolean | null;
  witness_function_names: string[];
  witness_function_ids: string[];
  root_index: number | null;
  certification: string | null;
  metadata: Record<string, unknown>;
  functions_by_role: Record<string, string[]>;
  raw_json: string;
  basis: {
    id: string;
    digest: string;
    assumed: boolean;
    functions: BasisFunction[];
  };
}

export interface GraphNode {
  id: number;
  x: number;
  y: number;
  depth: number;
  label: string;
  value: string;
  role: "root" | "selected_target" | "selected_path" | "target" | "state";
}

export interface GraphEdge {
  id: string;
  source: number;
  target: number;
  function_names: string[];
  highlighted: boolean;
}

export interface GraphDiagnostics {
  self_loop_groups: number;
  parallel_function_groups: number;
  convergent_nodes: number;
  directed_cycle_groups: number;
}

export interface GraphPayload {
  nodes: GraphNode[];
  edges: GraphEdge[];
  root_ids: number[];
  total_nodes: number;
  total_edges: number;
  rendered_nodes: number;
  rendered_edge_groups: number;
  truncated: boolean;
  diagnostics: GraphDiagnostics;
}

export interface WitnessGraphResponse {
  mode: "witnesses";
  graph: GraphPayload;
  processed_records: number;
  skipped_records: number;
  errors: string[];
}

export interface CandidateTask {
  node_id: number;
  distance: number;
  output: string;
  output_serialized: string;
  function_names: string[];
  certified: boolean;
}

export interface ExpansionGraphResponse {
  mode: "expansion";
  graph: GraphPayload;
  tasks: CandidateTask[];
  attempted_transitions: number;
  failed_transitions: number;
  skipped_self_loops: number;
  certified_depth: number;
  stop_reason: string | null;
}

export interface TaskQuery {
  split?: string;
  minDistance?: number;
  maxDistance?: number;
  functionKey?: string;
  functionRole?: string;
  taskPrefix?: string;
  cursor?: string;
  limit?: number;
}

export interface WitnessGraphRequest {
  row_id: number;
  scope: "selected" | "root";
  max_records: number;
  max_nodes: number;
  max_edges: number;
}

export interface ExpansionGraphRequest {
  row_id: number;
  input_serialized: string;
  function_ids: string[];
  max_depth: number;
  max_states: number;
  max_transitions: number;
  include_self_loops: boolean;
  max_nodes: number;
  max_edges: number;
}
