export interface StateView {
  id: string;
  wire: string;
  label: string;
  type: string;
  size: number;
}

export interface BasisFunction {
  name: string;
  input_type: string;
  output_type: string;
  code: string;
  function_id: string;
}

export interface BasisInfo {
  available: { id: string; alias: string | null }[];
  basis_set_id: string;
  digest: string;
  description: string;
  functions: BasisFunction[];
}

export interface Step {
  function: string;
  ok: boolean;
  state: StateView | null;
  error: string | null;
}

export interface TrajectoryResult {
  root: StateView;
  steps: Step[];
}

export interface Successor {
  function: string;
  ok: boolean;
  state: StateView | null;
  error: string | null;
  self_loop: boolean;
}

export interface SuccessorResult {
  state: StateView;
  successors: Successor[];
  applicable: number;
  dead_end: boolean;
}

export interface ExpansionNode extends StateView {
  depth: number;
  node_id: number;
}

export interface ExpansionEdge {
  source: number;
  target: number;
  function: string;
  depth: number;
}

export interface ExpansionStats {
  root_id: number;
  nodes: number;
  edges: number;
  certified_depth: number;
  attempted_transitions: number;
  failed_transitions: number;
  skipped_self_loops: number;
  complete: boolean;
  stop_reason: string | null;
  elapsed_seconds: number;
  by_depth: Record<string, number>;
}

export interface Expansion {
  nodes: ExpansionNode[];
  edges: ExpansionEdge[];
  stats: ExpansionStats;
  types: Record<string, number>;
  function_edges: Record<string, number>;
  idle_functions: string[];
}

export interface SolveAttempt {
  solver: string;
  success: boolean;
  functions: string[];
  output: StateView | null;
  error: string | null;
  elapsed_seconds: number;
}

export interface CorpusSummary {
  name: string;
  tasks: number;
  basis_set_id: string;
  splits: string[];
  missing_splits: string[];
  distances: Record<string, number>;
}

export interface CorpusTask {
  task_id: string;
  input: string;
  output: string;
  input_label: string;
  output_label: string;
  witness: string[];
  distance: number;
  certification: string;
  optimal_first: string[];
}
