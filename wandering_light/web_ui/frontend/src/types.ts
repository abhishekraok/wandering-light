// Type definitions for our application

export interface FunctionDef {
  name: string;
  input_type: string;
  output_type: string;
  code: string;
  usage_count: number;
  metadata: Record<string, any>;
}

export interface TypedListData {
  item_type: string;
  items: any[];
}

export interface NodeData {
  label: string;
  type: 'function_def' | 'typed_list';
  function?: FunctionDef;
  typedList?: TypedListData;
  result?: any[];
}

export interface GraphData {
  nodes: any[];
  edges: any[];
  name: string;
}

export interface ExecutionResult {
  node_id: string;
  result: any[];
  result_type: string;
}