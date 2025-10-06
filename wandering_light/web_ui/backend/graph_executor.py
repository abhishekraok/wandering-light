from collections import deque
from typing import Any

from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDef
from wandering_light.typed_list import TypedList


class GraphExecutor:
    def __init__(self, available_functions: list[FunctionDef], executor: Executor):
        self.available_functions = available_functions
        self.executor = executor

    def topological_sort(self, nodes: list[dict], edges: list[dict]) -> list[str]:
        """Sort nodes in topological order to determine execution sequence."""
        # Create a graph representation using adjacency list
        graph = {}
        in_degree = {}

        # Initialize the graph
        for node in nodes:
            node_id = node["id"]
            graph[node_id] = []
            in_degree[node_id] = 0

        # Build the graph
        for edge in edges:
            source = edge["source"]
            target = edge["target"]
            graph[source].append(target)
            in_degree[target] = in_degree.get(target, 0) + 1

        # Topological sort using Kahn's algorithm
        queue = deque([node_id for node_id in graph if in_degree[node_id] == 0])
        sorted_nodes = []

        while queue:
            current = queue.popleft()
            sorted_nodes.append(current)

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycles
        if len(sorted_nodes) != len(nodes):
            raise ValueError(
                "Graph contains a cycle and cannot be executed in sequence"
            )

        return sorted_nodes

    def get_input_nodes(
        self, node_id: str, nodes: list[dict], edges: list[dict]
    ) -> list[tuple[str, str]]:
        """Get all nodes that connect to the given node as inputs."""
        input_nodes = []

        for edge in edges:
            if edge["target"] == node_id:
                source_id = edge["source"]
                source_handle = edge.get("sourceHandle", "default")
                input_nodes.append((source_id, source_handle))

        return input_nodes

    def execute_graph(self, graph_data: dict[str, Any]) -> dict[str, Any]:
        """Execute the entire graph and return results for each node."""
        nodes = graph_data["nodes"]
        edges = graph_data["edges"]

        # Map of node id to its execution result
        node_results = {}

        # Get execution order
        execution_order = self.topological_sort(nodes, edges)

        # Execute each node in order
        for node_id in execution_order:
            # Find the node data
            node_data = next(node for node in nodes if node["id"] == node_id)

            # Skip if not an executable node
            node_type = node_data["type"]
            if node_type not in ["function_def", "typed_list"]:
                continue

            try:
                if node_type == "typed_list":
                    # For typed list nodes, just store the value
                    item_type_str = node_data["data"]["item_type"]
                    items = node_data["data"]["items"]

                    # Resolve the type class
                    module_name, _, class_name = item_type_str.rpartition(".")
                    import importlib

                    mod = importlib.import_module(module_name)
                    item_type = getattr(mod, class_name)

                    # Create the TypedList
                    typed_list = TypedList(items, item_type=item_type)
                    node_results[node_id] = typed_list

                elif node_type == "function_def":
                    # For function nodes, execute the function with inputs
                    function_name = node_data["data"]["name"]

                    # Get function from available functions
                    fn = None
                    for f in self.available_functions:
                        if f.name == function_name:
                            fn = f
                            break

                    if fn is None:
                        raise ValueError(
                            f"Function {function_name} not found in available functions"
                        )

                    # Get input from connected nodes
                    input_connections = self.get_input_nodes(node_id, nodes, edges)
                    if not input_connections:
                        raise ValueError(
                            f"Function node {node_id} has no input connections"
                        )

                    # For now, we just take the first input connection
                    # In a more advanced version, we could handle multiple inputs
                    input_node_id, _ = input_connections[0]

                    if input_node_id not in node_results:
                        raise ValueError(
                            f"Input node {input_node_id} has not been executed yet"
                        )

                    # Execute the function
                    input_data = node_results[input_node_id]
                    result = self.executor.execute(fn, input_data)
                    node_results[node_id] = result

            except (
                ValueError,
                TypeError,
                AttributeError,
                ImportError,
                RuntimeError,
                Exception,
            ) as e:
                return {
                    "status": "error",
                    "error": f"Error executing node {node_id}: {e!s}",
                }

        # Format results for return
        formatted_results = []
        for node_id, result in node_results.items():
            if isinstance(result, TypedList):
                formatted_results.append(
                    {
                        "node_id": node_id,
                        "result": result.items,
                        "result_type": f"{result.item_type.__module__}.{result.item_type.__qualname__}",
                    }
                )

        return {"status": "success", "results": formatted_results}
