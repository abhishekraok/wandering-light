import importlib
import json
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from wandering_light.common_functions import basic_fns
from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDef, FunctionDefSet
from wandering_light.solver import (
    TokenGeneratorPredictor,
    TrainedLLMTokenGenerator,
    TrajectorySolver,
)
from wandering_light.typed_list import TypedList
from wandering_light.web_ui.backend.graph_executor import GraphExecutor

# Initialize our app
app = FastAPI(title="Wandering Light Graph Editor API")

# Setup CORS to allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, limit this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize function storage and executor with basic functions
available_functions = list(
    basic_fns.functions
)  # Initialize with basic functions from common_functions.py
executor = Executor(available_functions)
graph_executor = GraphExecutor(available_functions, executor)


# Data models for API
class FunctionDefCreate(BaseModel):
    name: str
    input_type: str
    output_type: str
    code: str
    metadata: dict[str, Any] = {}


class TypedListCreate(BaseModel):
    item_type: str
    items: list[Any]


class GraphNodeCreate(BaseModel):
    id: str
    type: str  # "function_def" or "typed_list"
    data: dict[str, Any]
    position: dict[str, float]


class GraphEdgeCreate(BaseModel):
    id: str
    source: str
    target: str
    sourceHandle: str | None = None  # noqa: N815 (matches frontend API)
    targetHandle: str | None = None  # noqa: N815 (matches frontend API)


class GraphCreate(BaseModel):
    nodes: list[GraphNodeCreate]
    edges: list[GraphEdgeCreate]
    name: str


class ExecutionResult(BaseModel):
    node_id: str
    result: list[Any]
    result_type: str


class SolverRequest(BaseModel):
    input_list: TypedListCreate
    output_list: TypedListCreate
    checkpoint_path: str = "abhishekraok/induction-basicfns-opt125m-longsft"


class SolverResponse(BaseModel):
    success: bool
    predicted_functions: list[str]
    predicted_output: TypedListCreate | None
    error_msg: str | None = None


# API Endpoints
@app.get("/")
def read_root():
    return {"message": "Wandering Light Graph Editor API"}


@app.get("/functions")
def get_functions():
    return available_functions


@app.post("/functions")
def create_function(function: FunctionDefCreate):
    try:
        # Check if function already exists
        if any(f.name == function.name for f in available_functions):
            raise ValueError(f"Function {function.name} already exists")

        fn_def = FunctionDef(
            name=function.name,
            input_type=function.input_type,
            output_type=function.output_type,
            code=function.code,
            metadata=function.metadata,
        )
        available_functions.append(fn_def)

        # Update executor with new function list
        global executor, graph_executor  # noqa: PLW0603 (needed for web app state)
        executor = Executor(available_functions)
        graph_executor = GraphExecutor(available_functions, executor)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    else:
        return fn_def


@app.get("/functions/{name}")
def get_function(name: str):
    for fn in available_functions:
        if fn.name == name:
            return fn
    raise HTTPException(status_code=404, detail=f"Function {name} not found")


@app.post("/execute")
def execute_function(function_name: str, inputs: TypedListCreate):
    try:
        # Get the function
        fn = None
        for f in available_functions:
            if f.name == function_name:
                fn = f
                break

        if fn is None:
            raise HTTPException(
                status_code=404, detail=f"Function {function_name} not found"
            )

        # Create TypedList from input
        module_name, _, class_name = inputs.item_type.rpartition(".")

        mod = importlib.import_module(module_name)
        item_type = getattr(mod, class_name)

        typed_list = TypedList(inputs.items, item_type=item_type)

        # Execute
        result = executor.execute(fn, typed_list)

    except (
        ValueError,
        TypeError,
        AttributeError,
        ImportError,
        ModuleNotFoundError,
    ) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    else:
        # Convert to serializable format
        return {
            "items": result.items,
            "item_type": f"{result.item_type.__module__}.{result.item_type.__qualname__}",
        }


# WebSocket endpoint for real-time graph execution
@app.post("/solver/execute")
def execute_solver(request: SolverRequest):
    try:
        # Create TypedLists from inputs
        module_name, _, class_name = request.input_list.item_type.rpartition(".")
        mod = importlib.import_module(module_name)
        input_type = getattr(mod, class_name)
        input_list = TypedList(request.input_list.items, item_type=input_type)

        module_name, _, class_name = request.output_list.item_type.rpartition(".")
        mod = importlib.import_module(module_name)
        output_type = getattr(mod, class_name)
        expected_output_list = TypedList(
            request.output_list.items, item_type=output_type
        )

        # Use the solver to predict which functions would achieve the expected output
        token_generator = TrainedLLMTokenGenerator(
            model_or_path=request.checkpoint_path
        )
        predictor = TokenGeneratorPredictor(token_generator, budget=1)
        solver = TrajectorySolver(predictor)

        # Create FunctionDefSet from all available functions
        available_fn_set = FunctionDefSet(available_functions)

        # Solve to get predicted functions that would transform input to expected output
        result = solver.solve(input_list, expected_output_list, available_fn_set)

        if result.success and result.trajectory:
            predicted_functions = [fn.name for fn in result.trajectory.function_defs]
            predicted_output = result.trajectory.output
            predicted_output_create = TypedListCreate(
                item_type=f"{predicted_output.item_type.__module__}.{predicted_output.item_type.__qualname__}",
                items=predicted_output.items,
            )
        else:
            predicted_functions = []
            predicted_output_create = None

        return SolverResponse(
            success=result.success,
            predicted_functions=predicted_functions,
            predicted_output=predicted_output_create,
            error_msg=result.error_msg,
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/functions/search")
def search_functions(query: str = ""):
    """Search functions by name prefix for autocomplete."""
    if not query:
        return [f.name for f in available_functions]

    matching_functions = [
        f.name for f in available_functions if f.name.lower().startswith(query.lower())
    ]
    return matching_functions


@app.websocket("/ws/execute-graph")
async def execute_graph_ws(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            # Receive graph data
            data = await websocket.receive_text()
            graph_data = json.loads(data)

            # Execute the graph
            try:
                result = graph_executor.execute_graph(graph_data)
                await websocket.send_json(result)
            except Exception as e:
                await websocket.send_json({"status": "error", "error": str(e)})
    except WebSocketDisconnect:
        print("Client disconnected")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
