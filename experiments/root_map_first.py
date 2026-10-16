"""Experiment-only root-map proposal; fixed #186 contract in DESIGN.md."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from experiments.gumbel_retention_run import ShownTask
from wandering_light.stack_beam import beam_solve
from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    Config,
    Lst,
    Mark,
    format_stack,
    ops_for_basis,
    parse_stack,
    step_raw,
)

SETTINGS = {"beam_width": 64, "guard": "both", "max_state_atoms": 16384, "budget": None}
TASK_SECONDS = 5.0
BODY_SECONDS = 1.0
OPS = {op.name: op for op in ops_for_basis("mark-28")}


@dataclass(frozen=True)
class Template:
    reason: str
    inputs: tuple = ()
    outputs: tuple = ()
    body: ShownTask | None = None
    invocations: int = 0
    empty_body: bool = False


def derive(task):
    """Necessary constraints for this root, no-capture template, not all tasks."""
    if not task.inputs or len(task.inputs) != len(task.outputs):
        raise ValueError("malformed shown population")
    inputs = tuple(parse_stack(s, basis="mark-28") for s in task.inputs)
    outputs = tuple(parse_stack(s, basis="mark-28") for s in task.outputs)
    ins, outs, known = [], [], {}
    identity = True
    for before, after in zip(inputs, outputs, strict=True):
        if (
            not before
            or not after
            or not isinstance(before[-1], Lst)
            or not isinstance(after[-1], Lst)
        ):
            return Template("not_list_tops")
        if before[:-1] != after[:-1]:
            return Template("outer_rest_changed")
        if len(before[-1].items) != len(after[-1].items):
            return Template("length_changed")
        for given, wanted in zip(before[-1].items, after[-1].items, strict=True):
            if given in known and known[given] != wanted:
                return Template("contradictory_element")
            known[given] = wanted
            if isinstance(wanted, Mark):
                return Template("unsupported_mark_goal")
            pair = []
            for value in (given, wanted):
                text = format_stack(Config(stack=(value,)))
                if parse_stack(text, basis="mark-28") != (value,):
                    return Template("unrepresentable_singleton")
                pair.append(text)
            ins.append(pair[0])
            outs.append(pair[1])
            identity = identity and given == wanted
    return Template(
        "eligible",
        inputs,
        outputs,
        ShownTask(tuple(ins), tuple(outs)),
        len(ins),
        identity,
    )


def verify_wrapper(program, inputs, outputs, deadline):
    """Actual original whole-program execution; prospective check per outer op."""
    if len(program) > 48 or any(name not in OPS for name in program):
        raise ValueError("malformed reconstructed primitive path")
    product = inputs
    executions = fuel = attempts = 0
    for index, name in enumerate(program):
        if time.perf_counter() >= deadline:
            return {
                "fit": False,
                "reason": "wall_clock_cap",
                "completed_ops": index,
                "executions": executions,
                "fuel": fuel,
                "attempts": attempts,
            }
        moved = []
        for example, stack in enumerate(product):
            attempts += 1
            result = step_raw(OPS[name], stack, DEFAULT_FUEL_LIMIT, 0)
            if isinstance(result, str):
                return {
                    "fit": False,
                    "reason": "refusal",
                    "refusal": result,
                    "op_index": index,
                    "example": example,
                    "completed_ops": index,
                    "executions": executions,
                    "fuel": fuel,
                    "attempts": attempts,
                }
            executions += 1
            fuel += result[1]
            moved.append(result[0])
        product = tuple(moved)
    fit = product == outputs
    return {
        "fit": fit,
        "reason": "fit" if fit else "shown_mismatch",
        "completed_ops": len(program),
        "executions": executions,
        "fuel": fuel,
        "attempts": attempts,
    }


def search(task, scorer, deadline, depth, stage):
    started = time.perf_counter()
    result = beam_solve(task, scorer, max_depth=depth, deadline=deadline, **SETTINGS)
    return {
        "stage": stage,
        **asdict(result),
        "deadline": deadline,
        "depth_limit": depth,
        "elapsed_seconds": time.perf_counter() - started,
    }


def solve_task(task, scorer, arm):
    """One fixed whole-task deadline; no body rechoice and no new fallback clock."""
    if arm not in ("control", "candidate"):
        raise ValueError("unknown arm")
    started = time.perf_counter()
    deadline = started + TASK_SECONDS
    stages = []

    def finish(program, termination, source, eligibility=None, invocations=0):
        return {
            "program": list(program) if program is not None else None,
            "termination": termination,
            "source": source,
            "seconds": time.perf_counter() - started,
            "eligibility": eligibility,
            "invocations": invocations,
            "task_started": started,
            "task_deadline": deadline,
            "body_deadline": started + BODY_SECONDS,
            "stages": stages,
        }

    if arm == "control":
        result = search(task, scorer, deadline, 48, "control")
        stages.append(result)
        return finish(result["program"], result["termination"], "control")
    classified = time.perf_counter()
    template = derive(task)
    stages.append(
        {
            "stage": "classify",
            "reason": template.reason,
            "invocations": template.invocations,
            "elapsed_seconds": time.perf_counter() - classified,
        }
    )
    if template.reason == "eligible":
        body = None
        if template.empty_body:
            body = ()
            stages.append(
                {"stage": "empty_body", "program": [], "elapsed_seconds": 0.0}
            )
        elif time.perf_counter() < started + BODY_SECONDS:
            result = search(
                template.body, scorer, min(started + BODY_SECONDS, deadline), 45, "body"
            )
            stages.append(result)
            body = result["program"]
        else:
            stages.append(
                {
                    "stage": "body_skipped",
                    "reason": "slice_spent",
                    "elapsed_seconds": 0.0,
                }
            )
        if body is not None:
            if len(body) > 45:
                raise ValueError("body result exceeded fixed depth")
            program = ("[", *body, "]", "map")
            checked = time.perf_counter()
            verification = verify_wrapper(
                program, template.inputs, template.outputs, deadline
            )
            stages.append(
                {
                    "stage": "verify",
                    "program": list(program),
                    **verification,
                    "elapsed_seconds": time.perf_counter() - checked,
                }
            )
            if verification["fit"]:
                return finish(
                    program, "fit", "root_map", template.reason, template.invocations
                )
    if time.perf_counter() < deadline:
        result = search(task, scorer, deadline, 48, "fallback")
        stages.append(result)
        return finish(
            result["program"],
            result["termination"],
            "fallback",
            template.reason,
            template.invocations,
        )
    return finish(None, "wall_clock_cap", "none", template.reason, template.invocations)
