"""The baseline runner: the search backend and the model-response parser.

The parser is the part worth testing without an API key. A model's reply is
prose plus a program, and normalising it wrong either invents a solve or
throws one away.
"""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from stack_pbe_baseline import (
    bfs_solve,
    llm_prompt,
    parse_response,
    random_solve,
    solve_with_llm,
)

from wandering_light.stack_pbe import BY_NAME, build_task, draw_task
from wandering_light.stack_pbe_eval import grade


@pytest.fixture(scope="module")
def sum_task():
    rng = random.Random(4)
    return build_task(BY_NAME["sum"], draw_task(BY_NAME["sum"], rng, 4, held_out=6))


@pytest.fixture(scope="module")
def add_task():
    rng = random.Random(6)
    return build_task(
        BY_NAME["add_pair"], draw_task(BY_NAME["add_pair"], rng, 4, held_out=6)
    )


@pytest.mark.parametrize(
    "reply",
    [
        "0 [ add ] fold",
        "The answer is:\n```\n0 [ add ] fold\n```",
        "<think>fold it</think>\n0 [add] fold",
        "`0 [ add ] fold`",
        "Sum the list.\n0 [ add ] fold",
    ],
)
def test_parse_response_recovers_the_program(reply, sum_task):
    assert grade(parse_response(reply), sum_task).solved


def test_parse_response_does_not_repair_a_wrong_answer(sum_task):
    """An invented op name must stay invented, so grading can reject it."""
    assert not grade(parse_response("0 [ plus ] fold"), sum_task).parsed


def test_parse_response_survives_an_empty_reply(sum_task):
    assert parse_response("") == ""
    assert not grade(parse_response(""), sum_task).parsed


def test_bfs_finds_a_one_op_program(add_task):
    assert bfs_solve(add_task, budget=500) == ("add",)


def test_bfs_returns_nothing_past_its_budget(sum_task):
    """`0 [ add ] fold` is five ops; a tiny budget cannot reach it."""
    assert bfs_solve(sum_task, budget=5) is None


def test_bfs_never_reads_the_held_out_examples(sum_task, monkeypatch):
    """The protocol depends on this: a solver sees the shown examples only."""
    from dataclasses import replace

    blinded = replace(sum_task, held_out_inputs=(), held_out_outputs=(), held_out=0)
    assert bfs_solve(blinded, budget=20_000) == bfs_solve(sum_task, budget=20_000)


def test_guard_finds_the_same_quotation_building_program(sum_task):
    """`0 [ add ] fold` opens a `[`; a sound guard must not lose it.

    The guard changes what a budget reaches, never which programs exist, so
    at a budget where the unpruned search finds the witness the guarded one
    must return the very same program.
    """
    unpruned = bfs_solve(sum_task, budget=20_000)
    assert unpruned is not None
    assert bfs_solve(sum_task, budget=20_000, guard="both") == unpruned


def test_guard_leaves_first_order_search_alone(add_task):
    assert bfs_solve(add_task, budget=500, guard="both") == ("add",)


def test_close_guard_refuses_a_mark_bearing_target(add_task):
    """`close` is sound only because every target is mark-free."""
    from dataclasses import replace

    marked = replace(add_task, outputs=tuple("[ 1 MARK ]" for _ in add_task.outputs))
    with pytest.raises(ValueError, match="mark-free"):
        bfs_solve(marked, budget=10, guard="close")
    # The body arm does not depend on that property, so it must still run.
    assert bfs_solve(marked, budget=10, guard="body") is None


def test_random_finds_a_one_op_program(add_task):
    found = random_solve(add_task, budget=2_000, rng=random.Random(1), max_length=2)
    assert found is not None
    assert grade(found, add_task).shown


def test_prompt_shows_the_examples_and_hides_the_held_out_set(sum_task):
    prompt = llm_prompt(sum_task)
    for given in sum_task.inputs:
        assert given in prompt
    for hidden in sum_task.held_out_inputs:
        assert hidden not in prompt
    assert "fold" in prompt


def test_every_provider_is_reachable_by_name():
    """The four backends exist and take a model name. No key needed to check."""
    import wandering_light.solver as solver_module

    for name in (
        "OpenAITokenGenerator",
        "GeminiTokenGenerator",
        "OllamaTokenGenerator",
        "CodexExecTokenGenerator",
        "ClaudeExecTokenGenerator",
    ):
        assert hasattr(solver_module, name)
    generator = solver_module.AnthropicTokenGenerator("claude-opus-5", effort="low")
    assert generator.model == "claude-opus-5"
    assert generator.effort == "low"


def test_a_failed_generation_costs_one_candidate_not_the_run(sum_task):
    """The deadline that bounds a runaway must not end the evaluation itself.

    Without this the first timeout propagates out of the task loop and the run
    dies before writing any summary, which is worse than the hang it replaced.
    """

    class Exploding:
        def generate(self, prompt):
            raise TimeoutError("read timeout")

    assert solve_with_llm(sum_task, Exploding(), samples=2) == ["", ""]
    assert not grade("", sum_task).parsed


def test_codex_events_yield_the_last_message_and_count_tool_attempts():
    """`codex exec --json` is a stream; the answer is the last agent message.

    A completed item that is neither a message nor an error is a tool attempt.
    With the tool features off it fails closed, and the count is what lets a
    row say it stayed the API protocol (#123).
    """
    import json

    from wandering_light.solver import CodexExecTokenGenerator

    lines = [
        json.dumps({"type": "thread.started", "thread_id": "t"}),
        "not json",
        json.dumps(
            {
                "type": "item.completed",
                "item": {"type": "command_execution", "command": "python3 -c 1"},
            }
        ),
        json.dumps(
            {
                "type": "item.completed",
                "item": {"type": "error", "message": "Code Mode is unavailable"},
            }
        ),
        json.dumps(
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "0 [ add ] fold"},
            }
        ),
        json.dumps(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 7594, "output_tokens": 17},
            }
        ),
    ]
    message, meta = CodexExecTokenGenerator.parse_events(lines)
    assert message == "0 [ add ] fold"
    assert meta["tool_calls"] == 1
    assert meta["errors"] == ["Code Mode is unavailable"]
    assert meta["usage"]["input_tokens"] == 7594


def test_codex_command_disables_every_tool_and_replaces_the_instructions(tmp_path):
    (tmp_path / "auth.json").write_text("{}")
    from wandering_light.solver import CodexExecTokenGenerator

    generator = CodexExecTokenGenerator(
        "gpt-6-astra", effort="low", codex_home=tmp_path
    )
    argv = generator.command()
    assert "--disable" in argv and "shell_tool" in argv and "code_mode_host" in argv
    assert "--strict-config" in argv and "--ephemeral" in argv
    assert any(a.startswith("model_instructions_file=") for a in argv)
    assert 'model_reasoning_effort="low"' in argv
    assert argv[-1] == "-"
    assert (tmp_path / "instructions.md").read_text().strip()


def test_codex_runs_against_its_private_home_not_the_users(tmp_path, monkeypatch):
    """The private CODEX_HOME is only private if the subprocess is told to use it.

    Without `CODEX_HOME` in the child's environment, Codex loads the caller's
    home instead: its `AGENTS.md`, config and plugins land above the prompt.
    Every run in reports/stack-pbe-v4-codex-exec-20260910 was made that way,
    and the model could name the user's employer when asked.
    """
    real_home = tmp_path / "real"
    real_home.mkdir()
    (real_home / "auth.json").write_text("{}")
    monkeypatch.setenv("CODEX_HOME", str(real_home))
    from wandering_light.solver import CodexExecTokenGenerator

    private = tmp_path / "private"
    generator = CodexExecTokenGenerator("gpt-6-astra", codex_home=private)
    assert generator.env["CODEX_HOME"] == str(private)
    assert generator.env["CODEX_HOME"] != str(real_home)


def test_codex_timeout_reports_its_own_failure_not_the_previous_call(
    tmp_path, monkeypatch
):
    """A failed call must not inherit the previous task's telemetry.

    `solve_with_llm` reads `last_meta` on the failure path too, so if a timeout
    left the previous call's metadata in place the failed task would be
    credited with that task's tokens and duration (Codex review on PR #126).
    """
    import subprocess

    (tmp_path / "auth.json").write_text("{}")
    from wandering_light.solver import CodexExecTokenGenerator

    generator = CodexExecTokenGenerator("gpt-6-astra", codex_home=tmp_path, timeout=0.5)
    generator.last_meta = {
        "usage": {"input_tokens": 7594},
        "seconds": 12.3,
        "tool_calls": 0,
    }

    killed = []

    class Hanging:
        pid = 4242
        returncode = None

        def communicate(self, *args, **kwargs):
            if "timeout" in kwargs or args[1:]:
                raise subprocess.TimeoutExpired("codex", 0.5)
            return ("", "")

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: Hanging())
    monkeypatch.setattr(
        CodexExecTokenGenerator, "kill_group", staticmethod(killed.append)
    )

    with pytest.raises(subprocess.TimeoutExpired):
        generator.generate("anything")

    assert killed, "the process group must be killed, not left running"
    assert generator.last_meta["timed_out"] is True
    assert "usage" not in generator.last_meta, (
        "stale token counts leaked into a failed call"
    )
    assert generator.last_meta["seconds"] < 12.3


def test_codex_launches_in_its_own_process_group(tmp_path, monkeypatch):
    """A timeout must be able to kill the native binary, not just the launcher."""
    import subprocess

    (tmp_path / "auth.json").write_text("{}")
    from wandering_light.solver import CodexExecTokenGenerator

    generator = CodexExecTokenGenerator("gpt-6-astra", codex_home=tmp_path)
    seen = {}

    class Done:
        pid = 1
        returncode = 0

        def communicate(self, *args, **kwargs):
            return (
                '{"type":"item.completed","item":{"type":"agent_message","text":"dup"}}',
                "",
            )

    def record(*args, **kwargs):
        seen.update(kwargs)
        return Done()

    monkeypatch.setattr(subprocess, "Popen", record)
    assert generator.generate("prompt") == "dup"
    assert seen["start_new_session"] is True
    assert seen["env"]["CODEX_HOME"] == str(tmp_path)


def test_claude_result_yields_the_answer_and_flags_a_tool_call_written_as_text():
    """`claude -p --output-format json` is one object; the answer is `result`.

    Usage is renamed to the codex keys so one analysis reads both, and a tool
    call the model wrote into its answer -- it has no tool to call -- is
    flagged: the probe for opus-effort-curve (#137) saw exactly that.
    """
    import json

    from wandering_light.solver import ClaudeExecTokenGenerator

    def result(text, thinking=10819):
        return json.dumps(
            {
                "type": "result",
                "is_error": False,
                "num_turns": 1,
                "stop_reason": "end_turn",
                "duration_api_ms": 135241,
                "result": text,
                "usage": {
                    "input_tokens": 2,
                    "cache_creation_input_tokens": 4187,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 10881,
                    "output_tokens_details": {"thinking_tokens": thinking},
                },
                "modelUsage": {"claude-opus-5": {"outputTokens": 10881}},
                "permission_denials": [],
            }
        )

    message, meta = ClaudeExecTokenGenerator.parse_result(result("0 [ add ] fold"))
    assert message == "0 [ add ] fold"
    assert meta["tool_calls"] == 0 and meta["text_tool_call"] is False
    assert meta["errors"] == []
    assert meta["usage"]["input_tokens"] == 4189
    assert meta["usage"]["output_tokens"] == 10881
    assert meta["usage"]["reasoning_output_tokens"] == 10819
    assert meta["model"] == ["claude-opus-5"]

    call = '{"name":"Bash","input":{"command":"ls; grep -rl uncons ."}}'
    message, meta = ClaudeExecTokenGenerator.parse_result(result(call))
    assert message == call
    assert meta["text_tool_call"] is True

    _, meta = ClaudeExecTokenGenerator.parse_result("Not logged in")
    assert meta["errors"] and "unparseable" in meta["errors"][0]


def test_claude_command_turns_every_tool_off_and_replaces_the_system_prompt(tmp_path):
    from wandering_light.solver import ClaudeExecTokenGenerator

    generator = ClaudeExecTokenGenerator("opus", effort="xhigh", workdir=tmp_path)
    argv = generator.command()
    assert argv[:2] == ["claude", "-p"]
    assert argv[argv.index("--tools") + 1] == ""
    assert "--strict-mcp-config" in argv and "--no-session-persistence" in argv
    assert argv[argv.index("--setting-sources") + 1] == ""
    assert "no tools" in argv[argv.index("--system-prompt") + 1]
    assert argv[argv.index("--effort") + 1] == "xhigh"
    assert "--fallback-model" not in argv, "a fallback would answer as another model"
    assert "--bare" not in argv, "--bare disables the subscription's OAuth"
    assert generator.workdir == tmp_path and not any(tmp_path.iterdir())


def test_claude_runs_on_the_subscription_not_an_exported_api_key(tmp_path, monkeypatch):
    """An API key in the caller's shell would re-route billing without a trace.

    `--setting-sources ""` drops settings files, not environment variables
    (Codex review on PR #139). The subscription's OAuth is read from the home
    directory, which the child keeps.
    """
    import subprocess

    from wandering_light.solver import ClaudeExecTokenGenerator

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-this")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    monkeypatch.setenv("HOME", str(tmp_path))
    generator = ClaudeExecTokenGenerator("opus", workdir=tmp_path)
    assert "ANTHROPIC_API_KEY" not in generator.env
    assert "CLAUDE_CODE_USE_BEDROCK" not in generator.env
    assert generator.env["HOME"] == str(tmp_path)

    seen = {}

    class Done:
        pid = 1
        returncode = 0

        def communicate(self, *args, **kwargs):
            return ('{"result": "nil", "usage": {}, "modelUsage": {}}', "")

    def popen(*args, **kwargs):
        seen.update(kwargs)
        return Done()

    monkeypatch.setattr(subprocess, "Popen", popen)
    generator.generate("anything")
    assert seen["env"] is generator.env


def test_claude_timeout_reports_its_own_failure_not_the_previous_call(
    tmp_path, monkeypatch
):
    """Same contract as the codex generator: no stale telemetry on a failed call."""
    import subprocess

    from wandering_light.solver import ClaudeExecTokenGenerator, CodexExecTokenGenerator

    generator = ClaudeExecTokenGenerator("opus", workdir=tmp_path, timeout=0.5)
    generator.last_meta = {"usage": {"input_tokens": 4189}, "seconds": 99.0}
    killed = []

    class Hanging:
        pid = 4242
        returncode = None

        def communicate(self, *args, **kwargs):
            if "timeout" in kwargs or args[1:]:
                raise subprocess.TimeoutExpired("claude", 0.5)
            return ("", "")

    seen = {}

    def popen(*args, **kwargs):
        seen.update(kwargs)
        return Hanging()

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(
        CodexExecTokenGenerator, "kill_group", staticmethod(killed.append)
    )
    with pytest.raises(subprocess.TimeoutExpired):
        generator.generate("anything")
    assert killed
    assert seen["start_new_session"] is True and seen["cwd"] == tmp_path
    assert generator.last_meta["timed_out"] is True
    assert "usage" not in generator.last_meta


@pytest.mark.parametrize("provider", ["codex", "claude"])
def test_an_interrupted_call_kills_the_process_group(tmp_path, monkeypatch, provider):
    """Ctrl-C must not leave the model working in its own session.

    The baseline turns SIGTERM into KeyboardInterrupt and writes partial
    results; a child started with `start_new_session` does not see the
    signal and would keep spending tokens past the run (Codex review on
    PR #139). Same contract as the timeout path.
    """
    import subprocess

    from wandering_light.solver import ClaudeExecTokenGenerator, CodexExecTokenGenerator

    if provider == "codex":
        (tmp_path / "auth.json").write_text("{}")
        generator = CodexExecTokenGenerator("gpt-6-astra", codex_home=tmp_path)
    else:
        generator = ClaudeExecTokenGenerator("opus", workdir=tmp_path)
    generator.last_meta = {"usage": {"input_tokens": 1}, "seconds": 99.0}
    killed = []

    class Cancelled(BaseException):
        """Stands in for KeyboardInterrupt, which pytest itself intercepts."""

    class Interrupted:
        pid = 4242
        returncode = None
        reaped = False

        def communicate(self, *args, **kwargs):
            if "timeout" in kwargs or args[1:]:
                raise Cancelled
            self.reaped = True
            return ("", "")

    child = Interrupted()
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: child)
    monkeypatch.setattr(
        CodexExecTokenGenerator, "kill_group", staticmethod(killed.append)
    )
    with pytest.raises(Cancelled):
        generator.generate("anything")
    assert killed == [child] and child.reaped
    assert generator.last_meta["errors"] == ["interrupted"]
    assert "usage" not in generator.last_meta
    assert "timed_out" not in generator.last_meta


def test_gemini_sends_effort_as_thinking_level_and_bills_thoughts_as_output(
    monkeypatch,
):
    import httpx

    import wandering_light.solver as solver_module

    sent = {}

    def fake_post(url, headers, json, timeout):
        sent.update(url=url, body=json)
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "scratch", "thought": True},
                                {"text": "dup add"},
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 900,
                    "candidatesTokenCount": 5,
                    "thoughtsTokenCount": 120,
                },
                "modelVersion": "gemini-3.8-flash",
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    generator = solver_module.GeminiTokenGenerator(
        "models/gemini-3.8-flash", effort="high"
    )
    assert generator.generate("p") == "dup add"
    assert "models/gemini-3.8-flash:generateContent" in sent["url"]
    assert sent["body"]["generationConfig"]["thinkingConfig"] == {
        "thinkingLevel": "high"
    }
    assert generator.last_meta["usage"]["output_tokens"] == 125
    assert generator.last_meta["model"] == ["gemini-3.8-flash"]


def test_gemini_2_drops_effort_it_cannot_take():
    import wandering_light.solver as solver_module

    assert solver_module.GeminiTokenGenerator("gemini-2.5-flash", "high").effort is None
    assert solver_module.GeminiTokenGenerator("gemini-4-argon", "high").effort == "high"
