"""The CI workflows invoke this CLI by string, which nothing else checks.

A renamed flag turns a quality gate into a red build at best, and at worst into a skipped
step nobody reads -- and the private-eval workflow cannot run on a GitHub-hosted runner at
all, so its command would otherwise go unexercised until someone wires up a runner.
"""

import re

import pytest
from typer.testing import CliRunner

from rag_assistant.cli import app
from rag_assistant.config import PROJECT_ROOT

WORKFLOWS = sorted((PROJECT_ROOT / ".github" / "workflows").glob("*.yml"))
INVOCATION = re.compile(r"rag-assistant\s+([a-z-]+)((?:\s+--?[a-z-]+(?:\s+\S+)?)*)")


def _invocations(text: str) -> list[tuple[str, list[str]]]:
    # Shell line continuations first: a command split over lines is exactly the long one whose
    # flags most need checking, and leaving them out made this test pass by seeing nothing.
    text = re.sub(r"\\\s*\n\s*", " ", text)
    found = []
    for command, rest in INVOCATION.findall(text):
        flags = re.findall(r"--[a-z-]+", rest)
        found.append((command, flags))
    return found


def test_workflows_exist_and_call_the_cli():
    calls = [c for path in WORKFLOWS for c in _invocations(path.read_text())]

    assert {name for path in WORKFLOWS for name in [path.name]} >= {"ci.yml", "private-eval.yml"}
    assert calls, "no rag-assistant invocations found in any workflow"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_every_command_and_flag_a_workflow_uses_still_exists(path):
    runner = CliRunner()
    for command, flags in _invocations(path.read_text()):
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0, f"{path.name}: `{command}` is not a command"
        # Typer wraps help text, so compare against it with whitespace collapsed.
        help_text = " ".join(result.stdout.split())
        for flag in flags:
            assert flag in help_text, f"{path.name}: `{command}` has no {flag}"
