"""Invariants of the CI and deploy workflows.

Workflow files are the one part of the system nothing else tests, and they fail in a
direction that is invisible: a job that skips its own suite, or a deploy that ships a commit
whose tests were red, both present as a green tick. The repo already relies on three service-
container jobs failing when their suites skip -- that rule is only worth anything if it is
enforced on the next such job somebody adds, so it is asserted here rather than remembered.
"""

from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"
CI = WORKFLOWS / "ci.yml"
DEPLOY = WORKFLOWS / "deploy.yml"


def load(path: Path) -> dict:
    # PyYAML parses the `on:` key as the boolean True (YAML 1.1), so it is normalised here
    # rather than at every use site.
    parsed = yaml.safe_load(path.read_text())
    if True in parsed:
        parsed["on"] = parsed.pop(True)
    return parsed


# Every job whose suite skips itself without a service container. Each one exists precisely
# because the fake it replaced could not disagree with the code.
SERVICE_CONTAINER_JOBS = ["postgres", "redis", "chroma-server"]


@pytest.fixture(scope="module")
def ci() -> dict:
    return load(CI)


@pytest.fixture(scope="module")
def deploy() -> dict:
    return load(DEPLOY)


def job_script(job: dict) -> str:
    return "\n".join(str(step.get("run", "")) for step in job.get("steps", []))


@pytest.mark.parametrize("job_name", SERVICE_CONTAINER_JOBS)
def test_service_container_jobs_fail_when_their_suite_skips(ci, job_name):
    """A suite that skips itself produces a green job having verified nothing, which is worse
    than no job at all: it reads as coverage."""
    job = ci["jobs"][job_name]
    script = job_script(job)

    assert 'grep -q "skipped"' in script, (
        f"{job_name} does not fail when its suite skips -- an unreachable service container "
        "would produce a green job that tested nothing"
    )
    assert "-rs" in script, f"{job_name} must run pytest with -rs so skips appear in the log"


@pytest.mark.parametrize("job_name", SERVICE_CONTAINER_JOBS)
def test_service_container_jobs_declare_a_service(ci, job_name):
    assert ci["jobs"][job_name].get("services"), f"{job_name} declares no service container"


def test_service_images_are_pinned(ci):
    """A floating tag turns an upstream release into a red build on an unrelated PR, which is
    the fastest way to teach everyone to ignore a job."""
    for job_name in SERVICE_CONTAINER_JOBS:
        for service, spec in ci["jobs"][job_name]["services"].items():
            image = spec["image"]
            assert ":" in image, f"{job_name}/{service} image {image!r} has no tag"
            assert not image.endswith(":latest"), f"{job_name}/{service} pins :latest"


def test_static_analysis_blocks_rather_than_warns(ci):
    """Unlike `audit`, which warns about advisories with no released fix, a bandit finding is
    code in this repo that someone here can act on."""
    job = ci["jobs"]["static-analysis"]

    assert job.get("continue-on-error") is not True
    assert "bandit" in job_script(job)


def test_secret_scanning_reads_the_whole_history(ci):
    """Scanning only the tip misses the case that matters: a credential committed once and
    'removed' in a later commit is still in the history, and still compromised."""
    job = ci["jobs"]["static-analysis"]
    checkout = next(
        s for s in job["steps"] if str(s.get("uses", "")).startswith("actions/checkout")
    )

    assert checkout.get("with", {}).get("fetch-depth") == 0


def test_the_deploy_waits_for_ci_rather_than_racing_it(deploy):
    """A `push` trigger ships commits whose tests are still running."""
    triggers = deploy["on"]

    assert "push" not in triggers, "deploy triggers on push -- it would race the test suite"
    assert triggers["workflow_run"]["workflows"] == ["CI"]
    assert triggers["workflow_run"]["branches"] == ["main"]


def test_the_deploy_refuses_a_failed_ci_run(deploy):
    """`workflow_run` fires on failure too. Without an explicit conclusion check, a red CI
    run deploys -- which is the single worst bug a deploy workflow can have."""
    condition = deploy["jobs"]["deploy"]["if"]

    assert "workflow_run.conclusion == 'success'" in condition


def test_the_deploy_ships_the_commit_ci_actually_tested(deploy):
    """By the time this runs another push may have landed; deploying the branch tip would
    ship something no green run ever covered."""
    script = job_script(deploy["jobs"]["deploy"])

    assert "workflow_run.head_sha" in script


def test_the_deploy_verifies_readiness_not_just_acceptance(deploy):
    """A 200 from the control plane means the request was accepted, not that the service
    works. `/ready` is the one that pings Chroma and the embedding provider."""
    script = job_script(deploy["jobs"]["deploy"])

    assert "/ready" in script
    assert "/health" in script


def test_the_deploy_skips_rather_than_fails_when_unconfigured(deploy):
    """A fork, or this repo before anyone connects a host, should not carry a permanently red
    badge for a deployment it was never meant to do."""
    script = job_script(deploy["jobs"]["deploy"])

    assert "enabled=false" in script
    assert "::notice::" in script


def test_deploys_do_not_run_concurrently_or_cancel_each_other(deploy):
    """Interrupting a deploy mid-rollout leaves the service half-updated."""
    concurrency = deploy["concurrency"]

    assert concurrency["cancel-in-progress"] is False


def test_a_rollback_path_exists(deploy):
    """Continuous deployment without a rollback is just continuous deployment of bugs."""
    dispatch = deploy["on"]["workflow_dispatch"]

    assert "commit" in dispatch["inputs"], "no way to deploy a specific (known-good) SHA"
