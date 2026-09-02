"""Tests that the shipped alert rules and Grafana dashboard match the metrics we export.

Monitoring config rots silently. A renamed metric leaves a panel showing "No data" and an
alert that can never fire — and an alert that can never fire is worse than no alert, because
its presence is taken as coverage. These tests fail the build instead.
"""

import json
import re
from pathlib import Path

import pytest
import yaml
from prometheus_client import REGISTRY

from rag_assistant import metrics  # noqa: F401  -- registers the collectors

OPS = Path(__file__).resolve().parents[1] / "ops"
ALERTS_PATH = OPS / "prometheus" / "alerts.yml"
DASHBOARD_PATH = OPS / "grafana" / "dashboard.json"

_METRIC_RE = re.compile(r"\brag_[a-z0-9_]+\b")


def declared_metric_names() -> set[str]:
    """Every series name the app can emit, including the derived `_bucket`/`_count`/`_sum`
    and `_total` names Prometheus generates for histograms and counters."""
    names: set[str] = set()
    for name in REGISTRY._names_to_collectors:
        if not name.startswith("rag_"):
            continue
        names.add(name)
        names.update({f"{name}_bucket", f"{name}_count", f"{name}_sum", f"{name}_total"})
        if name.endswith("_total"):
            names.add(name.removesuffix("_total"))
    return names


def referenced_metrics(text: str) -> set[str]:
    return set(_METRIC_RE.findall(text))


# ---- alert rules ----


def test_alert_rules_are_valid_yaml_with_the_expected_shape():
    payload = yaml.safe_load(ALERTS_PATH.read_text())

    assert payload["groups"]
    for group in payload["groups"]:
        assert group["name"]
        for rule in group["rules"]:
            assert rule["alert"], "every rule needs a name"
            assert rule["expr"], "every rule needs an expression"
            assert rule["labels"]["severity"] in {"critical", "warning"}
            assert rule["annotations"]["summary"]


def test_every_metric_an_alert_references_actually_exists():
    """A renamed metric leaves an alert that can never fire, whose presence reads as
    coverage."""
    declared = declared_metric_names()

    referenced = referenced_metrics(ALERTS_PATH.read_text())

    assert referenced <= declared, f"alerts reference unknown metrics: {referenced - declared}"


def test_alerts_all_have_a_for_duration():
    """Without `for`, a single scrape blip pages someone."""
    payload = yaml.safe_load(ALERTS_PATH.read_text())

    for group in payload["groups"]:
        for rule in group["rules"]:
            assert rule.get("for"), f"{rule['alert']} has no `for` duration"


def test_the_availability_and_correctness_alerts_are_critical():
    """Being down, failing most requests, or serving answers from a mismatched embedding
    space are the three cases worth waking someone for."""
    payload = yaml.safe_load(ALERTS_PATH.read_text())
    severities = {
        rule["alert"]: rule["labels"]["severity"]
        for group in payload["groups"]
        for rule in group["rules"]
    }

    assert severities["RagAssistantDown"] == "critical"
    assert severities["RagAssistantHighErrorRate"] == "critical"
    assert severities["RagAssistantNotReady"] == "critical"


def test_ratio_alerts_guard_against_division_by_zero():
    """A rate ratio with no traffic is 0/0. Without clamping, the expression is NaN and the
    alert quietly stops evaluating."""
    payload = yaml.safe_load(ALERTS_PATH.read_text())

    for group in payload["groups"]:
        for rule in group["rules"]:
            # Quoted label values are stripped first: `route="/api/v1/research"` contains a
            # slash that is a path separator, not a division operator.
            expression = re.sub(r'"[^"]*"', '""', rule["expr"])
            if "/" in expression and "rate(" in expression:
                assert "clamp_min" in expression, f"{rule['alert']} divides without clamping"


# ---- dashboard ----


def test_the_dashboard_is_valid_json_with_panels():
    dashboard = json.loads(DASHBOARD_PATH.read_text())

    assert dashboard["uid"] == "rag-assistant"
    assert len(dashboard["panels"]) >= 8


def test_every_metric_the_dashboard_references_actually_exists():
    declared = declared_metric_names()

    referenced = referenced_metrics(DASHBOARD_PATH.read_text())

    assert referenced <= declared, f"dashboard references unknown metrics: {referenced - declared}"


def test_every_panel_has_a_title_and_at_least_one_query():
    dashboard = json.loads(DASHBOARD_PATH.read_text())

    for panel in dashboard["panels"]:
        assert panel["title"]
        assert panel["targets"], f"panel {panel['title']!r} has no query"


def test_panels_do_not_overlap_on_the_grid():
    """Grafana silently reflows overlapping panels into an unreadable layout."""
    dashboard = json.loads(DASHBOARD_PATH.read_text())
    occupied: set[tuple[int, int]] = set()

    for panel in dashboard["panels"]:
        pos = panel["gridPos"]
        cells = {
            (x, y)
            for x in range(pos["x"], pos["x"] + pos["w"])
            for y in range(pos["y"], pos["y"] + pos["h"])
        }
        assert not (cells & occupied), f"panel {panel['title']!r} overlaps another"
        occupied |= cells


@pytest.mark.parametrize(
    "metric",
    [
        "rag_http_requests_total",
        "rag_http_request_duration_seconds",
        "rag_llm_calls_total",
        "rag_llm_tokens_total",
        "rag_graph_runs_total",
        "rag_feedback_total",
    ],
)
def test_the_dashboard_covers_the_metrics_worth_watching(metric):
    """Coverage in the other direction: exporting a metric nobody graphs is how a signal
    stays invisible until someone goes looking for it during an incident."""
    assert metric in DASHBOARD_PATH.read_text()
