"""Tests focused on security rules."""

import pytest
from docker_optimizer.rules import (
    Severity,
    rule_no_user,
    rule_secret_in_env,
    rule_copy_dotenv,
    rule_run_curl_pipe_sh,
    run_all,
)
from docker_optimizer.parser import parse


def _df(content):
    return parse(content)


class TestNoUser:
    def test_no_user_directive(self):
        df = _df("FROM alpine:3.18\nCMD [\"sh\"]\n")
        findings = rule_no_user(df)
        assert len(findings) == 1
        assert findings[0].severity == Severity.CRITICAL

    def test_user_root_is_critical(self):
        df = _df("FROM alpine:3.18\nUSER root\nCMD [\"sh\"]\n")
        findings = rule_no_user(df)
        assert len(findings) == 1
        assert findings[0].severity == Severity.CRITICAL

    def test_user_0_is_critical(self):
        df = _df("FROM alpine:3.18\nUSER 0\n")
        findings = rule_no_user(df)
        assert len(findings) == 1

    def test_non_root_user_ok(self):
        df = _df("FROM alpine:3.18\nUSER appuser\n")
        findings = rule_no_user(df)
        assert findings == []

    def test_numeric_nonroot_ok(self):
        df = _df("FROM alpine:3.18\nUSER 1001:1001\n")
        findings = rule_no_user(df)
        assert findings == []


class TestSecretInEnv:
    def test_password_var(self):
        df = _df("FROM alpine:3.18\nENV DB_PASSWORD=secret\n")
        findings = rule_secret_in_env(df)
        assert len(findings) == 1
        assert findings[0].severity == Severity.CRITICAL

    def test_api_key_var(self):
        df = _df("FROM alpine:3.18\nENV API_KEY=abc123\n")
        findings = rule_secret_in_env(df)
        assert len(findings) == 1

    def test_safe_env_var(self):
        df = _df("FROM alpine:3.18\nENV APP_PORT=8080\n")
        findings = rule_secret_in_env(df)
        assert findings == []

    def test_case_insensitive(self):
        df = _df("FROM alpine:3.18\nENV db_password=x\n")
        findings = rule_secret_in_env(df)
        assert len(findings) == 1


class TestCopyDotenv:
    def test_copy_dotenv(self):
        df = _df("FROM alpine:3.18\nCOPY .env /app/.env\n")
        findings = rule_copy_dotenv(df)
        assert len(findings) == 1
        assert findings[0].severity == Severity.CRITICAL

    def test_copy_pem(self):
        df = _df("FROM alpine:3.18\nCOPY server.pem /certs/server.pem\n")
        findings = rule_copy_dotenv(df)
        assert len(findings) == 1

    def test_copy_normal_file_ok(self):
        df = _df("FROM alpine:3.18\nCOPY app.py /app/app.py\n")
        findings = rule_copy_dotenv(df)
        assert findings == []


class TestCurlPipeSh:
    def test_curl_pipe_bash(self):
        df = _df("FROM ubuntu:22.04\nRUN curl https://x.com/install.sh | bash\n")
        findings = rule_run_curl_pipe_sh(df)
        assert len(findings) == 1
        assert findings[0].severity == Severity.CRITICAL

    def test_wget_pipe_sh(self):
        df = _df("FROM ubuntu:22.04\nRUN wget -q https://x.com/setup | sh\n")
        findings = rule_run_curl_pipe_sh(df)
        assert len(findings) == 1

    def test_safe_curl(self):
        df = _df("FROM ubuntu:22.04\nRUN curl -o /tmp/file.txt https://example.com/data.txt\n")
        findings = rule_run_curl_pipe_sh(df)
        assert findings == []
