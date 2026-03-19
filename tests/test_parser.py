"""Tests for the Dockerfile parser."""

import pytest
from docker_optimizer.parser import parse


def test_basic_from():
    df = parse("FROM ubuntu:22.04\nRUN echo hi\n")
    assert len(df.instructions) == 2
    assert df.instructions[0].cmd == "FROM"
    assert df.instructions[0].value == "ubuntu:22.04"
    assert df.instructions[1].cmd == "RUN"


def test_continuation_lines():
    content = "RUN apt-get update \\\n && apt-get install -y curl\n"
    df = parse(content)
    assert len(df.instructions) == 1
    assert df.instructions[0].cmd == "RUN"
    assert "apt-get install" in df.instructions[0].value


def test_comments_and_blanks_skipped():
    content = (
        "# comment\n"
        "\n"
        "FROM alpine:3.18\n"
        "# another comment\n"
        "CMD [\"sh\"]\n"
    )
    df = parse(content)
    assert len(df.instructions) == 2
    assert df.instructions[0].cmd == "FROM"
    assert df.instructions[1].cmd == "CMD"


def test_base_image_property():
    df = parse("FROM python:3.12-slim AS builder\n")
    assert df.base_image == "python:3.12-slim"


def test_multistage_detection():
    content = "FROM golang:1.21 AS build\nFROM gcr.io/distroless/base\n"
    df = parse(content)
    assert df.is_multistage is True
    assert len(df.from_instructions) == 2


def test_has_cmd():
    df = parse("FROM alpine:3.18\nUSER appuser\n")
    assert df.has_cmd("USER") is True
    assert df.has_cmd("HEALTHCHECK") is False


def test_line_numbers():
    content = "# skip\nFROM alpine:3.18\nRUN echo hi\n"
    df = parse(content)
    assert df.instructions[0].line_no == 2  # FROM is on line 2
    assert df.instructions[1].line_no == 3


def test_empty_dockerfile():
    df = parse("# just a comment\n\n")
    assert df.instructions == []
