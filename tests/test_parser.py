"""Test cases for Dockerfile parser."""

from docker_optimizer.parser import DockerfileParser


class TestDockerfileParser:
    """Test cases for DockerfileParser."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = DockerfileParser()

    def test_parse_basic_dockerfile(self):
        """Test parsing a basic Dockerfile."""
        dockerfile_content = """
FROM ubuntu:20.04
RUN apt-get update
COPY . /app
WORKDIR /app
EXPOSE 8080
"""

        instructions = self.parser.parse(dockerfile_content)

        assert len(instructions) == 5
        assert instructions[0]["instruction"] == "FROM"
        assert instructions[0]["value"] == "ubuntu:20.04"
        assert instructions[1]["instruction"] == "RUN"
        assert instructions[1]["value"] == "apt-get update"

    def test_parse_with_comments_and_empty_lines(self):
        """Test parsing Dockerfile with comments and empty lines."""
        dockerfile_content = """
# Base image
FROM alpine:3.18

# Update packages
RUN apk update

# Copy application
COPY . /app
"""

        instructions = self.parser.parse(dockerfile_content)

        assert len(instructions) == 3
        assert all(
            inst["instruction"] in ["FROM", "RUN", "COPY"] for inst in instructions
        )

    def test_extract_from_instructions(self):
        """Test extracting FROM instructions."""
        dockerfile_content = """
FROM node:18 AS builder
FROM alpine:3.18
"""

        instructions = self.parser.parse(dockerfile_content)
        from_instructions = self.parser.extract_from_instructions(instructions)

        assert len(from_instructions) == 2
        assert "node:18 AS builder" in from_instructions
        assert "alpine:3.18" in from_instructions

    def test_extract_run_instructions(self):
        """Test extracting RUN instructions."""
        dockerfile_content = """
FROM ubuntu:20.04
RUN apt-get update
RUN apt-get install -y curl
COPY . /app
RUN chmod +x /app/start.sh
"""

        instructions = self.parser.parse(dockerfile_content)
        run_instructions = self.parser.extract_run_instructions(instructions)

        assert len(run_instructions) == 3
        assert "apt-get update" in run_instructions
        assert "apt-get install -y curl" in run_instructions
        assert "chmod +x /app/start.sh" in run_instructions

    def test_extract_copy_instructions(self):
        """Test extracting COPY instructions."""
        dockerfile_content = """
FROM ubuntu:20.04
COPY package.json /app/
COPY src/ /app/src/
ADD https://example.com/file.tar.gz /tmp/
"""

        instructions = self.parser.parse(dockerfile_content)
        copy_instructions = self.parser.extract_copy_instructions(instructions)

        assert len(copy_instructions) == 2
        assert "package.json /app/" in copy_instructions
        assert "src/ /app/src/" in copy_instructions

    def test_has_instruction(self):
        """Test checking for instruction existence."""
        dockerfile_content = """
FROM ubuntu:20.04
RUN apt-get update
COPY . /app
USER 1000
"""

        instructions = self.parser.parse(dockerfile_content)

        assert self.parser.has_instruction(instructions, "FROM")
        assert self.parser.has_instruction(instructions, "RUN")
        assert self.parser.has_instruction(instructions, "COPY")
        assert self.parser.has_instruction(instructions, "USER")
        assert not self.parser.has_instruction(instructions, "WORKDIR")

    def test_parse_empty_dockerfile(self):
        """Test parsing empty Dockerfile."""
        instructions = self.parser.parse("")
        assert len(instructions) == 0

    def test_parse_comments_only(self):
        """Test parsing Dockerfile with only comments."""
        dockerfile_content = """
# This is a comment
# Another comment
"""

        instructions = self.parser.parse(dockerfile_content)
        assert len(instructions) == 0
