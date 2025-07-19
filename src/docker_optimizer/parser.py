"""Dockerfile parsing utilities."""

from typing import Any, Dict, List


class DockerfileParser:
    """Parser for Dockerfile content."""

    def parse(self, dockerfile_content: str) -> List[Dict[str, Any]]:
        """Parse Dockerfile content into structured format.

        Args:
            dockerfile_content: Raw Dockerfile content

        Returns:
            List of parsed instructions with metadata
        """
        instructions = []
        lines = dockerfile_content.strip().split("\n")

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Parse instruction
            parts = line.split(None, 1)
            if not parts:
                continue

            instruction = parts[0].upper()
            value = parts[1] if len(parts) > 1 else ""

            instructions.append(
                {
                    "instruction": instruction,
                    "value": value,
                    "line_number": line_num,
                    "raw_line": line,
                }
            )

        return instructions

    def extract_from_instructions(
        self, instructions: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract all FROM instructions."""
        return [inst["value"] for inst in instructions if inst["instruction"] == "FROM"]

    def extract_run_instructions(self, instructions: List[Dict[str, Any]]) -> List[str]:
        """Extract all RUN instructions."""
        return [inst["value"] for inst in instructions if inst["instruction"] == "RUN"]

    def extract_copy_instructions(
        self, instructions: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract all COPY instructions."""
        return [inst["value"] for inst in instructions if inst["instruction"] == "COPY"]

    def has_instruction(
        self, instructions: List[Dict[str, Any]], instruction_name: str
    ) -> bool:
        """Check if a specific instruction exists."""
        return any(
            inst["instruction"] == instruction_name.upper() for inst in instructions
        )
