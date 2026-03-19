"""Dockerfile parser — turns raw text into a structured instruction list."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Instruction:
    """A single parsed Dockerfile instruction."""

    cmd: str          # Uppercase command: FROM, RUN, COPY, …
    value: str        # Everything after the command
    line_no: int      # 1-based line number of the *first* line
    raw: str          # Original text (may span continuation lines)


@dataclass
class ParsedDockerfile:
    """Result of parsing a Dockerfile."""

    instructions: List[Instruction] = field(default_factory=list)

    # Convenience accessors
    @property
    def from_instructions(self) -> List[Instruction]:
        return [i for i in self.instructions if i.cmd == "FROM"]

    @property
    def run_instructions(self) -> List[Instruction]:
        return [i for i in self.instructions if i.cmd == "RUN"]

    @property
    def copy_instructions(self) -> List[Instruction]:
        return [i for i in self.instructions if i.cmd == "COPY"]

    @property
    def add_instructions(self) -> List[Instruction]:
        return [i for i in self.instructions if i.cmd == "ADD"]

    @property
    def env_instructions(self) -> List[Instruction]:
        return [i for i in self.instructions if i.cmd == "ENV"]

    @property
    def user_instructions(self) -> List[Instruction]:
        return [i for i in self.instructions if i.cmd == "USER"]

    @property
    def base_image(self) -> Optional[str]:
        """The first FROM image (before AS alias)."""
        for instr in self.from_instructions:
            parts = re.split(r'\s+AS\s+', instr.value, flags=re.IGNORECASE)
            return parts[0].strip()
        return None

    @property
    def is_multistage(self) -> bool:
        return len(self.from_instructions) > 1

    def has_cmd(self, cmd: str) -> bool:
        return any(i.cmd == cmd.upper() for i in self.instructions)


def parse(content: str) -> ParsedDockerfile:
    """Parse Dockerfile text into a ParsedDockerfile.

    Handles:
    - Line continuations (backslash at end of line)
    - Comments (# …)
    - Blank lines
    - Here-docs are treated as plain text (no special handling needed for rules)
    """
    lines = content.splitlines()
    result = ParsedDockerfile()

    i = 0
    while i < len(lines):
        raw_line = lines[i]
        stripped = raw_line.strip()

        # Skip blanks and comments
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        # Collect continuation lines
        logical_line = stripped
        start_line_no = i + 1  # 1-based
        raw_parts = [raw_line]

        while logical_line.endswith("\\"):
            logical_line = logical_line[:-1].rstrip()  # drop the backslash
            i += 1
            if i >= len(lines):
                break
            next_stripped = lines[i].strip()
            if next_stripped.startswith("#"):
                # comment inside continuation — skip
                continue
            raw_parts.append(lines[i])
            logical_line = logical_line + " " + next_stripped

        # Split cmd from value
        parts = logical_line.split(None, 1)
        if not parts:
            i += 1
            continue

        cmd = parts[0].upper()
        value = parts[1] if len(parts) > 1 else ""

        result.instructions.append(
            Instruction(
                cmd=cmd,
                value=value,
                line_no=start_line_no,
                raw="\n".join(raw_parts),
            )
        )
        i += 1

    return result
