"""Rule-based optimizer — 18 concrete rules for Dockerfile analysis.

Each rule is a function that receives a ParsedDockerfile and returns a list
of Finding objects (possibly empty).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional

from .parser import ParsedDockerfile, Instruction


class Severity(str, Enum):
    CRITICAL = "critical"
    WARNING  = "warning"
    INFO     = "info"


@dataclass
class Finding:
    rule_id: str
    severity: Severity
    line_no: Optional[int]
    title: str
    detail: str
    suggestion: str


RuleFunc = Callable[[ParsedDockerfile], List[Finding]]

# ─────────────────────────────────────────────
# Individual rules
# ─────────────────────────────────────────────

def rule_no_latest_tag(df: ParsedDockerfile) -> List[Finding]:
    """RULE-01 — Pinned base image tags."""
    findings = []
    for instr in df.from_instructions:
        image = instr.value.split()[0]  # drop AS alias
        # scratch is ok, images with @ digest are ok
        if image.lower() == "scratch":
            continue
        if "@" in image:
            continue
        if ":" not in image or image.endswith(":latest"):
            tag = "latest" if ":" not in image else image.split(":", 1)[1]
            findings.append(Finding(
                rule_id="RULE-01",
                severity=Severity.WARNING,
                line_no=instr.line_no,
                title="Unpinned base image tag",
                detail=f"FROM {image!r} uses tag '{tag}'. Floating tags break reproducibility.",
                suggestion=(
                    f"Pin to a specific version, e.g. `FROM {image.split(':')[0]}:3.12-slim` "
                    "or use a digest (`@sha256:…`)."
                ),
            ))
    return findings


def rule_run_chaining(df: ParsedDockerfile) -> List[Finding]:
    """RULE-02 — Consecutive RUN instructions create unnecessary layers."""
    runs = df.run_instructions
    if len(runs) < 2:
        return []
    # Find groups of consecutive RUN instructions (no other layer-creating instr between them)
    groups: List[List[Instruction]] = []
    current_group: List[Instruction] = []
    layer_cmds = {"RUN", "COPY", "ADD"}
    for instr in df.instructions:
        if instr.cmd == "RUN":
            current_group.append(instr)
        elif instr.cmd in layer_cmds:
            if len(current_group) > 1:
                groups.append(current_group)
            current_group = []
    if len(current_group) > 1:
        groups.append(current_group)

    findings = []
    for group in groups:
        findings.append(Finding(
            rule_id="RULE-02",
            severity=Severity.WARNING,
            line_no=group[0].line_no,
            title="Multiple consecutive RUN instructions",
            detail=(
                f"{len(group)} consecutive RUN instructions (lines "
                f"{group[0].line_no}–{group[-1].line_no}) each create a separate layer."
            ),
            suggestion=(
                "Combine with `&&` and line continuations:\n"
                "  RUN apt-get update \\\n"
                "   && apt-get install -y curl \\\n"
                "   && rm -rf /var/lib/apt/lists/*"
            ),
        ))
    return findings


def rule_apt_no_clean(df: ParsedDockerfile) -> List[Finding]:
    """RULE-03 — apt-get without cache cleanup."""
    findings = []
    for instr in df.run_instructions:
        v = instr.value
        if "apt-get" in v and "apt-get install" in v:
            if "rm -rf /var/lib/apt/lists" not in v:
                findings.append(Finding(
                    rule_id="RULE-03",
                    severity=Severity.WARNING,
                    line_no=instr.line_no,
                    title="apt-get cache not cleaned",
                    detail="apt-get install without `rm -rf /var/lib/apt/lists/*` leaves package lists in the image.",
                    suggestion="Add `&& rm -rf /var/lib/apt/lists/*` to the same RUN layer.",
                ))
    return findings


def rule_apt_no_install_recommends(df: ParsedDockerfile) -> List[Finding]:
    """RULE-04 — apt-get install without --no-install-recommends."""
    findings = []
    for instr in df.run_instructions:
        v = instr.value
        if "apt-get install" in v and "--no-install-recommends" not in v:
            findings.append(Finding(
                rule_id="RULE-04",
                severity=Severity.INFO,
                line_no=instr.line_no,
                title="apt-get install without --no-install-recommends",
                detail="Installing recommended packages adds unnecessary size.",
                suggestion="Use `apt-get install --no-install-recommends <packages>`.",
            ))
    return findings


def rule_pip_no_cache(df: ParsedDockerfile) -> List[Finding]:
    """RULE-05 — pip install without --no-cache-dir."""
    findings = []
    for instr in df.run_instructions:
        v = instr.value
        if re.search(r"pip\d*\s+install", v) and "--no-cache-dir" not in v:
            findings.append(Finding(
                rule_id="RULE-05",
                severity=Severity.INFO,
                line_no=instr.line_no,
                title="pip install without --no-cache-dir",
                detail="pip caches downloaded wheels inside the image by default.",
                suggestion="Use `pip install --no-cache-dir <packages>`.",
            ))
    return findings


def rule_npm_cache(df: ParsedDockerfile) -> List[Finding]:
    """RULE-06 — npm install without cache cleanup."""
    findings = []
    for instr in df.run_instructions:
        v = instr.value
        if re.search(r"npm\s+install", v) and "npm cache clean" not in v:
            findings.append(Finding(
                rule_id="RULE-06",
                severity=Severity.INFO,
                line_no=instr.line_no,
                title="npm install without cache cleanup",
                detail="`npm install` leaves a cache in /root/.npm, bloating the image.",
                suggestion=(
                    "Add `&& npm cache clean --force` in the same RUN layer, "
                    "or use `npm ci --omit=dev` for production images."
                ),
            ))
    return findings


def rule_no_user(df: ParsedDockerfile) -> List[Finding]:
    """RULE-07 — Container runs as root (no USER instruction)."""
    if df.has_cmd("USER"):
        # Check if the last USER instruction isn't root/0
        users = df.user_instructions
        last_user = users[-1].value.strip()
        if last_user in ("root", "0", "0:0"):
            return [Finding(
                rule_id="RULE-07",
                severity=Severity.CRITICAL,
                line_no=users[-1].line_no,
                title="Container runs as root",
                detail=f"Last USER instruction sets user to '{last_user}'.",
                suggestion="Add a non-root user and switch to it:\n  RUN useradd -m appuser\n  USER appuser",
            )]
        return []
    return [Finding(
        rule_id="RULE-07",
        severity=Severity.CRITICAL,
        line_no=None,
        title="No USER instruction — container runs as root",
        detail="Without a USER directive the container process runs as root, which is a security risk.",
        suggestion="Add a non-root user before CMD/ENTRYPOINT:\n  RUN useradd -m appuser\n  USER appuser",
    )]


def rule_secret_in_env(df: ParsedDockerfile) -> List[Finding]:
    """RULE-08 — Secrets in ENV variables."""
    secret_patterns = re.compile(
        r'(password|passwd|secret|api_key|apikey|api_secret|'
        r'private_key|access_key|auth_token|db_pass|database_password|jwt_secret)',
        re.IGNORECASE,
    )
    findings = []
    for instr in df.env_instructions:
        v = instr.value
        if secret_patterns.search(v):
            findings.append(Finding(
                rule_id="RULE-08",
                severity=Severity.CRITICAL,
                line_no=instr.line_no,
                title="Potential secret in ENV instruction",
                detail=f"ENV instruction may contain a secret: {v[:80]}",
                suggestion=(
                    "Do NOT bake secrets into the image. Use Docker BuildKit secrets "
                    "(`--secret id=…`), runtime env vars, or a secrets manager."
                ),
            ))
    return findings


def rule_copy_dotenv(df: ParsedDockerfile) -> List[Finding]:
    """RULE-09 — Copying .env or credential files into the image."""
    risky = re.compile(r'\.env(\b|$)|credentials|\.pem|\.key|\.p12|id_rsa|\.aws', re.IGNORECASE)
    findings = []
    for instr in [*df.copy_instructions, *df.add_instructions]:
        if risky.search(instr.value):
            findings.append(Finding(
                rule_id="RULE-09",
                severity=Severity.CRITICAL,
                line_no=instr.line_no,
                title=f"Potentially sensitive file copied into image ({instr.cmd})",
                detail=f"{instr.cmd} {instr.value[:80]}",
                suggestion=(
                    "Never COPY credential/secret files into the image. "
                    "Add them to .dockerignore and inject at runtime."
                ),
            ))
    return findings


def rule_add_instead_of_copy(df: ParsedDockerfile) -> List[Finding]:
    """RULE-10 — ADD used where COPY is sufficient."""
    findings = []
    for instr in df.add_instructions:
        v = instr.value.strip()
        # ADD has special powers for URLs and tar extraction — but most uses don't need them
        is_url = re.match(r'https?://', v)
        is_tar = re.search(r'\.(tar|tar\.gz|tgz|tar\.bz2|tar\.xz)', v)
        if not is_url and not is_tar:
            findings.append(Finding(
                rule_id="RULE-10",
                severity=Severity.INFO,
                line_no=instr.line_no,
                title="ADD used where COPY is preferred",
                detail=(
                    f"ADD {v[:60]} — ADD has implicit URL-fetch and auto-extraction behaviour "
                    "that can introduce unexpected results."
                ),
                suggestion="Replace ADD with COPY unless you need URL fetching or auto-tar extraction.",
            ))
    return findings


def rule_alpine_or_slim(df: ParsedDockerfile) -> List[Finding]:
    """RULE-11 — Suggest slimmer base image variants."""
    findings = []
    heavy = {
        "ubuntu": "ubuntu:24.04 or debian:bookworm-slim",
        "debian": "debian:bookworm-slim",
        "centos": "alpine or a distroless image",
        "fedora": "alpine or a distroless image",
        "amazonlinux": "alpine or a distroless image",
    }
    for instr in df.from_instructions:
        image = instr.value.split()[0].lower()
        if "@" in image:
            continue
        name = image.split(":")[0].split("/")[-1]  # strip registry prefix
        for heavy_name, suggestion in heavy.items():
            if name == heavy_name and "slim" not in image and "alpine" not in image:
                findings.append(Finding(
                    rule_id="RULE-11",
                    severity=Severity.INFO,
                    line_no=instr.line_no,
                    title=f"Heavy base image: {image}",
                    detail=f"'{image}' is a full OS image; your final image may be much larger than necessary.",
                    suggestion=f"Consider {suggestion} to reduce image size.",
                ))
                break
    return findings


def rule_multistage_opportunity(df: ParsedDockerfile) -> List[Finding]:
    """RULE-12 — Suggest multi-stage builds when build tools are present."""
    if df.is_multistage:
        return []  # already multi-stage
    build_tools = re.compile(
        r'\b(gcc|g\+\+|make|cmake|cargo\s+build|go\s+build|mvn|gradle|'
        r'npm\s+run\s+build|yarn\s+build|build-essential|rustc)\b'
    )
    for instr in df.run_instructions:
        if build_tools.search(instr.value):
            return [Finding(
                rule_id="RULE-12",
                severity=Severity.WARNING,
                line_no=instr.line_no,
                title="Build tools present — consider multi-stage build",
                detail=(
                    "Build-time dependencies (compilers, build systems) are included in the final image."
                ),
                suggestion=(
                    "Use a multi-stage build:\n"
                    "  FROM builder-image AS build\n"
                    "  RUN cargo build --release\n"
                    "\n"
                    "  FROM gcr.io/distroless/cc AS final\n"
                    "  COPY --from=build /app/target/release/app /app\n"
                    "  CMD [\"/app\"]"
                ),
            )]
    return []


def rule_dockerignore(df: ParsedDockerfile) -> List[Finding]:
    """RULE-13 — Remind to maintain .dockerignore."""
    # We can only flag the pattern COPY . . as a hint
    findings = []
    for instr in df.copy_instructions:
        parts = instr.value.split()
        if len(parts) >= 2 and parts[0] in (".", "./") and parts[1] in (".", "./", "/app", "/"):
            findings.append(Finding(
                rule_id="RULE-13",
                severity=Severity.INFO,
                line_no=instr.line_no,
                title="COPY . . — ensure .dockerignore is present",
                detail=(
                    "Copying the entire context without a .dockerignore can include "
                    ".git/, __pycache__/, node_modules/, .env files, etc."
                ),
                suggestion=(
                    "Create a .dockerignore file with at minimum:\n"
                    "  .git/\n  __pycache__/\n  *.pyc\n  node_modules/\n"
                    "  .env\n  .DS_Store\n  *.log"
                ),
            ))
    return findings


def rule_healthcheck(df: ParsedDockerfile) -> List[Finding]:
    """RULE-14 — No HEALTHCHECK instruction."""
    if not df.has_cmd("HEALTHCHECK"):
        return [Finding(
            rule_id="RULE-14",
            severity=Severity.INFO,
            line_no=None,
            title="No HEALTHCHECK instruction",
            detail="Without HEALTHCHECK, orchestrators cannot detect unhealthy containers.",
            suggestion=(
                "Add a health check, e.g.:\n"
                "  HEALTHCHECK --interval=30s --timeout=5s \\\n"
                "    CMD curl -f http://localhost:8080/health || exit 1"
            ),
        )]
    return []


def rule_expose_all_ports(df: ParsedDockerfile) -> List[Finding]:
    """RULE-15 — EXPOSE 0.0.0.0 or very broad port ranges."""
    findings = []
    for instr in [i for i in df.instructions if i.cmd == "EXPOSE"]:
        # Flag if a range like 1-65535 is used
        if re.search(r'\d+\s*-\s*\d+', instr.value):
            lo, hi = map(int, re.findall(r'\d+', instr.value)[:2])
            if hi - lo > 100:
                findings.append(Finding(
                    rule_id="RULE-15",
                    severity=Severity.WARNING,
                    line_no=instr.line_no,
                    title="Very broad EXPOSE port range",
                    detail=f"EXPOSE {instr.value} exposes a large range of ports.",
                    suggestion="Expose only the specific ports your service needs.",
                ))
    return findings


def rule_run_curl_pipe_sh(df: ParsedDockerfile) -> List[Finding]:
    """RULE-16 — Dangerous curl/wget | sh pattern."""
    pipe_sh = re.compile(r'(curl|wget)\s.+\|\s*(ba)?sh')
    findings = []
    for instr in df.run_instructions:
        if pipe_sh.search(instr.value):
            findings.append(Finding(
                rule_id="RULE-16",
                severity=Severity.CRITICAL,
                line_no=instr.line_no,
                title="Remote script piped directly to shell",
                detail=f"RUN {instr.value[:80]} — blindly executes remote code.",
                suggestion=(
                    "Download the script first, verify its checksum or signature, "
                    "then execute. Never pipe untrusted URLs directly into a shell."
                ),
            ))
    return findings


def rule_sudo_in_run(df: ParsedDockerfile) -> List[Finding]:
    """RULE-17 — sudo used inside container build."""
    findings = []
    for instr in df.run_instructions:
        if re.search(r'\bsudo\b', instr.value):
            findings.append(Finding(
                rule_id="RULE-17",
                severity=Severity.INFO,
                line_no=instr.line_no,
                title="sudo used in RUN instruction",
                detail="Build steps typically run as root; sudo is unnecessary and may not be installed.",
                suggestion="Remove sudo. RUN instructions in Dockerfiles already run as root by default.",
            ))
    return findings


def rule_workdir_vs_cd(df: ParsedDockerfile) -> List[Finding]:
    """RULE-18 — RUN cd … used instead of WORKDIR."""
    findings = []
    for instr in df.run_instructions:
        if re.match(r'cd\s+\S+\s*&&', instr.value.strip()):
            findings.append(Finding(
                rule_id="RULE-18",
                severity=Severity.INFO,
                line_no=instr.line_no,
                title="RUN cd … used instead of WORKDIR",
                detail="`cd` only changes the directory for that RUN instruction.",
                suggestion=(
                    "Use WORKDIR to set the working directory persistently:\n"
                    "  WORKDIR /app"
                ),
            ))
    return findings


# ─────────────────────────────────────────────
# Rule registry
# ─────────────────────────────────────────────

ALL_RULES: List[RuleFunc] = [
    rule_no_latest_tag,
    rule_run_chaining,
    rule_apt_no_clean,
    rule_apt_no_install_recommends,
    rule_pip_no_cache,
    rule_npm_cache,
    rule_no_user,
    rule_secret_in_env,
    rule_copy_dotenv,
    rule_add_instead_of_copy,
    rule_alpine_or_slim,
    rule_multistage_opportunity,
    rule_dockerignore,
    rule_healthcheck,
    rule_expose_all_ports,
    rule_run_curl_pipe_sh,
    rule_sudo_in_run,
    rule_workdir_vs_cd,
]


def run_all(df: ParsedDockerfile) -> List[Finding]:
    """Run every rule and return the combined findings."""
    results: List[Finding] = []
    for rule in ALL_RULES:
        results.extend(rule(df))
    # Sort: critical first, then by line number
    severity_order = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}
    results.sort(key=lambda f: (severity_order[f.severity], f.line_no or 9999))
    return results
