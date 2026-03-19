# docker-opt — Dockerfile Analyser

Static analysis for Dockerfiles. Catches size bloat, security holes, and build-speed issues before they hit production.

No external API calls. No Docker daemon required. Just parse + rules.

---

## Install

```bash
pip install docker-optimizer-agent
```

Or from source:

```bash
git clone https://github.com/danieleschmidt/docker-optimizer-agent
cd docker-optimizer-agent
pip install -e .
```

---

## Quick Start

```bash
docker-opt analyze Dockerfile
```

Output:

```
╭─ docker-opt › Dockerfile ─╮
│   Health score: 0/100     │
╰───────────────────────────╯
  4 critical  4 warning  4 info  │  base image: ubuntu:latest  │  instructions: 9

  Rule       Sev           Line   Title
 ─────────────────────────────────────────────────────────────────────────
  RULE-08    🔴 critical      3   Potential secret in ENV instruction
  RULE-09    🔴 critical      9   Potentially sensitive file copied into image
  RULE-16    🔴 critical     12   Remote script piped directly to shell
  RULE-07    🔴 critical      —   No USER instruction — container runs as root
  RULE-01    🟡 warning       1   Unpinned base image tag
  RULE-02    🟡 warning       5   Multiple consecutive RUN instructions
  RULE-03    🟡 warning       6   apt-get cache not cleaned
  RULE-11    🔵 info          1   Heavy base image: ubuntu:latest
  RULE-04    🔵 info          6   apt-get install without --no-install-recommends
  RULE-14    🔵 info          —   No HEALTHCHECK instruction

─── 🔴 RULE-08 — Potential secret in ENV instruction  line 3 ───
  ENV instruction may contain a secret: DB_PASSWORD=supersecret123
  ▶  Do NOT bake secrets into the image. Use Docker BuildKit secrets
     (`--secret id=…`), runtime env vars, or a secrets manager.

─── 🔴 RULE-09 — Potentially sensitive file copied into image  line 9 ───
  COPY .env /app/.env
  ▶  Never COPY credential/secret files into the image. Add them to
     .dockerignore and inject at runtime.

─── 🔴 RULE-16 — Remote script piped directly to shell  line 12 ───
  RUN curl https://setup.example.com/install.sh | bash
  ▶  Download the script first, verify its checksum or signature,
     then execute. Never pipe untrusted URLs directly into a shell.
```

---

## Options

```
docker-opt analyze [OPTIONS] [DOCKERFILE]

  Analyze DOCKERFILE and report optimization suggestions.
  Defaults to 'Dockerfile' in the current directory.

Options:
  --json                Output findings as JSON.
  --min-severity TEXT   Only show findings at/above this level (info/warning/critical).
  -q, --quiet           Exit silently; only set exit code.
  --help                Show this message and exit.
```

**Exit codes:** `0` = no critical findings. `1` = one or more critical findings.

---

## JSON Output

```bash
docker-opt analyze --json Dockerfile
```

```json
{
  "file": "Dockerfile",
  "score": 0,
  "base_image": "ubuntu:latest",
  "total_instructions": 9,
  "findings": [
    {
      "rule_id": "RULE-08",
      "severity": "critical",
      "line_no": 3,
      "title": "Potential secret in ENV instruction",
      "detail": "ENV instruction may contain a secret: DB_PASSWORD=supersecret123",
      "suggestion": "Do NOT bake secrets into the image. Use Docker BuildKit secrets…"
    }
  ]
}
```

---

## Rules Reference

| Rule    | Severity | Category | Description |
|---------|----------|----------|-------------|
| RULE-01 | warning  | reproducibility | Unpinned/floating base image tag (`latest` or no tag) |
| RULE-02 | warning  | size/speed | Multiple consecutive RUN instructions — each adds a layer |
| RULE-03 | warning  | size | `apt-get install` without `rm -rf /var/lib/apt/lists/*` |
| RULE-04 | info     | size | `apt-get install` without `--no-install-recommends` |
| RULE-05 | info     | size | `pip install` without `--no-cache-dir` |
| RULE-06 | info     | size | `npm install` without cache cleanup |
| RULE-07 | critical | security | No USER instruction — container runs as root |
| RULE-08 | critical | security | Secret-like value in ENV (password, api_key, token…) |
| RULE-09 | critical | security | Sensitive file COPYed into image (.env, .pem, credentials…) |
| RULE-10 | info     | correctness | ADD used where COPY is preferred |
| RULE-11 | info     | size | Heavy base image (ubuntu, debian, centos) — slimmer alternatives exist |
| RULE-12 | warning  | size | Build tools present but no multi-stage build |
| RULE-13 | info     | correctness | `COPY . .` without a .dockerignore file |
| RULE-14 | info     | reliability | No HEALTHCHECK instruction |
| RULE-15 | warning  | security | Very broad EXPOSE port range |
| RULE-16 | critical | security | Remote script piped directly to shell (`curl … | bash`) |
| RULE-17 | info     | correctness | `sudo` used inside a RUN instruction |
| RULE-18 | info     | correctness | `RUN cd …` used instead of WORKDIR |

---

## Python API

```python
from docker_optimizer import analyze, analyze_file

# From a string
report = analyze(dockerfile_content)

# From a file
report = analyze_file("path/to/Dockerfile")

print(f"Score: {report.score}/100")
for finding in report.findings:
    print(f"[{finding.severity.value.upper()}] {finding.rule_id}: {finding.title}")
    print(f"  → {finding.suggestion}\n")
```

---

## Example: Before & After

**Before (score: 0/100)**

```dockerfile
FROM ubuntu:latest
ENV DB_PASSWORD=mypassword
RUN apt-get update
RUN apt-get install -y python3 pip
COPY . .
CMD ["python3", "app.py"]
```

**After (score: 87/100)**

```dockerfile
FROM ubuntu:24.04
WORKDIR /app
COPY requirements.txt .
RUN apt-get update \
 && apt-get install --no-install-recommends -y python3 python3-pip \
 && rm -rf /var/lib/apt/lists/*
COPY . .
RUN useradd -m appuser
USER appuser
HEALTHCHECK --interval=30s --timeout=5s \
  CMD curl -f http://localhost:8000/health || exit 1
CMD ["python3", "app.py"]
```

Secrets are injected at runtime via environment variables or a secrets manager, not baked into the image.

---

## Development

```bash
git clone https://github.com/danieleschmidt/docker-optimizer-agent
cd docker-optimizer-agent
pip install -e ".[dev]"
pytest
```

---

## License

MIT
