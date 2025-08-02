# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for the Docker Optimizer Agent project.

## ADR Format

Each ADR follows this template:

```markdown
# ADR-XXXX: Decision Title

## Status
[Proposed | Accepted | Deprecated | Superseded]

## Context
What is the issue that we're seeing that is motivating this decision or change?

## Decision
What is the change that we're proposing or have agreed to implement?

## Consequences
What becomes easier or more difficult to do and any risks introduced by this change?
```

## Index

- [ADR-0001: Use Python for Core Implementation](0001-python-core-implementation.md)
- [ADR-0002: External Security Scanner Integration](0002-external-security-integration.md)
- [ADR-0003: Multi-stage Build Optimization Strategy](0003-multistage-build-strategy.md)

## Contributing

When making significant architectural decisions:

1. Create a new ADR using the next sequential number
2. Follow the standard template format
3. Include the ADR in this index
4. Reference the ADR in relevant documentation and code comments