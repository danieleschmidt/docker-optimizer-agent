FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ ./src/

RUN pip install --no-cache-dir -e .

RUN useradd -m appuser
USER appuser

ENTRYPOINT ["docker-opt"]
CMD ["--help"]
