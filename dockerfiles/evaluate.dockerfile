FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY uv.lock uv.lock
COPY src/ src/

ENV PYTHONUNBUFFERED=1

WORKDIR /
RUN uv sync --locked --no-cache

ENTRYPOINT ["uv", "run", "src/template/evaluate.py", "models/model.pth"]