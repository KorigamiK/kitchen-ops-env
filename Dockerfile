FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.8.5 /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV ENABLE_WEB_INTERFACE=true

COPY pyproject.toml uv.lock README.md openenv.yaml ./
COPY server ./server
COPY kitchen_ops_env ./kitchen_ops_env
COPY inference.py ./inference.py

RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "--no-dev", "python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
