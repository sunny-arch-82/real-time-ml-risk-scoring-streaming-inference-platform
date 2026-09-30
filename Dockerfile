FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra streaming
COPY configs ./configs
COPY data/sample ./data/sample
ENV RISKFLOW_HOME=/app/artifacts RISKFLOW_REGISTRY=/app/models
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "riskflow.api:app", "--host", "0.0.0.0", "--port", "8000"]
