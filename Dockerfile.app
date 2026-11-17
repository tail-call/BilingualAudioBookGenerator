FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /uvx /bin/

WORKDIR /app

ENV UV_PYTHON_PREFERENCE=only-system
ENV UV_PYTHON_DOWNLOADS=never
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1
ENV PATH="/app/.venv/bin:$PATH"

# So far only dependencies
COPY pyproject.toml uv.lock ./

# Populate the venv so it's cached nicely
# --mount=type=cache persists uv's download cache across builds
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Now copy the app
COPY bookaudio/ bookaudio/
COPY main.py main.py
COPY run-app.sh run-app.sh

RUN chmod +x run-app.sh
CMD ["./run-app.sh"]

