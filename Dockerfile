# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.8.3

# System deps (build-essential for some libs like xgboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirement spec first (if you have requirements.txt later replace this block)
# We will generate it dynamically using pip freeze inside venv creation; for now install base tools

# Install uv for fast dependency installs
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy project
COPY . .

RUN uv venv --python 3.11 && \
    . .venv/bin/activate && \
    uv pip install --system --upgrade pip && \
    uv pip install django djangorestframework scikit-learn xgboost optuna loguru pandas numpy plotly tiktoken tqdm joblib

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Expose port
EXPOSE 8000


# Collect static (no-op currently) & run migrations (even if stateless) then start server
CMD ["bash", "-lc", "python manage.py migrate --noinput || true; python manage.py runserver 0.0.0.0:8000"]
