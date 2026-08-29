# RiskIQ backend — Phase 0 development image.
# Phase 10 adds a slim production target that excludes the dev/CI tooling.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# The PyTorch CPU index is added alongside PyPI rather than replacing it. On Linux the
# default PyPI torch wheel bundles the CUDA runtime — roughly 2GB the image never uses,
# because Tier-2 trains and scores on CPU for determinism (see requirements.txt).
RUN pip install --upgrade pip \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["sh", "-c", "bash scripts/download_models.sh && uvicorn app.main:app --host 0.0.0.0 --port $PORT"]