# ---------- Build ----------
FROM python:3.12-slim-bookworm AS build

ARG VIRTUAL_ENV=/opt/venv
ENV VIRTUAL_ENV=${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && python -m venv "${VIRTUAL_ENV}" \
    && pip install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Copy only files that affect dependency resolution (adjust to your repo)
COPY pyproject.toml ./
COPY app/__init__.py ./app/__init__.py

# 2) Install dependencies (this layer stays cached when only code changes)
RUN set -eux; \
    if [ -f ./pip.conf ]; then \
    export PIP_CONFIG_FILE="$(pwd)/pip.conf"; \
    fi; \
    pip install ".[mcp_control]"; \
    pip uninstall -y torchvision torchaudio torchcodec diffusers rerun-sdk || true; \
    pip install "rerun-sdk==0.24.1"

# 3) Now copy the rest of the source (this is what you change often)
COPY . .

# 4) Install *your* package without re-solving / re-downloading deps
# (Keeps import paths consistent for `python -m app.main`)
RUN pip install --no-deps -e .

# ---------- Runtime ----------
FROM python:3.12-slim-bookworm AS runtime

ARG VIRTUAL_ENV=/opt/venv
ENV VIRTUAL_ENV=${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV TELEOP=1

WORKDIR /app
COPY --from=build ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY --from=build /app /app
COPY config/ /config

ENTRYPOINT ["python3", "-m", "app.main"]
