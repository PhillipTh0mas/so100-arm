
FROM debian:bookworm as base-image

ARG VIRTUAL_ENV=/root/venv

RUN apt-get update \
    && apt-get install --no-install-suggests --no-install-recommends -y \
    build-essential \
    python3-pip \
    python3 \
    libpython3-dev \
    python3-venv \
    git \
    && python3 -m venv ${VIRTUAL_ENV} \
    && ${VIRTUAL_ENV}/bin/pip install --upgrade pip setuptools wheel \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN ${VIRTUAL_ENV}/bin/python3 -m pip install -U pip

RUN set -eux; \
    if [ -f ./pip.conf ]; then \
    echo "Found pip.conf, pointing PIP_CONFIG_FILE at it"; \
    export PIP_CONFIG_FILE="$(pwd)/pip.conf"; \
    else \
    echo "No pip.conf found, using default indexes"; \
    fi; \
    ${VIRTUAL_ENV}/bin/python3 -m pip install ".[mcp_control]" \
    && ${VIRTUAL_ENV}/bin/python3 -m pip uninstall torchvision torchaudio torchcodec diffusers rerun-sdk -y  \
    && ${VIRTUAL_ENV}/bin/python3 -m pip install "rerun-sdk==0.24.1"

FROM python:3.12-slim-bookworm

ARG VIRTUAL_ENV=/root/venv
COPY --from=base-image ${VIRTUAL_ENV} ${VIRTUAL_ENV}
ENV TELEOP=1

ENTRYPOINT ["/root/venv/bin/python3", "-m", "app.main"]
