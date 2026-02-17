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
COPY . .

RUN set -eux; \
    if [ -f ./pip.conf ]; then \
    echo "Found pip.conf, pointing PIP_CONFIG_FILE at it"; \
    export PIP_CONFIG_FILE="$(pwd)/pip.conf"; \
    else \
    echo "No pip.conf found, using default indexes"; \
    fi; \
    pip install .

# ---------- Runtime ----------
FROM python:3.12-slim-bookworm AS runtime

ARG VIRTUAL_ENV=/opt/venv
ENV VIRTUAL_ENV=${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

WORKDIR /app
COPY --from=build ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY --from=build /app /app

ENTRYPOINT ["python3", "-m", "app.main"]
