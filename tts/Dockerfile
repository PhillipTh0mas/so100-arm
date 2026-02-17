# ---------- Build stage ----------
FROM debian:bookworm AS build

ARG VIRTUAL_ENV=/make87/venv
ENV VIRTUAL_ENV=${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    build-essential \
    && python3 -m venv "${VIRTUAL_ENV}" \
    && pip install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# dependency layer (cache-friendly)
COPY pyproject.toml ./
# If you have one, also copy: poetry.lock / uv.lock / requirements*.txt
RUN pip install "pyttsx3>=2.90"

# now install your package
COPY . .
RUN pip install .

# ---------- Runtime stage ----------
FROM debian:bookworm-slim AS runtime

ARG VIRTUAL_ENV=/make87/venv
ENV VIRTUAL_ENV=${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    espeak-ng libespeak-ng1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY --from=build /app /app

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import pyttsx3; pyttsx3.init(); print('ok')" || exit 1

ENTRYPOINT ["python3", "-m", "app.main"]
