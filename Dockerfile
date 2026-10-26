FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY . .

ARG INSTALL_EXTRAS=geo,ml
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir torch torchvision --index-url "${TORCH_INDEX_URL}" && \
    python -m pip install --no-cache-dir ".[${INSTALL_EXTRAS}]"

LABEL org.opencontainers.image.source="https://github.com/veerendrakosuri/deforestation-detection" \
      org.opencontainers.image.description="Sentinel-2 forest-change screening CLI" \
      org.opencontainers.image.licenses="MIT"

ENTRYPOINT ["forest-change"]
CMD ["--help"]
