FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY . .

ARG INSTALL_EXTRAS=geo,ml
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir ".[${INSTALL_EXTRAS}]"

ENTRYPOINT ["forest-change"]
CMD ["--help"]
