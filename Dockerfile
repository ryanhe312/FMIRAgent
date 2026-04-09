FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir uv

# Install Python dependencies first for better layer caching.
COPY . /app
RUN uv sync

ENV PYTHONUNBUFFERED=1
EXPOSE 8989