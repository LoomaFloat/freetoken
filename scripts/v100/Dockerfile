ARG BASE=nvidia/cuda:12.6.3-devel-ubuntu22.04
FROM ghcr.io/astral-sh/uv:0.8.22 AS uv
FROM ${BASE}
COPY --from=uv /uv /uvx /usr/local/bin/
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates git ninja-build binutils rsync \
 && rm -rf /var/lib/apt/lists/*
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_CACHE_DIR=/cache/uv \
    UV_LINK_MODE=copy \
    CUDA_HOME=/usr/local/cuda
RUN uv python install 3.10 3.11 3.12 3.13
WORKDIR /work
