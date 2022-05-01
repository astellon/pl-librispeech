ARG CUDA_VERSION=11.6.0
ARG CUDNN_VERSION=8

FROM debian:bullseye-slim as setup

ENV DEBIAN_FRONTEND=noninteractive
ENV PYENV_ROOT=/opt/pyenv
ENV POETRY_HOME=/opt/poetry
ENV PATH=${POETRY_HOME}/bin:${PYENV_ROOT}/bin:${PYENV_ROOT}/shims:${PATH}

RUN apt-get update && \
    apt-get install -y  git make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

RUN curl https://pyenv.run | bash

ARG PYTHON_VERSION=3.10.4

RUN pyenv install ${PYTHON_VERSION}
RUN pyenv global ${PYTHON_VERSION}

ENV POETRY_HOME=/opt/poetry

RUN curl -sSL https://install.python-poetry.org | python

COPY pyproject.toml pyproject.toml

RUN poetry config virtualenvs.create false && poetry install --no-dev

FROM debian:bullseye-slim as runner

RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

ENV PYENV_ROOT=/opt/pyenv
ENV POETRY_HOME=/opt/poetry
ENV PATH=${POETRY_HOME}/bin:${PYENV_ROOT}/bin:${PYENV_ROOT}/shims:${PATH}

COPY --from=setup /opt /opt

COPY entrypoint.sh entrypoint.sh

ENTRYPOINT [ "entrypoint.sh" ]