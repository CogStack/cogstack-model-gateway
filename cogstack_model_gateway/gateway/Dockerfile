FROM ubuntu:24.04

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV HTTP_PROXY=$HTTP_PROXY
ENV HTTPS_PROXY=$HTTPS_PROXY
ENV NO_PROXY=$NO_PROXY
ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTPS_PROXY
ENV no_proxy=$NO_PROXY
ENV DEBIAN_FRONTEND=noninteractive
ENV POETRY_HOME=/etc/poetry
ENV PATH="${POETRY_HOME}/bin:${PATH}"

RUN apt-get update && apt-get install -y curl python3
RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root --no-directory

COPY cogstack_model_gateway/common ./cogstack_model_gateway/common
COPY cogstack_model_gateway/gateway ./cogstack_model_gateway/gateway
COPY config.json README.md ./

RUN poetry install --only main

EXPOSE 8000

CMD ["poetry", "run", "fastapi", "run", "--workers", "4", "cogstack_model_gateway/gateway/main.py"]
