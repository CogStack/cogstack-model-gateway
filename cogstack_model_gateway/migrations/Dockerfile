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
ENV PYTHONPATH=/app

COPY pyproject.toml poetry.lock ./
RUN poetry install --with migrations --no-root --no-directory

COPY cogstack_model_gateway/common/db.py ./cogstack_model_gateway/common/db.py
COPY cogstack_model_gateway/common/models.py ./cogstack_model_gateway/common/models.py
COPY cogstack_model_gateway/common/tasks.py ./cogstack_model_gateway/common/tasks.py
COPY cogstack_model_gateway/migrations ./cogstack_model_gateway/migrations

CMD ["poetry", "run", "alembic", "-c", "cogstack_model_gateway/migrations/alembic.ini", "upgrade", "head"]
