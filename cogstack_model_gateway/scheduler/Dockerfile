FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV POETRY_HOME=/etc/poetry
ENV PATH="${POETRY_HOME}/bin:${PATH}"

RUN apt-get update && apt-get install -y curl python3
RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root --no-directory

COPY cogstack_model_gateway/common ./cogstack_model_gateway/common
COPY cogstack_model_gateway/scheduler ./cogstack_model_gateway/scheduler
COPY config.json README.md ./

RUN poetry install --only main

CMD ["poetry", "run", "python3", "cogstack_model_gateway/scheduler/main.py"]
