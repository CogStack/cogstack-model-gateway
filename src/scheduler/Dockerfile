FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV POETRY_HOME=/etc/poetry
ENV PATH="${POETRY_HOME}/bin:${PATH}"

RUN apt-get update && apt-get install -y curl python3
RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry install

COPY src/common/ common/
COPY src/scheduler/ scheduler/
COPY config.json scheduler/

CMD ["poetry", "run", "python3", "scheduler/main.py"]
