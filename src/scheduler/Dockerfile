FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y python3

WORKDIR /app

COPY ../../pyproject.toml ../../poetry.lock /app/
RUN poetry install

COPY ../common/ /app/common/
COPY . /app/scheduler/

CMD ["python3", "main.py"]
