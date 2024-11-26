FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y python3

WORKDIR /app

COPY ../../pyproject.toml ../../poetry.lock /app/
RUN poetry install

COPY ../common/ /app/common/

COPY . /app/gateway/

EXPOSE 8000

CMD ["fastapi", "run", "--workers", "4", "app/gateway/main.py"]
