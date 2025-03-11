# Contributing

Thank you for your interest in contributing to the project! Here are some useful instructions for
getting you started.

## Installation

1. Clone the repository:

    ```shell
    git clone https://github.com/CogStack/cogstack-model-gateway.git
    cd cogstack-model-gateway
    ```

2. Install Poetry if you haven't already:

    ```shell
    curl -sSL https://install.python-poetry.org | python3 -
    ```

3. Install the project dependencies:

    ```shell
    poetry install --with dev
    ```

4. Activate the virtual environment:

    ```shell
    eval $(poetry env activate)
    ```

## Pre-commit Hooks

We use pre-commit hooks to ensure code quality. To set them up, follow these steps:

1. Install the pre-commit hooks to your local Git repository:

    ```shell
    pre-commit install
    ```

2. (optional) To run the pre-commit hooks manually on all files, use:

    ```shell
    pre-commit run --all-files
    ```

## Running

To run the project locally, the [run_local.py](./run_local.py) script is provided as a shortcut for
spinning up the required external services (e.g. PostgreSQL, RabbitMQ, MinIO) as Docker containers
and starting the Gateway, Scheduler, and Ripper as separate processes on the host machine. It also
watches for changes in the codebase and restarts the services when necessary. Before running the
script, ensure you have configured the environment with the required variables, as described in the
[Installation](./README.md#installation) section of the README, and then execute the following
inside the virtual environment:

```shell
python run_local.py
```

Note that the script is meant as a convenience tool for local development, therefore coming with no
quality guarantees, and should not be used in production.

## Testing

To run the tests, execute the following command inside the virtual environment:

```shell
pytest
```

### On Mac

If you're running the integration tests on Mac you may need to take some additional steps to ensure
that the host machine can access the Docker containers. This is because Docker for Mac runs in a
virtual machine and, as a result, the containers are not directly accessible from the host. Given
that the services running as part of the integration tests use IP addresses to communicate with the
containers, an easy solution to avoid manual configuration would be using a tool like
[docker-mac-net-connect](https://github.com/chipmk/docker-mac-net-connect).
