# Contributing

Thank you for your interest in contributing to the project! Here are some useful instructions for
getting you started.

## Installation

1. Clone the repository:

    ```shell
    git clone https://github.com/CogStack/CogStack-ModelGateway.git cms-gateway
    cd cms-gateway
    ```

2. Install Poetry if you haven't already:

    ```shell
    curl -sSL https://install.python-poetry.org | python3 -
    ```

3. Install the project dependencies:

    ```shell
    poetry install --with dev
    ```

## Pre-commit Hooks

We use pre-commit hooks to ensure code quality. To set them up, follow these steps:

1. Install pre-commit if you haven't already:

    ```shell
    pip install pre-commit
    ```

2. Install the pre-commit hooks to your local Git repository:

    ```shell
    pre-commit install
    ```

3. (optional) To run the pre-commit hooks manually on all files, use:

    ```shell
    pre-commit run --all-files
    ```
