# CogStack Model Gateway

The CogStack Model Gateway (CMG) is a service that provides a unified interface for accessing
machine learning models deployed as standalone servers. It implements service discovery and enables
scheduling incoming tasks based on their priority, as well the state of the cluster. The project is
designed to work with [Cogstack ModelServe (CMS)](https://github.com/CogStack/CogStack-ModelServe)
model server instances and consists of two main components:

* **Model Gateway**: A RESTful API that provides a unified interface for accessing machine learning
  models deployed as standalone servers. The gateway is responsible for assigning a priority to each
  incoming task and publishing it to a queue for processing. On top of the API endpoints provided by
  CMS, the gateway also exposes endpoints for monitoring the state of submitted tasks and fetching
  their results, as well as for discovering available model servers and deploying new ones from
  previously trained models.
* **Task Scheduler**: A service that schedules queued tasks for execution based on their priority.
  The scheduler is responsible for ensuring that tasks are processed in a timely manner and that the
  cluster is not overloaded.

## Content

* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Usage](#usage)
* [Development](#development)

## Prerequisites

In order to run the CogStack Model Gateway, you need:

* [Docker](https://www.docker.com/) installed on the host
* An instance of the [CogStack ModelServe](https://github.com/CogStack/CogStack-ModelServe) stack,
  including a configured model tracking server (e.g. MLflow). The Gateway uses the external CMS
  network for model discovery and to communicate with the model servers. You should make a note of
  the CMS Docker project name as well as the tracking server URL, which are required for setting up
  the Gateway.

## Installation

Installing the CogStack Model Gateway is possible using Docker Compose, while configuration is done
through environment variables. Before deploying the Gateway, make sure to set the required variables
either by exporting them in the shell or by creating a `.env` file in the root directory of the
project. The following variables are required:

* `MLFLOW_TRACKING_URI`: The URI for the MLflow tracking server.
* `CMS_PROJECT_NAME`: The name of the Docker project where the CogStack ModelServe stack is running.
* `CMS_HOST_URL` (optional): Useful when running CogStack ModelServe instances behind a proxy. If
  omitted, the Gateway will attempt to reach the services directly over the internal Docker network.
* `CMG_SCHEDULER_MAX_CONCURRENT_TASKS`: The max number of concurrent tasks the scheduler can handle.
* `CMG_DB_USER`: The username for the PostgreSQL database.
* `CMG_DB_PASSWORD`: The password for the PostgreSQL database.
* `CMG_DB_NAME`: The name of the PostgreSQL database.
* `CMG_QUEUE_USER`: The username for the RabbitMQ message broker.
* `CMG_QUEUE_PASSWORD`: The password for the RabbitMQ message broker.
* `CMG_QUEUE_NAME`: The name of the RabbitMQ queue.
* `CMG_OBJECT_STORE_ACCESS_KEY`: The access key for the MinIO object store.
* `CMG_OBJECT_STORE_SECRET_KEY`: The secret key for the MinIO object store.
* `CMG_OBJECT_STORE_BUCKET_TASKS`: The name of the MinIO bucket for storing task payloads.
* `CMG_OBJECT_STORE_BUCKET_RESULTS`: The name of the MinIO bucket for storing task results.

An example configuration is provided below, using the default project name for the CMS stack (i.e.
"cms"), forcing the scheduler to handle only one task at a time, using the internal Docker service
name in the MLflow URI, and setting up the remaining services with sample credentials that fulfill
their respective service validation requirements (e.g. MinIO secret key minimum length, underscores
not allowed in MinIO bucket names). The configuration should be saved in a `.env` file in the root
directory of the project before running Docker Compose (or sourced directly in the shell):

```shell
CMS_PROJECT_NAME=<cms-docker-compose-project-name>  # e.g. cms

# (optional) Useful when running CMS behind a proxy
CMS_HOST_URL=https://<proxy-docker-service-name>/cms  # e.g. https://proxy/cms

CMG_SCHEDULER_MAX_CONCURRENT_TASKS=1

# Postgres
CMG_DB_USER=admin
CMG_DB_PASSWORD=admin
CMG_DB_HOST=postgres
CMG_DB_PORT=5432
CMG_DB_NAME=cmg_tasks

# RabbitMQ
CMG_QUEUE_USER=admin
CMG_QUEUE_PASSWORD=admin
CMG_QUEUE_HOST=rabbitmq
CMG_QUEUE_PORT=5672
CMG_QUEUE_NAME=cmg_tasks

# MinIO
CMG_OBJECT_STORE_ACCESS_KEY=admin
CMG_OBJECT_STORE_SECRET_KEY=admin123
CMG_OBJECT_STORE_HOST=minio
CMG_OBJECT_STORE_PORT=9000
CMG_OBJECT_STORE_BUCKET_TASKS=cmg-tasks
CMG_OBJECT_STORE_BUCKET_RESULTS=cmg-results

# MLflow (use container IP when running locally)
MLFLOW_TRACKING_URI=http://<mlflow-docker-service-name>:<mlflow-port>  # e.g. http://mlflow-ui:5000
```

To install the CogStack Model Gateway, clone the repository and run `docker compose` inside the root
directory. It is recommended to set the `CMG_COMMIT_SHA` environment variable to the current commit
hash which will be added as a label to all containers for easier tracking in production:

```shell
CMG_COMMIT_SHA=$(git rev-parse HEAD) docker compose -f docker-compose.yaml up
```

This will spin up the following services:

* **Model Gateway**: The main service that provides a RESTful API for accessing machine learning
  models deployed as standalone CMS servers.
* **Task Scheduler**: A service that schedules queued tasks for execution based on their priority.
* **Ripper**: A service responsible for removing model servers deployed through the Gateway that
  have exceeded their TTL.
* **PostgreSQL**: A database used for storing task metadata (e.g. status, result references).
* **RabbitMQ**: A message broker used for task queuing and communication between the Gateway and the
  Scheduler.
* **MinIO**: An object storage service used for storing task results, as well as incoming request
  payloads.
* **pgAdmin**: A web-based interface for managing the PostgreSQL database.

## Usage

The Gateway exposes 2 main HTTP endpoints, one for interacting with the model servers and one for
monitoring the state of submitted tasks. The following endpoints are available:

* **Model Servers**: Interact with CMS model servers.

  * `GET /models`: List all available model servers (i.e. Docker containers with the
    "org.cogstack.model-serve" label and "com.docker.compose.project" set to `$CMS_PROJECT_NAME`).

    * **Query Parameters**:
      * `verbose (bool)`: Include model metadata from the tracking server (if available).

  * `GET /models/{model_server_name}/info`: Get information about a specific model (equivalent to
    the `/info` CMS endpoint).
  * `POST /models/{model_server_name}`: Deploy a new model server from a previously trained model.

    * **Body**:
      * `tracking_id (str)`: The tracking ID of the run that generated the model to serve (e.g.
        MLflow run ID), used to fetch the model URI (optional if model_uri is provided explicitly).
      * `model_uri (str)`: The URI of the model to serve (optional if tracking_id is provided).
      * `ttl (int, default=86400)`: The deployed model will be deleted after TTL seconds (defaults
        to 1 day). Set -1 as the TTL value to protect the model from being deleted.

  * `POST /models/{model_server_name}/tasks/{task_name}`: Execute a task on the specified model
    server, providing any query parameters or request body required (follows the CMS API, striving
    to support the same endpoints).

* **Tasks**: Monitor the state of submitted tasks.

  * `GET /tasks`: List all submitted tasks (currently not allowed, will be enabled once users are
    introduced).
  * `GET /tasks/{task_id}`: Get information about a specific task.

    * **Query Parameters**:
      * `detail (bool)`: Include detailed information about the task (e.g. result reference, error
        message, model tracking ID).
      * `download (bool)`: Download the result of the task (if available).

## Development

The project is still under active development. In the future we will be focusing on the following:

* **Tests**: Adding unit tests for every component of the project (only the `common` package is
  currently tested appropriately) and extending the integration tests to cover the training and
  evaluation CMS endpoints.
* **User management**: Introduce users and bind task requests to them, to control access to results
  and generate notifications.
* **Smart scheduling**: Implement a more sophisticated scheduling algorithm that takes into account
  the state of the cluster.
* **CI/CD**: Set up a continuous integration and deployment pipeline for the project.
* **Monitoring**: Integrate with Prometheus and Grafana.
