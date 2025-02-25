import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import mlflow
import pytest
import requests
from docker.models.containers import Container
from fastapi.testclient import TestClient
from git import Repo
from testcontainers.compose import DockerCompose
from testcontainers.core.container import DockerClient, DockerContainer
from testcontainers.minio import MinioContainer
from testcontainers.postgres import PostgresContainer
from testcontainers.rabbitmq import RabbitMqContainer

from cogstack_model_gateway.common.object_store import ObjectStoreManager
from cogstack_model_gateway.common.queue import QueueManager
from cogstack_model_gateway.common.tasks import Status, Task, TaskManager

POSTGRES_IMAGE = "postgres:17.2"
RABBITMQ_IMAGE = "rabbitmq:4.0.4-management-alpine"
MINIO_IMAGE = "minio/minio:RELEASE.2024-11-07T00-52-20Z"

SCHEDULER_SCRIPT_PATH = "cogstack_model_gateway/scheduler/main.py"

TEST_ASSETS = Path("tests/integration/assets")
TEST_CMS_ENV_FILE = TEST_ASSETS / "cms.env"
TEST_CMS_MODEL_PACK = TEST_ASSETS / "simple_model4test-3.9-1.12.0_edeb88f7986cb05c.zip"

COGSTACK_MODEL_SERVE_REPO = "https://github.com/CogStack/CogStack-ModelServe.git"
COGSTACK_MODEL_SERVE_COMMIT = "a55be7b10a83e3bdbdbd1a9e13248e1557fdb0db"
COGSTACK_MODEL_SERVE_LOCAL_PATH = Path("downloads/CogStack-ModelServe")
COGSTACK_MODEL_SERVE_COMPOSE = "docker-compose.yml"
COGSTACK_MODEL_SERVE_COMPOSE_PROJECT_NAME = "cmg-test"
COGSTACK_MODEL_SERVE_COMPOSE_MLFLOW = "docker-compose-mlflow.yml"
COGSTACK_MODEL_SERVE_NETWORK = "cogstack-model-serve_cms"

TEST_MODEL_SERVICE = "medcat-umls"

ANNOTATION_FIELDS_BASE = [
    "start",
    "end",
    "label_name",
    "label_id",
    "categories",
    "accuracy",
    "meta_anns",
]
ANNOTATION_FIELDS_JSONL = [*ANNOTATION_FIELDS_BASE, "doc_name"]
ANNOTATION_FIELDS_JSON = [*ANNOTATION_FIELDS_BASE, "athena_ids"]

log = logging.getLogger("cmg.tests.integration")


def setup_testcontainers(request: pytest.FixtureRequest):
    postgres = PostgresContainer(POSTGRES_IMAGE)
    rabbitmq = RabbitMqContainer(RABBITMQ_IMAGE)
    minio = MinioContainer(MINIO_IMAGE)

    containers = [postgres, rabbitmq, minio]
    request.addfinalizer(lambda: remove_testcontainers(containers))

    start_testcontainers(containers)

    return postgres, rabbitmq, minio


def setup_scheduler(request: pytest.FixtureRequest):
    scheduler_process = None
    try:
        scheduler_process = start_scheduler()
    except Exception as e:
        pytest.fail(f"Failed to start scheduler: {e}")
    finally:
        request.addfinalizer(
            lambda: stop_scheduler(scheduler_process) if scheduler_process else None
        )


def setup_cms(request: pytest.FixtureRequest, cleanup_cms: bool) -> dict[str, dict]:
    try:
        clone_cogstack_model_serve()
    except Exception as e:
        pytest.fail(f"Failed to clone CogStack Model Serve: {e}")

    try:
        compose_envs = start_cogstack_model_serve([TEST_MODEL_SERVICE])
    except Exception as e:
        pytest.fail(f"Failed to start CogStack Model Serve: {e}")

    if cleanup_cms:
        request.addfinalizer(lambda: stop_cogstack_model_serve(compose_envs))
    else:
        request.addfinalizer(
            lambda: log.warning("Skipping cleanup of CogStack Model Serve resources")
        )

    return get_service_address_mapping(compose_envs)


def start_testcontainers(containers: list[DockerContainer]):
    log.info("Starting test containers...")
    for testcontainer in containers:
        log.debug(f"Starting testcontainer with image '{testcontainer.image}'...")
        testcontainer.start()


def remove_testcontainers(containers: list[DockerContainer]):
    log.info("Removing test containers...")
    for testcontainer in containers:
        log.debug(f"Removing testcontainer with image '{testcontainer.image}'...")
        testcontainer.stop()


def configure_environment(
    postgres: PostgresContainer,
    rabbitmq: RabbitMqContainer,
    minio: MinioContainer,
    extras: dict = None,
):
    log.info("Setting environment variables...")
    queue_connection_params = rabbitmq.get_connection_params()
    minio_host, minio_port = minio.get_config()["endpoint"].split(":")
    env = {
        "CMG_DB_USER": postgres.username,
        "CMG_DB_PASSWORD": postgres.password,
        "CMG_DB_HOST": postgres.get_container_host_ip(),
        "CMG_DB_PORT": postgres.get_exposed_port(postgres.port),
        "CMG_DB_NAME": "test",
        "CMG_QUEUE_USER": rabbitmq.username,
        "CMG_QUEUE_PASSWORD": rabbitmq.password,
        "CMG_QUEUE_HOST": queue_connection_params.host,
        "CMG_QUEUE_PORT": str(queue_connection_params.port),
        "CMG_QUEUE_NAME": "test",
        "CMG_OBJECT_STORE_HOST": minio_host,
        "CMG_OBJECT_STORE_PORT": minio_port,
        "CMG_OBJECT_STORE_ACCESS_KEY": minio.access_key,
        "CMG_OBJECT_STORE_SECRET_KEY": minio.secret_key,
        "CMG_OBJECT_STORE_BUCKET_TASKS": "test-tasks",
        "CMG_OBJECT_STORE_BUCKET_RESULTS": "test-results",
        "CMG_SCHEDULER_MAX_CONCURRENT_TASKS": "1",
        "CMS_PROJECT_NAME": COGSTACK_MODEL_SERVE_COMPOSE_PROJECT_NAME,
        **(extras or {}),
    }
    log.debug(env)
    os.environ.update(env)


def start_scheduler():
    log.info("Starting scheduler...")
    return subprocess.Popen(
        ["poetry", "run", "python3", SCHEDULER_SCRIPT_PATH],
        start_new_session=True,
        env=dict(os.environ),
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )


def stop_scheduler(scheduler_process: subprocess.Popen):
    log.info("Stopping scheduler...")
    scheduler_process.kill()


def clone_cogstack_model_serve():
    log.info("Cloning CogStack Model Serve...")
    try:
        if not os.path.exists(COGSTACK_MODEL_SERVE_LOCAL_PATH):
            log.debug("Repository does not exist locally, cloning...")
            repo = Repo.clone_from(COGSTACK_MODEL_SERVE_REPO, COGSTACK_MODEL_SERVE_LOCAL_PATH)
        elif not os.path.exists(COGSTACK_MODEL_SERVE_LOCAL_PATH / ".git"):
            log.debug("Dir exists locally but is not a git repository, removing and cloning...")
            remove_cogstack_model_serve()
            repo = Repo.clone_from(COGSTACK_MODEL_SERVE_REPO, COGSTACK_MODEL_SERVE_LOCAL_PATH)
        else:
            log.debug("Repository exists locally, fetching the latest changes...")
            repo = Repo(COGSTACK_MODEL_SERVE_LOCAL_PATH)
            repo.remotes.origin.fetch()

        repo.git.checkout(COGSTACK_MODEL_SERVE_COMMIT)
    except Exception:
        remove_cogstack_model_serve(ignore_errors=True)
        raise


def remove_cogstack_model_serve(ignore_errors: bool = False):
    log.info("Removing CogStack Model Serve...")
    shutil.rmtree(COGSTACK_MODEL_SERVE_LOCAL_PATH, ignore_errors=ignore_errors)


def start_cogstack_model_serve(model_services: list[str]) -> list[DockerCompose]:
    log.info("Deploying CogStack Model Serve (this might take a few minutes)...")
    with open(TEST_CMS_ENV_FILE) as f:
        const_envvars = f.read()

    env_file_path = COGSTACK_MODEL_SERVE_LOCAL_PATH / ".env"
    with open(env_file_path, "w") as env_file:
        env_file.write(const_envvars)
        env_file.write(f"CMS_UID={os.getuid()}\n")
        env_file.write(f"CMS_GID={os.getgid()}\n")
        env_file.write(f"COMPOSE_PROJECT_NAME={COGSTACK_MODEL_SERVE_COMPOSE_PROJECT_NAME}\n")
        env_file.write(f"MODEL_PACKAGE_FULL_PATH={TEST_CMS_MODEL_PACK.absolute()}\n")

    log.debug(f"CogStack Model Serve environment file: {env_file_path}")

    compose: DockerCompose = None
    compose_mlflow: DockerCompose = None

    try:
        compose = DockerCompose(
            context=COGSTACK_MODEL_SERVE_LOCAL_PATH,
            compose_file_name=COGSTACK_MODEL_SERVE_COMPOSE,
            env_file=".env",
            services=model_services,
        )
        compose.start()

        compose_mlflow = DockerCompose(
            context=COGSTACK_MODEL_SERVE_LOCAL_PATH,
            compose_file_name=COGSTACK_MODEL_SERVE_COMPOSE_MLFLOW,
            env_file=".env",
            services=["mlflow-ui", "mlflow-db", "minio", "model-bucket-init"],
        )
        compose_mlflow.start()
        return [compose, compose_mlflow]
    except subprocess.CalledProcessError as e:
        log.info(e.stderr)
        stop_cogstack_model_serve([env for env in (compose, compose_mlflow) if env])
        raise


def _get_container_address(
    container: Container, network: str = COGSTACK_MODEL_SERVE_NETWORK
) -> str:
    return container["NetworkSettings"]["Networks"][network]["IPAddress"]


def _get_container_ports(container: Container) -> set[int]:
    return {p["PrivatePort"] for p in container["Ports"]}


def get_service_address_mapping(compose_envs: list[DockerCompose]) -> dict[str, dict]:
    docker_client = DockerClient()
    return {
        c.Service: {
            "address": _get_container_address(container),
            "port": _get_container_ports(container).pop(),
        }
        for compose_env in compose_envs
        for c in compose_env.get_containers()
        if (container := docker_client.get_container(c.ID))
    }


def stop_cogstack_model_serve(compose_envs: list[DockerCompose]):
    log.info("Stopping CogStack Model Serve")
    for compose_env in compose_envs:
        log.debug(f"Stopping {compose_env}...")
        compose_env.stop()
    remove_cogstack_model_serve()


def validate_api_response(
    response: requests.Response, expected_status_code: int, return_json: bool = False
):
    assert response.status_code == expected_status_code
    response_json = response.json()
    assert all(key in response_json for key in ["uuid", "status"])
    return response_json if return_json else None


def verify_task_submitted_successfully(task_uuid: str, tm: TaskManager):
    assert tm.get_task(task_uuid), "Failed to submit task: not found in the database"


def wait_for_task_completion(task_uuid: str, tm: TaskManager, expected_status: Status) -> Task:
    """Wait for a task to complete and verify its results."""
    while (task := tm.get_task(task_uuid)).status != expected_status:
        if task.status in [Status.FAILED, Status.SUCCEEDED, Status.REQUEUED]:
            pytest.fail(f"Task '{task_uuid}' completed with unexpected status '{task.status}'")

    # Verify task results
    if expected_status == Status.SUCCEEDED:
        assert task.error_message is None, f"Task failed unexpectedly: {task.error_message}"
        assert task.result is not None, "Task results are missing"
    elif expected_status == Status.FAILED:
        assert task.error_message is not None, "Task failed without an error message"

    return task


def verify_task_payload_in_object_store(
    task_payload_key: str, expected_payload: bytes, task_object_store_manager: ObjectStoreManager
):
    """Verify that the task payload was stored in the object store."""
    assert task_object_store_manager.get_object(task_payload_key) == expected_payload


def verify_queue_is_empty(queue_manager: QueueManager):
    """Verify that the queue is empty after the task is processed."""
    assert queue_manager.is_queue_empty()


def download_result_object(
    key: str,
    results_object_store_manager: ObjectStoreManager,
    format: str = "json",
) -> tuple:
    result = results_object_store_manager.get_object(key)
    try:
        if format == "json":
            result_json = json.loads(result.decode("utf-8"))
        elif format == "jsonl":
            result_json = [json.loads(line) for line in result.decode("utf-8").split("\n") if line]
        elif format == "text":
            result_json = result.decode("utf-8")
        else:
            pytest.fail(f"Unsupported format: {format}")
    except json.JSONDecodeError as e:
        pytest.fail(f"Failed to parse the result as JSON: {result}, {e}")

    return result, result_json


def verify_annotation_contains_keys(annotation: dict, expected_keys: list[str]):
    """Verify that the annotation contains the expected keys."""
    assert all(key in annotation for key in expected_keys)


def verify_results_match_api_info(client: TestClient, task: Task, result: bytes):
    """Verify results match the information exposed through the user-facing API."""
    response = client.get(f"/tasks/{task.uuid}", params={"detail": True})
    assert response.status_code == 200

    response_json = response.json()
    assert response_json["uuid"] == task.uuid
    assert response_json["status"] == task.status
    assert response_json["error_message"] == task.error_message
    assert response_json["tracking_id"] == task.tracking_id

    # Download results and verify they match the provided ones
    download_results = client.get(f"/tasks/{task.uuid}", params={"detail": True, "download": True})
    assert download_results.status_code == 200
    assert download_results.content == result


def parse_mlflow_url(url: str) -> tuple:
    response = requests.get(url)
    assert response.status_code == 200

    try:
        parsed_url = urlparse(url)
    except Exception as e:
        pytest.fail(f"Failed to parse URL: {url}, {e}")

    tracking_uri = f"{parsed_url.scheme}://{parsed_url.netloc}"
    assert tracking_uri == mlflow.get_tracking_uri()

    path_parts = parsed_url.fragment.split("/")
    assert len(path_parts) >= 4 and path_parts[1] == "experiments" and path_parts[3] == "runs"

    experiment_id, run_id = path_parts[2], path_parts[4]

    try:
        _ = mlflow.get_experiment(experiment_id)
    except Exception as e:
        pytest.fail(f"Failed to get experiment '{experiment_id}': {e}")

    try:
        run = mlflow.get_run(run_id)
    except Exception as e:
        pytest.fail(f"Failed to get run '{run_id}': {e}")

    assert run.info.experiment_id == experiment_id

    return tracking_uri, experiment_id, run_id
