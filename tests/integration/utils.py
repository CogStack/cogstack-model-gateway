import logging
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from docker.models.containers import Container
from git import Repo
from testcontainers.compose import DockerCompose
from testcontainers.core.container import DockerClient, DockerContainer
from testcontainers.minio import MinioContainer
from testcontainers.postgres import PostgresContainer
from testcontainers.rabbitmq import RabbitMqContainer

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
COGSTACK_MODEL_SERVE_COMPOSE_MLFLOW = "docker-compose-mlflow.yml"
COGSTACK_MODEL_SERVE_NETWORK = "cogstack-model-serve_cms"

TEST_MODEL_SERVICE = "medcat-umls"

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
