import os
import shutil
import subprocess
import sys
from pathlib import Path

from git import Repo
from testcontainers.compose import DockerCompose
from testcontainers.core.container import DockerContainer
from testcontainers.minio import MinioContainer
from testcontainers.postgres import PostgresContainer
from testcontainers.rabbitmq import RabbitMqContainer

SCHEDULER_SCRIPT_PATH = "cogstack_model_gateway/scheduler/main.py"

COGSTACK_MODEL_SERVE_REPO = "https://github.com/CogStack/CogStack-ModelServe.git"
COGSTACK_MODEL_SERVE_COMMIT = "ac0a8c15e0596846c0d193cc71fd5347a2fc9631"
COGSTACK_MODEL_SERVE_LOCAL_PATH = Path("downloads/CogStack-ModelServe")
COGSTACK_MODEL_SERVE_COMPOSE = "docker-compose.yml"
COGSTACK_MODEL_SERVE_COMPOSE_MLFLOW = "docker-compose-mlflow.yml"


def start_testcontainers(containers: list[DockerContainer]):
    print("Starting test containers")
    for testcontainer in containers:
        testcontainer.start()


def remove_testcontainers(containers: list[DockerContainer]):
    print("Removing test containers")
    for testcontainer in containers:
        testcontainer.stop()


def configure_environment(
    postgres: PostgresContainer, rabbitmq: RabbitMqContainer, minio: MinioContainer
):
    print("Setting environment variables")
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
        "MLFLOW_TRACKING_URI": "http://mlflow-ui:5000",
    }

    os.environ.update(env)


def start_scheduler():
    print("Starting scheduler")
    return subprocess.Popen(
        ["poetry", "run", "python3", SCHEDULER_SCRIPT_PATH],
        stdout=sys.stdout,
        stderr=sys.stderr,
        start_new_session=True,
    )


def stop_scheduler(scheduler_process: subprocess.Popen):
    print("Stopping scheduler")
    scheduler_process.kill()


def clone_cogstack_model_serve():
    print("Cloning CogStack Model Serve")
    if not os.path.exists(COGSTACK_MODEL_SERVE_LOCAL_PATH):
        repo = Repo.clone_from(COGSTACK_MODEL_SERVE_REPO, COGSTACK_MODEL_SERVE_LOCAL_PATH)
    else:
        repo = Repo(COGSTACK_MODEL_SERVE_LOCAL_PATH)
        repo.remotes.origin.fetch()

    repo.git.checkout(COGSTACK_MODEL_SERVE_COMMIT)


def remove_cogstack_model_serve():
    print("Removing CogStack Model Serve")
    shutil.rmtree(COGSTACK_MODEL_SERVE_LOCAL_PATH)


def start_cogstack_model_serve() -> list[DockerCompose]:
    print("Deploying CogStack Model Serve")
    env_file_path = COGSTACK_MODEL_SERVE_LOCAL_PATH / ".env"
    with open(env_file_path, "w") as env_file:
        env_file.write(
            "MODEL_PACKAGE_FULL_PATH=/home/phoevos/cogstack/models/medmen_wstatus_2021_oct.zip\n"
        )
        env_file.write(f"CMS_UID={os.getuid()}\n")
        env_file.write(f"CMS_GID={os.getgid()}\n")
        env_file.write("MLFLOW_DB_USERNAME=admin\n")
        env_file.write("MLFLOW_DB_PASSWORD=admin\n")
        env_file.write("AWS_ACCESS_KEY_ID=admin\n")
        env_file.write("AWS_SECRET_ACCESS_KEY=admin123\n")

    compose: DockerCompose = None
    compose_mlflow: DockerCompose = None

    try:
        compose = DockerCompose(
            context=COGSTACK_MODEL_SERVE_LOCAL_PATH,
            compose_file_name=COGSTACK_MODEL_SERVE_COMPOSE,
            env_file=".env",
            services=["medcat-umls"],
        )
        compose.start()

        compose_mlflow = DockerCompose(
            context=COGSTACK_MODEL_SERVE_LOCAL_PATH,
            compose_file_name=COGSTACK_MODEL_SERVE_COMPOSE_MLFLOW,
            env_file=".env",
        )
        compose_mlflow.start()

        return [compose, compose_mlflow]
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        stop_cogstack_model_serve([env for env in (compose, compose_mlflow) if env])
        raise


def stop_cogstack_model_serve(compose_envs: list[DockerCompose]):
    print("Stopping CogStack Model Serve")
    for compose_env in compose_envs:
        compose_env.stop()
