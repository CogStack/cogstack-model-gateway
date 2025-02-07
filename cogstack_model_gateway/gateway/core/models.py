import os

import docker
import mlflow
from docker.models.containers import Container
from mlflow.entities.model_registry import RegisteredModel

from cogstack_model_gateway.common.containers import (
    IS_MODEL_LABEL,
    MANAGED_BY_LABEL,
    MANAGED_BY_LABEL_VALUE,
    MODEL_URI_LABEL,
    PROJECT_NAME_LABEL,
    SERVICE_NAME_LABEL,
    TTL_LABEL,
)

CMS_PROJECT_ENV_VAR = "CMS_PROJECT_NAME"
CMS_DOCKER_NETWORK = "cogstack-model-serve_cms"


def get_running_models() -> list[dict]:
    client = docker.from_env()
    cms_project = os.getenv(CMS_PROJECT_ENV_VAR)
    if not cms_project:
        raise ValueError(f"Environment variable {CMS_PROJECT_ENV_VAR} is not set.")

    containers = client.containers.list(
        filters={
            "status": "running",
            "label": [IS_MODEL_LABEL, f"{PROJECT_NAME_LABEL}={cms_project}"],
        }
    )
    return [
        {"name": c.labels.get(SERVICE_NAME_LABEL, c.name), "uri": c.labels.get(MODEL_URI_LABEL)}
        for c in containers
    ]


def get_model_meta(model_uri: str) -> RegisteredModel:
    try:
        client = mlflow.tracking.MlflowClient()
        return client.get_registered_model(model_uri)
    except mlflow.exceptions.MlflowException:
        return None


def run_model_container(model_name: str, model_uri: str, ttl: int):
    client = docker.from_env()
    cms_project = os.getenv(CMS_PROJECT_ENV_VAR)
    if not cms_project:
        raise ValueError(f"Environment variable {CMS_PROJECT_ENV_VAR} is not set.")

    labels = {
        PROJECT_NAME_LABEL: cms_project,
        IS_MODEL_LABEL: model_name,
        MODEL_URI_LABEL: model_uri,
        TTL_LABEL: str(ttl),
        MANAGED_BY_LABEL: MANAGED_BY_LABEL_VALUE,
    }

    base_cmd = "python cli/cli.py serve"
    model_type_arg = "--model-type medcat_umls"
    model_name_arg = f"--model-name {model_name}"
    mlflow_uri_arg = f"--mlflow-model-uri {model_uri}"
    host_arg = "--host 0.0.0.0"
    port_arg = "--port 8000"

    container: Container = client.containers.run(
        "cogstacksystems/cogstack-modelserve:dev",
        command=[
            "sh",
            "-c",
            f"{base_cmd} {model_type_arg} {model_name_arg} {mlflow_uri_arg} {host_arg} {port_arg}",
        ],
        detach=True,
        environment={
            "ENABLE_TRAINING_APIS": "true",
            "ENABLE_EVALUATION_APIS": "true",
            "ENABLE_PREVIEWS_APIS": "true",
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "MLFLOW_S3_ENDPOINT_URL": os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000"),
            "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-ui:5000"),
            "MLFLOW_TRACKING_USERNAME": os.getenv("MLFLOW_TRACKING_USERNAME", "admin"),
            "MLFLOW_TRACKING_PASSWORD": os.getenv("MLFLOW_TRACKING_PASSWORD", "password"),
            "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": os.getenv(
                "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "true"
            ),
            "GELF_INPUT_URI": os.getenv("GELF_INPUT_URI", "http://graylog:12201"),
            "AUTH_USER_ENABLED": os.getenv("AUTH_USER_ENABLED", "false"),
            "AUTH_JWT_SECRET": os.getenv("AUTH_JWT_SECRET"),
            "AUTH_ACCESS_TOKEN_EXPIRE_SECONDS": os.getenv(
                "AUTH_ACCESS_TOKEN_EXPIRE_SECONDS", "3600"
            ),
            "AUTH_DATABASE_URL": os.getenv(
                "AUTH_DATABASE_URL", "sqlite+aiosqlite:///./cms-users.db"
            ),
            "HTTP_PROXY": os.getenv("HTTP_PROXY"),
            "HTTPS_PROXY": os.getenv("HTTPS_PROXY"),
            "NO_PROXY": os.getenv("NO_PROXY", "mlflow-ui,minio,graylog,auth-db,localhost"),
            "http_proxy": os.getenv("HTTP_PROXY"),
            "https_proxy": os.getenv("HTTPS_PROXY"),
            "no_proxy": os.getenv("NO_PROXY", "mlflow-ui,minio,graylog,auth-db,localhost"),
        },
        labels=labels,
        name=model_name,
        network=CMS_DOCKER_NETWORK,
        volumes={
            "retrained-models": {"bind": "/app/model/retrained", "mode": "rw"},
        },
        ports={"8000/tcp": None},
        healthcheck={
            "test": ["CMD", "curl", "-f", "http://localhost:8000/info"],
            "interval": 90 * 1000000 * 1000,
            "timeout": 10 * 1000000 * 1000,
            "retries": 3,
            "start_period": 60 * 1000000 * 1000,
        },
    )

    return container
