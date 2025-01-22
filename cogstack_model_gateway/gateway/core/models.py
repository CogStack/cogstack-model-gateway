import os
from functools import lru_cache

import docker
import mlflow
from mlflow.entities.model_registry import RegisteredModel

IS_MODEL_LABEL = "org.cogstack.model-serve"
MODEL_URI_LABEL = "org.cogstack.model-serve.uri"
PROJECT_NAME_LABEL = "com.docker.compose.project"
SERVICE_NAME_LABEL = "com.docker.compose.service"

CMS_PROJECT_ENV_VAR = "CMS_PROJECT_NAME"


@lru_cache
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
