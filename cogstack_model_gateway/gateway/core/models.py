from functools import lru_cache

import docker
import mlflow
from mlflow.entities.model_registry import RegisteredModel

IS_MODEL_LABEL = "org.cogstack.model-serve"
MODEL_URI_LABEL = "org.cogstack.model-serve.uri"
SERVICE_NAME_LABEL = "com.docker.compose.service"


@lru_cache
def get_running_models() -> list[dict]:
    client = docker.from_env()
    containers = client.containers.list(filters={"label": IS_MODEL_LABEL})
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
