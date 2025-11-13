import os

import docker
from docker.models.containers import Container

from cogstack_model_gateway.common.config import get_config
from cogstack_model_gateway.common.containers import PROJECT_NAME_LABEL, SERVICE_NAME_LABEL
from cogstack_model_gateway.common.models import ModelDeploymentType

CMS_PROJECT_ENV_VAR = "CMS_PROJECT_NAME"
CMS_DOCKER_NETWORK = "cogstack-model-serve_cms"


def _parse_cpus_to_nano(cpus_str: str) -> int:
    """Parse Docker CPU string (e.g., '2.0', '0.5') to nano CPUs.

    Docker API expects CPU limits as nano_cpus (1 CPU = 1e9 nano CPUs).

    Args:
        cpus_str: CPU specification like '2.0', '1.5', '0.5'.

    Returns:
        CPU count in nano CPUs (integer).

    Raises:
        ValueError: If the CPU format is invalid.
    """
    try:
        cpus_float = float(cpus_str)
        if cpus_float <= 0:
            raise ValueError("CPU value must be positive")
        return int(cpus_float * 1e9)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid CPU format: {cpus_str}. Expected a positive number.") from e


def get_running_models(cms_project: str) -> list[dict]:
    """Get a list of running containers corresponding to model servers."""
    config = get_config()
    client = docker.from_env()
    if not cms_project:
        raise ValueError(
            "CogStack ModelServe Docker Compose project name was not provided."
            f" Please try setting the '{CMS_PROJECT_ENV_VAR}' environment variable."
        )

    containers = client.containers.list(
        filters={
            "status": "running",
            "label": [config.labels.cms_model_label, f"{PROJECT_NAME_LABEL}={cms_project}"],
        }
    )
    return [
        {
            "name": c.labels.get(SERVICE_NAME_LABEL, c.name),
            "uri": c.labels.get(config.labels.cms_model_uri_label),
            "deployment_type": c.labels.get(config.labels.deployment_type_label),
        }
        for c in containers
    ]


def run_model_container(
    model_name: str,
    model_uri: str,
    ttl: int,
    cms_project: str,
    deployment_type: ModelDeploymentType,
    resources: dict | None = None,
) -> Container:
    """Run a Docker container for a model server.

    The container is started with the `cogstack-modelserve` image as well as the specified model
    name which is used as the Docker service name and the tracking server URI for the trained model
    to be deployed. The new CogStack Model Serve instance is given labels to identify it as a model
    server managed by the CogStack Model Gateway, with the specified TTL label determining its
    expiration time. Apart from that, it's configured in the same way as the services included in
    the CogStack Model Serve stack.

    Args:
        model_name: Docker service name for the model.
        model_uri: URI pointing to the model artifact (e.g. MLflow model URI).
        ttl: Fixed time-to-live in seconds (predominantly used for manual deployments).
        cms_project: CogStack ModelServe Docker Compose project name.
        deployment_type: Type of deployment (ModelDeploymentType enum).
        resources: Optional resource limits/reservations dict with structure:
            {
                "limits": {"memory": "4g", "cpus": "2.0"},
                "reservations": {"memory": "2g"}
            }

    Returns:
        The created Docker container.
    """
    config = get_config()
    client = docker.from_env()
    if not cms_project:
        raise ValueError(
            "CogStack ModelServe Docker Compose project name was not provided."
            f" Please try setting the '{CMS_PROJECT_ENV_VAR}' environment variable."
        )

    labels = {
        # The project name is set by Docker when deploying CMS through its compose file. We have to
        # set it explicitly here to ensure that model servers deployed through the gateway can be
        # identified/listed/deleted in the same way as the ones deployed through Docker compose.
        PROJECT_NAME_LABEL: cms_project,
        config.labels.cms_model_label: model_name,
        config.labels.cms_model_uri_label: model_uri,
        config.labels.ttl_label: str(ttl),
        config.labels.managed_by_label: config.labels.managed_by_value,
        config.labels.deployment_type_label: deployment_type.value,
    }

    base_cmd = "python cli/cli.py serve"
    model_type_arg = "--model-type medcat_umls"
    model_name_arg = f"--model-name {model_name}"
    mlflow_uri_arg = f"--mlflow-model-uri {model_uri}"
    host_arg = "--host 0.0.0.0"
    port_arg = "--port 8000"

    resource_kwargs = {}
    if resources:
        limits = resources.get("limits", {})
        reservations = resources.get("reservations", {})

        resource_kwargs = {
            **({"mem_limit": limits["memory"]} if limits.get("memory") else {}),
            **({"mem_reservation": reservations["memory"]} if reservations.get("memory") else {}),
            **({"nano_cpus": _parse_cpus_to_nano(limits["cpus"])} if limits.get("cpus") else {}),
        }

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
        **resource_kwargs,
    )

    return container
