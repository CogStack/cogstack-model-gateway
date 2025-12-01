import docker
from docker.models.containers import Container

from cogstack_model_gateway.common.config import get_config
from cogstack_model_gateway.common.containers import PROJECT_NAME_LABEL, SERVICE_NAME_LABEL
from cogstack_model_gateway.common.models import ModelDeploymentType


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


def get_running_models() -> list[dict]:
    """Get a list of running containers corresponding to model servers."""
    config = get_config()
    client = docker.from_env()

    containers = client.containers.list(
        filters={
            "status": "running",
            "label": [
                config.labels.cms_model_label,
                f"{PROJECT_NAME_LABEL}={config.cms.project_name}",
            ],
        }
    )
    return [
        {
            "service_name": c.labels.get(SERVICE_NAME_LABEL, c.name),
            "model_uri": c.labels.get(config.labels.cms_model_uri_label),
            "deployment_type": c.labels.get(config.labels.deployment_type_label)
            or ModelDeploymentType.STATIC.value,
        }
        for c in containers
    ]


def run_model_container(
    model_name: str,
    model_uri: str,
    model_type: str,
    deployment_type: ModelDeploymentType,
    ttl: int = -1,
    resources: dict | None = None,
) -> Container:
    """Run a Docker container for a model server.

    The container is started with the configured CogStack ModelServe image and the specified model
    name, type, and URI. The new CogStack Model Serve instance is given labels to identify it as a
    model server managed by the CogStack Model Gateway, with the specified TTL label determining
    its expiration time.

    Args:
        model_name: Docker service name for the model.
        model_uri: URI pointing to the model artifact (e.g. MLflow model URI).
        model_type: Type of model (e.g., 'medcat_umls', 'medcat_snomed', 'transformers').
        deployment_type: Type of deployment (ModelDeploymentType enum).
        ttl: Fixed time-to-live in seconds (predominantly used for manual deployments).
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

    labels = {
        # The project name is set by Docker when deploying CMS through its compose file. We have to
        # set it explicitly here to ensure that model servers deployed through the gateway can be
        # identified/listed/deleted in the same way as the ones deployed through Docker compose.
        PROJECT_NAME_LABEL: config.cms.project_name,
        config.labels.cms_model_label: model_name,
        config.labels.cms_model_uri_label: model_uri,
        config.labels.ttl_label: str(ttl),
        config.labels.managed_by_label: config.labels.managed_by_value,
        config.labels.deployment_type_label: deployment_type.value,
    }

    base_cmd = "/.venv/bin/python cli/cli.py serve"
    model_type_arg = f"--model-type {model_type}"
    model_name_arg = f"--model-name {model_name}"
    mlflow_uri_arg = f"--mlflow-model-uri {model_uri}"
    host_arg = "--host 0.0.0.0"
    port_arg = f"--port {config.cms.server_port}"

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
        config.cms.image,
        command=[
            "sh",
            "-c",
            f"{base_cmd} {model_type_arg} {model_name_arg} {mlflow_uri_arg} {host_arg} {port_arg}",
        ],
        detach=True,
        environment={
            "CMS_MODEL_NAME": model_name,
            "CMS_MODEL_TYPE": model_type,
            "ENABLE_TRAINING_APIS": str(config.cms.enable_training_apis).lower(),
            "ENABLE_EVALUATION_APIS": str(config.cms.enable_evaluation_apis).lower(),
            "ENABLE_PREVIEWS_APIS": str(config.cms.enable_previews_apis).lower(),
            "AWS_ACCESS_KEY_ID": config.cms.tracking.s3.access_key_id or "",
            "AWS_SECRET_ACCESS_KEY": config.cms.tracking.s3.secret_access_key or "",
            "MLFLOW_S3_ENDPOINT_URL": config.cms.tracking.s3.endpoint_url,
            "MLFLOW_TRACKING_URI": config.cms.tracking.uri,
            "MLFLOW_TRACKING_USERNAME": config.cms.tracking.username,
            "MLFLOW_TRACKING_PASSWORD": config.cms.tracking.password,
            "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": str(
                config.cms.tracking.enable_system_metrics_logging
            ).lower(),
            "GELF_INPUT_URI": config.cms.gelf_input_uri,
            "AUTH_USER_ENABLED": str(config.cms.auth.user_enabled).lower(),
            "AUTH_JWT_SECRET": config.cms.auth.jwt_secret or "",
            "AUTH_ACCESS_TOKEN_EXPIRE_SECONDS": str(config.cms.auth.access_token_expire_seconds),
            "AUTH_DATABASE_URL": config.cms.auth.database_url,
            "HTTP_PROXY": config.cms.proxy.http_proxy or "",
            "HTTPS_PROXY": config.cms.proxy.https_proxy or "",
            "NO_PROXY": config.cms.proxy.no_proxy,
            "http_proxy": config.cms.proxy.http_proxy or "",
            "https_proxy": config.cms.proxy.https_proxy or "",
            "no_proxy": config.cms.proxy.no_proxy,
        },
        labels=labels,
        name=model_name,
        network=config.cms.network,
        volumes={name: {"bind": path, "mode": "rw"} for name, path in config.cms.volumes.items()},
        ports={f"{config.cms.server_port}/tcp": None},
        healthcheck={
            "test": ["CMD", "curl", "-f", f"http://localhost:{config.cms.server_port}/info"],
            "interval": config.cms.health_check.interval * 1000000 * 1000,
            "timeout": config.cms.health_check.timeout * 1000000 * 1000,
            "retries": config.cms.health_check.retries,
            "start_period": config.cms.health_check.start_period * 1000000 * 1000,
        },
        **resource_kwargs,
    )

    return container
