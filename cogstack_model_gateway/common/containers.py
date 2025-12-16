import logging

import docker
from docker.models.containers import Container

from cogstack_model_gateway.common.config import get_config
from cogstack_model_gateway.common.models import ModelDeploymentType

PROJECT_NAME_LABEL = "com.docker.compose.project"
SERVICE_NAME_LABEL = "com.docker.compose.service"

log = logging.getLogger("cmg.common")


def get_models(all: bool = False, managed_only: bool = False) -> list[dict]:
    """Get model containers with filtering.

    Args:
        all: If True, includes paused, restarting, and stopped containers.
            If False, only running (default).
        managed_only: If True, only CMG-managed containers (excludes 'static' deployments).

    Returns:
        List of dicts with: service_name, model_uri, deployment_type, ip_address, container object.

    Raises:
        docker.errors.APIError: If Docker API call fails.
    """
    config = get_config()
    client = docker.from_env()

    containers = client.containers.list(
        all=all,
        filters={
            "label": [
                config.labels.cms_model_label,
                f"{PROJECT_NAME_LABEL}={config.cms.project_name}",
                *(
                    [f"{config.labels.managed_by_label}={config.labels.managed_by_value}"]
                    if managed_only
                    else []
                ),
            ]
        },
    )

    return [
        {
            "service_name": c.labels.get(SERVICE_NAME_LABEL, c.name),
            "model_uri": c.labels.get(config.labels.cms_model_uri_label),
            "deployment_type": (
                c.labels.get(config.labels.deployment_type_label)
                or ModelDeploymentType.STATIC.value
            ),
            "ip_address": c.attrs.get("NetworkSettings", {})
            .get("Networks", {})
            .get(config.cms.network, {})
            .get("IPAddress"),
            "container": c,
        }
        for c in containers
    ]


def stop_and_remove_model_container(container: Container) -> None:
    """Stop and remove a model container using the Docker client.

    Args:
        container: Docker container object to remove.

    Raises:
        docker.errors.APIError: If container removal fails.
    """
    log.info(
        f"Stopping and removing container '{container.name}'"
        f" (id={container.id}, status={container.status})"
    )

    if container.status == "running":
        container.stop()
        log.debug(f"Container {container.name} stopped")

    container.remove()
    log.debug(f"Successfully removed container: {container.name}")
