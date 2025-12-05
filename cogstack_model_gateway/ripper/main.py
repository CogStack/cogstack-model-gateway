import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from enum import Enum

import docker
from dateutil import parser
from docker.models.containers import Container
from prometheus_client import start_http_server

from cogstack_model_gateway.common.config import Config, load_config
from cogstack_model_gateway.common.containers import PROJECT_NAME_LABEL, SERVICE_NAME_LABEL
from cogstack_model_gateway.common.db import DatabaseManager
from cogstack_model_gateway.common.logging import configure_logging
from cogstack_model_gateway.common.models import ModelDeploymentType, ModelManager
from cogstack_model_gateway.ripper.prometheus.metrics import (
    containers_checked_total,
    containers_purged_total,
    model_idle_time_seconds,
)


class RemovalReason(Enum):
    FIXED_TTL_EXPIRED = "fixed_ttl_expired"
    IDLE_TTL_EXPIRED = "idle_ttl_expired"


log = logging.getLogger("cmg.ripper")


def stop_and_remove_container(
    container: Container,
    model_name: str,
    model_manager: ModelManager,
    deployment_type: ModelDeploymentType,
    reason: RemovalReason,
    idle_time: float | None = None,
):
    """Stop and remove a Docker container and delete its database record."""
    log.info(
        f"Stopping and removing container: {container.name}"
        f" (model={model_name}, deployment_type={deployment_type.value}, reason={reason.value})"
    )

    try:
        container.stop()
        container.remove()
        log.debug(f"Successfully removed container: {container.name}")
    except Exception as e:
        log.error(f"Error removing container {container.name}: {e}")
        raise

    try:
        if model_manager.delete_model(model_name):
            log.debug(f"Deleted model record from database: {model_name}")
        else:
            log.warning(f"Model record not found in database: {model_name}")
    except Exception as e:
        # Skip raising since container is already removed
        log.error(f"Error deleting model record {model_name} from database: {e}")

    containers_purged_total.labels(deployment_type=deployment_type.value, reason=reason.value).inc()

    if reason == RemovalReason.IDLE_TTL_EXPIRED and idle_time is not None:
        model_idle_time_seconds.labels(model=model_name).observe(idle_time)


def should_remove_by_fixed_ttl(container: Container, ttl_label: str) -> bool:
    """Check if a container should be removed based on fixed TTL from labels."""
    ttl = int(container.labels.get(ttl_label, -1))

    if ttl == -1:
        return False  # Skip containers with TTL set to -1

    created_at = parser.isoparse(container.attrs["Created"])
    expiration_time = created_at + timedelta(seconds=ttl)

    return datetime.now(UTC) >= expiration_time


def should_remove_by_idle_ttl(model_name: str, model_manager: ModelManager) -> tuple[bool, float]:
    """Check if a container should be removed based on its idle time."""
    return model_manager.is_model_idle(model_name)


def purge_expired_containers(config: Config):
    """Run periodically and purge Docker containers that have exceeded their TTL.

    List Docker containers and fetch the ones managed by the CogStack Model Gateway that correspond
    to model servers according to their labels, implementing a dual TTL strategy based on deployment
    type:
    - STATIC: These models are not managed by the Gateway and should never be auto-removed (skip)
    - MANUAL: Fixed TTL from container labels (time since creation; containers without a TTL label
      or with a TTL value of -1 are skipped)
    - AUTO: Idle TTL from database records (time since last usage)

    After removing a container, also delete the corresponding model record from database. Finally,
    sleep for the specified interval before repeating the process.
    """
    client = docker.from_env()
    model_manager: ModelManager = config.model_manager

    while True:
        containers = client.containers.list(
            all=True,
            filters={
                "label": [
                    f"{config.labels.managed_by_label}={config.labels.managed_by_value}",
                    config.labels.cms_model_label,
                    f"{PROJECT_NAME_LABEL}={config.cms.project_name}",
                ]
            },
        )
        log.debug(f"Checking {len(containers)} managed containers for expiration")

        with ThreadPoolExecutor() as executor:
            futures = []
            for container in containers:
                container: Container
                model_name = container.labels.get(SERVICE_NAME_LABEL, container.name)

                try:
                    deployment_type = ModelDeploymentType(
                        container.labels.get(
                            config.labels.deployment_type_label, ModelDeploymentType.STATIC.value
                        )
                    )
                    containers_checked_total.labels(deployment_type=deployment_type.value).inc()
                except ValueError:
                    log.warning(
                        f"Unknown deployment type '{deployment_type}' for container"
                        f" {container.name}, skipping removal"
                    )
                    continue

                if deployment_type == ModelDeploymentType.STATIC:
                    log.debug(f"Skipping static deployment: {container.name}")
                    continue

                elif deployment_type == ModelDeploymentType.MANUAL:
                    if should_remove_by_fixed_ttl(container, config.labels.ttl_label):
                        log.info(f"Manual deployment {container.name} exceeded fixed TTL")
                        futures.append(
                            executor.submit(
                                stop_and_remove_container,
                                container,
                                model_name,
                                model_manager,
                                deployment_type,
                                RemovalReason.FIXED_TTL_EXPIRED,
                            )
                        )

                elif deployment_type == ModelDeploymentType.AUTO:
                    should_remove, idle_time = should_remove_by_idle_ttl(model_name, model_manager)
                    if should_remove:
                        log.info(
                            f"Auto deployment {container.name} exceeded idle TTL"
                            f" (idle time ~{idle_time:.0f}s)"
                        )
                        futures.append(
                            executor.submit(
                                stop_and_remove_container,
                                container,
                                model_name,
                                model_manager,
                                deployment_type,
                                RemovalReason.IDLE_TTL_EXPIRED,
                                idle_time,
                            )
                        )

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    log.error(f"Error removing container: {e}")

        time.sleep(config.ripper.interval)


def main():
    """Run the ripper service."""
    configure_logging()
    config = load_config(os.getenv("CONFIG_FILE"))
    config.model_manager = ModelManager(
        db_manager=DatabaseManager(
            user=config.db.user,
            password=config.db.password,
            host=config.db.host,
            port=config.db.port,
            db_name=config.db.name,
        )
    )

    start_http_server(config.ripper.metrics_port)

    log.info(f"Starting ripper with interval={config.ripper.interval}s")
    purge_expired_containers(config)


if __name__ == "__main__":
    sys.exit(main())
